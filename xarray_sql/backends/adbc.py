"""ADBC engine adapter.

[ADBC](https://arrow.apache.org/adbc/) (Arrow Database Connectivity) is
a database-neutral API whose drivers speak Arrow natively: PostgreSQL,
SQLite, Snowflake, BigQuery, Flight SQL, DuckDB, and more, each behind
the same DBAPI-style ``Connection``. This adapter registers a Dataset on
any of them through ADBC's bulk-ingest call, so one code path reaches
every database with an ADBC driver.

Unlike the DataFusion and DuckDB adapters, registration here *copies*
the data: an ADBC database is typically a separate process or service
that cannot call back into Python to scan a lazy Dataset while a query
runs. The Dataset is streamed chunk by chunk (prefetched, with bounded
memory) into a new table, so only the ingest itself touches the source
data, and queries afterwards run entirely inside the database.

Results come back through ADBC's Arrow cursor API, which
[xarray_sql.to_dataset][] accepts directly::

    cur = con.cursor()
    cur.execute("SELECT time, lat, lon, t2m FROM era5 ORDER BY time")
    out = xql.to_dataset(cur, template=ds)

This adapter never imports ``adbc_driver_manager`` at runtime —
detection is by connection type — so ADBC stays a purely optional
dependency (``pip install xarray-sql[adbc]`` plus a driver package).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

import pyarrow as pa
import xarray as xr

from ..df import (
    Chunks,
    TableNames,
    group_vars_by_dims,
    resolve_table_names,
    shared_coord_arrays,
)
from .base import register_adapter
from .pyarrow import XarrayPushdownDataset

if TYPE_CHECKING:
    from adbc_driver_manager import dbapi

__all__ = ["ADBCAdapter"]

IngestMode = Literal["create", "append", "replace", "create_append"]


def _quote(identifier: str) -> str:
    """Render *identifier* as a quoted SQL identifier."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _is_clickhouse(con: dbapi.Connection) -> bool:
    """Whether *con* is connected to ClickHouse.

    Drivers name their database in ``adbc_get_info``. The ClickHouse
    driver does not implement it, so only a driver without it is probed
    with a query against ClickHouse's ``system.one`` table.
    """
    try:
        info = con.adbc_get_info()
    except Exception:  # noqa: BLE001 — unimplemented; probe instead
        pass
    else:
        return str(info.get("vendor_name", "")).lower() == "clickhouse"
    try:
        with con.cursor() as cur:
            cur.execute("SELECT 1 FROM system.one")
            cur.fetchall()
    except Exception:  # noqa: BLE001
        return False
    return True


_CLICKHOUSE_TYPES = {
    pa.bool_(): "Bool",
    pa.int8(): "Int8",
    pa.int16(): "Int16",
    pa.int32(): "Int32",
    pa.int64(): "Int64",
    pa.uint8(): "UInt8",
    pa.uint16(): "UInt16",
    pa.uint32(): "UInt32",
    pa.uint64(): "UInt64",
    pa.float16(): "Float32",
    pa.float32(): "Float32",
    pa.float64(): "Float64",
    pa.string(): "String",
    pa.large_string(): "String",
    pa.binary(): "String",
    pa.large_binary(): "String",
    pa.date32(): "Date32",
}

_TIMESTAMP_PRECISION = {"s": 0, "ms": 3, "us": 6, "ns": 9}


def _clickhouse_type(field: pa.Field, key: bool) -> str:
    """The ClickHouse column type for an Arrow field.

    Timestamps without a zone are declared UTC, which is what their
    values mean; ClickHouse also parses string literals compared with a
    column in that column's zone, so ``time >= '2020-01-01'`` means UTC
    rather than the server's local time. Sort-key columns and floats
    (which carry NaN) are not ``Nullable``.
    """
    arrow_type = field.type
    if pa.types.is_timestamp(arrow_type):
        precision = _TIMESTAMP_PRECISION[arrow_type.unit]
        zone = arrow_type.tz or "UTC"
        name = f"DateTime64({precision}, '{zone}')"
    elif arrow_type in _CLICKHOUSE_TYPES:
        name = _CLICKHOUSE_TYPES[arrow_type]
    else:
        raise TypeError(
            f"no ClickHouse column type for {field.name!r} of Arrow type "
            f"{arrow_type}; create the table yourself and register with "
            f'mode="append"'
        )
    if key or pa.types.is_floating(arrow_type) or not field.nullable:
        return name
    return f"Nullable({name})"


def _clickhouse_ddl(
    table: str,
    schema: pa.Schema,
    dims: tuple[str, ...],
    *,
    mode: IngestMode,
    temporary: bool,
    database: str | None,
) -> list[str]:
    """Statements that prepare *table* for an append-mode ingest.

    ClickHouse's ADBC driver only appends, so the table is created here
    for every other mode. Tables are sorted by their dimensions, so
    ClickHouse's primary index skips data on dimension predicates the
    way chunk pruning does in the other engines.
    """
    if mode == "append":
        return []
    target = _quote(table)
    if database is not None:
        target = f"{_quote(database)}.{target}"
    columns = ", ".join(
        f"{_quote(field.name)} {_clickhouse_type(field, field.name in dims)}"
        for field in schema
    )
    kind = "TEMPORARY TABLE" if temporary else "TABLE"
    if temporary:
        engine = "ENGINE = Memory"
    else:
        order = ", ".join(_quote(dim) for dim in dims) or "tuple()"
        engine = f"ENGINE = MergeTree ORDER BY ({order})"
    statements = []
    if mode == "replace":
        statements.append(f"DROP {kind} IF EXISTS {target}")
    exists = " IF NOT EXISTS" if mode == "create_append" else ""
    statements.append(f"CREATE {kind}{exists} {target} ({columns}) {engine}")
    return statements


def _ingest(
    con: dbapi.Connection,
    table: str,
    ds: xr.Dataset,
    chunks: Chunks,
    *,
    dims: tuple[str, ...],
    mode: IngestMode,
    temporary: bool,
    clickhouse: bool,
    db_schema_name: str | None = None,
    **kwargs: Any,
) -> None:
    """Stream *ds* into *table* through ADBC bulk ingest.

    The full scan of an
    [XarrayPushdownDataset][xarray_sql.backends.pyarrow.XarrayPushdownDataset]
    prefetches chunks on a thread pool while the driver writes earlier
    batches, so the source read and the database write overlap.
    """
    dataset = XarrayPushdownDataset(ds, chunks, **kwargs)
    if clickhouse:
        for statement in _clickhouse_ddl(
            table,
            dataset.schema,
            dims,
            mode=mode,
            temporary=temporary,
            database=db_schema_name,
        ):
            with con.cursor() as cur:
                cur.execute(statement)
        mode, temporary = "append", False
    reader = dataset.scanner().to_reader()
    with con.cursor() as cur:
        cur.adbc_ingest(
            table,
            reader,
            mode=mode,
            db_schema_name=db_schema_name,
            temporary=temporary,
        )


def _schema_exists(con: dbapi.Connection, name: str) -> bool:
    """Whether the database already has a schema named exactly *name*."""
    try:
        objects = con.adbc_get_objects(
            depth="db_schemas", db_schema_filter=name
        ).read_all()
    except Exception:  # noqa: BLE001 — metadata unsupported; assume not
        return False
    return any(
        schema["db_schema_name"] == name
        for catalog in objects.to_pylist()
        for schema in catalog["catalog_db_schemas"] or []
    )


def _connection_usable(con: dbapi.Connection) -> bool:
    """Whether *con* still runs statements after one of them failed."""
    try:
        with con.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchall()
    except Exception:  # noqa: BLE001
        return False
    return True


def _create_schema(
    con: dbapi.Connection, name: str, *, clickhouse: bool = False
) -> bool:
    """Ensure the database schema *name* exists; whether it does.

    An existing schema is used as is: creating it can need privileges on
    the whole database (PostgreSQL checks them even for ``IF NOT
    EXISTS``) that a role granted only that schema lacks.

    Not every ADBC database has schemas (SQLite does not), and creating
    one can fail for lack of privileges. When the connection survives the
    failure, the caller falls back to flat table names. On databases
    where a failed statement aborts the transaction (PostgreSQL), nothing
    after it could run, so this raises instead of falling back. In
    ClickHouse, a database plays the role of a schema.
    """
    if _schema_exists(con, name):
        return True
    kind = "DATABASE" if clickhouse else "SCHEMA"
    try:
        with con.cursor() as cur:
            cur.execute(f"CREATE {kind} IF NOT EXISTS {_quote(name)}")
    except Exception as exc:
        if not _connection_usable(con):
            raise RuntimeError(
                f"Could not create the {name!r} schema to hold the dimension "
                f"groups of {name!r} ({exc}). The failure aborted the "
                f"connection's transaction, so call con.rollback() before "
                f"using it again. Create the schema beforehand (or grant "
                f"the privilege to), or pass temporary=True to register "
                f"flat {name}_<group> tables instead."
            ) from exc
        warnings.warn(
            f"Could not create the {name!r} schema to hold the dimension "
            f"groups of {name!r} ({exc}); registering them as flat "
            f"{name}_<group> tables instead.",
            RuntimeWarning,
            stacklevel=4,
        )
        return False
    return True


@register_adapter
class ADBCAdapter:
    """Registers Datasets on ADBC DBAPI connections."""

    @staticmethod
    def matches(con: object) -> TypeGuard[dbapi.Connection]:
        # Every driver's ``dbapi.connect`` returns this class or a
        # subclass of it (e.g. ``adbc_driver_sqlite.dbapi``'s).
        return any(
            cls.__module__ == "adbc_driver_manager.dbapi"
            and cls.__qualname__ == "Connection"
            for cls in type(con).__mro__
        )

    @staticmethod
    def register(
        con: dbapi.Connection,
        name: str,
        ds: xr.Dataset,
        *,
        chunks: Chunks = None,
        table_names: TableNames = None,
        mode: IngestMode = "create",
        temporary: bool = False,
        **kwargs: Any,
    ) -> dbapi.Connection:
        """Ingest ``ds`` into tables on an ADBC connection.

        Datasets whose variables all share the same dimensions become a
        single table named ``name``. Mixed-dimension datasets are split
        into one table per dimension group, created in a database schema
        named ``name`` so they are queried as ``name.group`` — the same
        spelling every other engine uses::

            xql.register(con, 'era5', ds, table_names={
                ('time', 'latitude', 'longitude'): 'surface',
                ('time', 'level', 'latitude', 'longitude'): 'atmosphere',
            })
            cur.execute('SELECT ... FROM era5.surface')

        An existing schema is used as is. On databases without schemas
        (SQLite), and for temporary tables, which most drivers cannot
        place in a schema, the groups are created as flat
        ``name_group`` tables instead. If creating the schema fails in a
        way that aborts the transaction (PostgreSQL without the
        ``CREATE`` privilege), this raises ``RuntimeError``; roll back,
        then create the schema beforehand or pass ``temporary=True``.

        On ClickHouse, whose driver can only append, the adapter creates
        the tables itself: ``MergeTree`` tables sorted by their
        dimensions (``Memory`` for temporary ones), in a ClickHouse
        database named ``name`` for a mixed-dimension Dataset, with
        timestamps declared ``DateTime64(9, 'UTC')``.

        Registration runs inside the connection's current transaction:
        the tables are visible to this connection immediately, and to
        others once you call ``con.commit()`` (unless the connection is
        in autocommit mode).

        Args:
            mode: What to do when a table already exists, as in ADBC's
                ``adbc_ingest``: ``"create"`` (default) raises,
                ``"replace"`` drops and recreates it, ``"append"`` and
                ``"create_append"`` add rows to it.
            temporary: Create temporary tables, which the database drops
                when the connection closes.
            **kwargs: Forwarded to
                [XarrayPushdownDataset][xarray_sql.backends.pyarrow.XarrayPushdownDataset]
                (``batch_size``, ``prefetch``, ``prefetch_bytes``,
                ``coalesce_rows``) to tune the ingest scan.
        """
        groups = group_vars_by_dims(ds)
        names = resolve_table_names(ds, table_names, case_insensitive=True)
        clickhouse = _is_clickhouse(con)
        if len(groups) <= 1:
            _ingest(
                con,
                name,
                ds,
                chunks,
                dims=next(iter(groups), ()),
                mode=mode,
                temporary=temporary,
                clickhouse=clickhouse,
                **kwargs,
            )
            return con

        in_schema = not temporary and _create_schema(
            con, name, clickhouse=clickhouse
        )
        coord_arrays = shared_coord_arrays(ds)
        for dims, var_names in groups.items():
            group = names[dims]
            _ingest(
                con,
                group if in_schema else f"{name}_{group}",
                ds[var_names],
                chunks,
                dims=dims,
                mode=mode,
                temporary=temporary,
                clickhouse=clickhouse,
                db_schema_name=name if in_schema else None,
                coord_arrays=coord_arrays,
                **kwargs,
            )
        return con
