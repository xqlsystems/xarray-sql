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

import contextlib
import warnings
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, TypeGuard

import xarray as xr

from ..df import (
    Chunks,
    TableNames,
    group_vars_by_dims,
    resolve_table_names,
    shared_coord_arrays,
)
from ._adbc_dialects import (
    Dialect,
    IngestMode,
    dialect_for,
    ingestable,
)
from .base import register_adapter
from .pyarrow import XarrayPushdownDataset

if TYPE_CHECKING:
    from adbc_driver_manager import dbapi

__all__ = ["ADBCAdapter"]


def _ingest(
    con: dbapi.Connection,
    table: str,
    ds: xr.Dataset,
    chunks: Chunks,
    *,
    dialect: Dialect,
    dims: tuple[str, ...],
    mode: IngestMode,
    temporary: bool,
    ingest_options: Mapping[str, str] | None,
    db_schema_name: str | None = None,
    **kwargs: Any,
) -> None:
    """Stream *ds* into *table* through ADBC bulk ingest.

    The full scan of an
    [XarrayPushdownDataset][xarray_sql.backends.pyarrow.XarrayPushdownDataset]
    prefetches chunks on a thread pool while the driver writes earlier
    batches, so the source read and the database write overlap.
    """
    reader = XarrayPushdownDataset(ds, chunks, **kwargs).scanner().to_reader()
    reader = ingestable(reader, dialect)
    if dialect.table_ddl is not None:
        for statement in dialect.table_ddl(
            table,
            reader.schema,
            dims,
            mode=mode,
            temporary=temporary,
            database=db_schema_name,
        ):
            with con.cursor() as cur:
                cur.execute(statement)
        mode, temporary = "append", False
    target_schema = db_schema_name
    in_database: contextlib.AbstractContextManager[None] = (
        contextlib.nullcontext()
    )
    if (
        db_schema_name is not None
        and dialect.target_schema == "default_database"
    ):
        in_database = _default_database(con, db_schema_name, dialect)
        target_schema = None
    with in_database, con.cursor() as cur:
        if ingest_options:
            cur.adbc_statement.set_options(**ingest_options)
        cur.adbc_ingest(
            table,
            reader,
            mode=mode,
            db_schema_name=target_schema,
            temporary=temporary,
        )
    if dialect.analyze:
        target = dialect.quote_identifier(table)
        if db_schema_name is not None:
            target = f"{dialect.quote_identifier(db_schema_name)}.{target}"
        with con.cursor() as cur:
            cur.execute(f"ANALYZE {target}")


@contextlib.contextmanager
def _default_database(
    con: dbapi.Connection, database: str, dialect: Dialect
) -> Iterator[None]:
    """Make *database* the connection's default while in the block.

    For drivers that create an ingest's table in the default database
    whatever target schema they are given. The previous default is
    restored afterwards; MySQL cannot unset a default database, so a
    connection that had none keeps *database*.
    """
    with con.cursor() as cur:
        cur.execute("SELECT DATABASE()")
        (previous,) = cur.fetchone()
        cur.execute(f"USE {dialect.quote_identifier(database)}")
    try:
        yield
    finally:
        if previous is not None:
            with con.cursor() as cur:
                cur.execute(f"USE {dialect.quote_identifier(previous)}")


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


def _warn_on_lost_precision(ds: xr.Dataset, dialect: Dialect) -> None:
    """Warn if *dialect* would truncate a time coordinate of *ds*."""
    if dialect.timestamp_unit == "ns":
        return
    for name, coord in ds.coords.items():
        if coord.dtype.kind != "M":
            continue
        values = coord.values.astype("datetime64[ns]")
        if (values.astype("datetime64[us]") != values).any():
            warnings.warn(
                f"{dialect.name} stores times to the microsecond, so the "
                f"sub-microsecond part of {name!r} will be truncated.",
                RuntimeWarning,
                stacklevel=4,
            )


def _warn_on_folded_names(names: list[str], dialect: Dialect) -> None:
    """Warn about names *dialect*'s database would fold if unquoted."""
    if dialect.folds is None:
        return
    fold = str.lower if dialect.folds == "lower" else str.upper
    folded = [n for n in names if fold(n) != n]
    if folded:
        quoted = ", ".join(dialect.quote_identifier(n) for n in folded)
        warnings.warn(
            f"{dialect.name} folds unquoted names to {dialect.folds}case, "
            f"so quote these in queries: {quoted}.",
            RuntimeWarning,
            stacklevel=4,
        )


def _flat(name: str, reason: str) -> bool:
    """Warn that *name*'s groups become flat tables; always ``False``."""
    warnings.warn(
        f"Registering the dimension groups of {name!r} as flat "
        f"{name}_<group> tables: {reason}.",
        RuntimeWarning,
        stacklevel=4,
    )
    return False


def _create_schema(con: dbapi.Connection, name: str, dialect: Dialect) -> bool:
    """Ensure the database schema *name* exists; whether it does.

    An existing schema is used as is: creating it can need privileges on
    the whole database (PostgreSQL checks them even for ``IF NOT
    EXISTS``) that a role granted only that schema lacks.

    Creating one can fail for lack of privileges. When the connection
    survives the failure, the caller falls back to flat table names. On
    databases where a failed statement aborts the transaction
    (PostgreSQL), nothing after it could run, so this raises instead.
    """
    if not dialect.schemas:
        return _flat(name, f"{dialect.name} has no schemas to hold them")
    if _schema_exists(con, name):
        return True
    try:
        with con.cursor() as cur:
            cur.execute(dialect.create_schema_sql(name))
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
        return _flat(name, f"could not create the {name!r} schema ({exc})")
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
        ingest_options: Mapping[str, str] | None = None,
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
        timestamps declared ``DateTime64(p, 'UTC')`` at the coordinate's
        precision (``p`` is 9 for ``datetime64[ns]``).

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
                when the connection closes. Raises ``ValueError`` on
                databases whose driver cannot (DataFusion, Trino, Spark,
                BigQuery, Databricks, Snowflake).
            ingest_options: Driver-specific statement options set on each
                ingest, e.g. Spark's
                ``{"spark.ingest.staging_area_uri": "s3://bucket/path"}``.
            **kwargs: Forwarded to
                [XarrayPushdownDataset][xarray_sql.backends.pyarrow.XarrayPushdownDataset]
                (``batch_size``, ``prefetch``, ``prefetch_bytes``,
                ``coalesce_rows``) to tune the ingest scan.
        """
        groups = group_vars_by_dims(ds)
        names = resolve_table_names(ds, table_names, case_insensitive=True)
        dialect = dialect_for(con)
        if temporary and not dialect.temporary_tables:
            raise ValueError(
                f"{dialect.name}'s ADBC driver does not support temporary "
                f"tables; register without temporary=True and drop the "
                f"tables when done."
            )
        _warn_on_lost_precision(ds, dialect)
        _warn_on_folded_names(
            [name] if len(groups) <= 1 else [name, *names.values()], dialect
        )
        if len(groups) <= 1:
            _ingest(
                con,
                name,
                ds,
                chunks,
                dialect=dialect,
                dims=next(iter(groups), ()),
                mode=mode,
                temporary=temporary,
                ingest_options=ingest_options,
                **kwargs,
            )
            return con

        in_schema = not temporary and _create_schema(con, name, dialect)
        coord_arrays = shared_coord_arrays(ds)
        for dims, var_names in groups.items():
            group = names[dims]
            _ingest(
                con,
                group if in_schema else f"{name}_{group}",
                ds[var_names],
                chunks,
                dialect=dialect,
                dims=dims,
                mode=mode,
                temporary=temporary,
                ingest_options=ingest_options,
                db_schema_name=name if in_schema else None,
                coord_arrays=coord_arrays,
                **kwargs,
            )
        return con
