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


def _ingest(
    con: dbapi.Connection,
    table: str,
    ds: xr.Dataset,
    chunks: Chunks,
    *,
    mode: IngestMode,
    temporary: bool,
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


def _create_schema(con: dbapi.Connection, name: str) -> bool:
    """Ensure the database schema *name* exists; whether it does.

    An existing schema is used as is: creating it can need privileges on
    the whole database (PostgreSQL checks them even for ``IF NOT
    EXISTS``) that a role granted only that schema lacks.

    Not every ADBC database has schemas (SQLite does not), and creating
    one can fail for lack of privileges. When the connection survives the
    failure, the caller falls back to flat table names. On databases
    where a failed statement aborts the transaction (PostgreSQL), nothing
    after it could run, so this raises instead of falling back.
    """
    if _schema_exists(con, name):
        return True
    try:
        with con.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote(name)}")
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
        if len(groups) <= 1:
            _ingest(
                con,
                name,
                ds,
                chunks,
                mode=mode,
                temporary=temporary,
                **kwargs,
            )
            return con

        in_schema = not temporary and _create_schema(con, name)
        coord_arrays = shared_coord_arrays(ds)
        for dims, var_names in groups.items():
            group = names[dims]
            _ingest(
                con,
                group if in_schema else f"{name}_{group}",
                ds[var_names],
                chunks,
                mode=mode,
                temporary=temporary,
                db_schema_name=name if in_schema else None,
                coord_arrays=coord_arrays,
                **kwargs,
            )
        return con
