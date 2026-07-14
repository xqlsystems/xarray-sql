"""Engine-adapter dispatch for :func:`xarray_sql.register`.

An *engine adapter* implements one seam: given an engine's native
connection object and a lazy ``xarray.Dataset``, register the Dataset as
a queryable table on that connection. The Arrow C-stream protocol is the
common wire between xarray and every engine; adapters differ only in how
a stream is attached to the connection and in what pushdown the engine
can do against it.

Adapters self-describe which connections they accept via ``matches``,
which must not require the engine's package to be importable (detection
is by type inspection), so optional engines stay optional.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import xarray as xr

from ..df import Chunks


@runtime_checkable
class EngineAdapter(Protocol):
    """One engine's implementation of the register seam."""

    @staticmethod
    def matches(con: Any) -> bool:
        """Whether *con* is a connection this adapter can register into."""
        ...

    @staticmethod
    def register(
        con: Any,
        name: str,
        ds: xr.Dataset,
        *,
        chunks: Chunks = None,
        **kwargs: Any,
    ) -> Any:
        """Register *ds* as table *name* on *con*; returns *con*."""
        ...

    @staticmethod
    def run_sql(con: Any, sql: str) -> None:
        """Execute a SQL statement on *con* for its side effects.

        Used by cross-engine helpers (:func:`xarray_sql.materialize`,
        the documented caching and pyramid recipes) that issue DDL/DML in the engine's
        own dialect.
        """
        ...


_ADAPTERS: list[type[EngineAdapter]] = []


def register_adapter(cls: type) -> type:
    """Class decorator adding an adapter to the dispatch list."""
    _ADAPTERS.append(cls)
    return cls


def get_adapter(con: Any) -> type[EngineAdapter]:
    """Return the first adapter whose ``matches(con)`` is true."""
    for adapter in _ADAPTERS:
        if adapter.matches(con):
            return adapter
    raise TypeError(
        f"No xarray-sql engine adapter for connection of type "
        f"{type(con).__module__}.{type(con).__qualname__}. "
        f"Supported: DataFusion SessionContext and DuckDB connections."
    )


def register(
    con: Any,
    name: str,
    ds: xr.Dataset,
    *,
    chunks: Chunks = None,
    **kwargs: Any,
) -> Any:
    """Register a lazy xarray Dataset as a table on an engine connection.

    The engine is inferred from the connection type. Data is not read at
    registration time; the engine pulls Arrow record batches lazily during
    query execution. Write your SQL in the engine's own dialect and use
    the engine's extension ecosystem directly — xarray-sql translates the
    data, not the queries.

    Example (DuckDB)::

        import duckdb
        import xarray_sql as xql

        con = duckdb.connect()
        xql.register(con, "era5", ds)
        rel = con.sql("SELECT time, AVG(t2m) AS t2m FROM era5 GROUP BY time")
        result = xql.to_dataset(rel, template=ds)

    Args:
        con: An engine connection: a ``datafusion.SessionContext`` (or
            :class:`xarray_sql.XarrayContext`) or a
            ``duckdb.DuckDBPyConnection``.
        name: The table name to register the Dataset under. Datasets
            whose variables have differing dimensions are split into one
            table per dimension group (a SQL schema ``name.group`` on
            DataFusion; ``name_group`` tables on DuckDB).
        ds: An xarray Dataset.
        chunks: Xarray-like chunks specification controlling partition
            granularity. Defaults to the Dataset's existing chunks.
        **kwargs: Adapter-specific options, forwarded as-is — e.g.
            ``table_names`` on DataFusion, ``batch_size`` / ``prefetch``
            on DuckDB.

    Returns:
        The connection, to allow chaining.
    """
    return get_adapter(con).register(con, name, ds, chunks=chunks, **kwargs)
