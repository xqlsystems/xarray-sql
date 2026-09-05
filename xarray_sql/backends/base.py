"""Engine-adapter dispatch for [xarray_sql.register][].

An *engine adapter* implements the register seam: given an engine's native
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

from typing import Any, Protocol, TypeGuard, TypeVar, cast

import xarray as xr

from ..df import Chunks, TableNames

ConT = TypeVar("ConT")
"""An engine's native connection type (e.g. ``duckdb.DuckDBPyConnection``)."""


class EngineAdapter(Protocol[ConT]):
    """One engine's implementation of the register seam."""

    @staticmethod
    def matches(con: object) -> TypeGuard[ConT]:
        """Whether *con* is a connection this adapter can register into."""
        ...

    @staticmethod
    def register(
        con: ConT,
        name: str,
        ds: xr.Dataset,
        *,
        chunks: Chunks = None,
        table_names: TableNames = None,
        **kwargs: Any,
    ) -> ConT:
        """Register *ds* as table *name* on *con*; returns *con*.

        ``table_names`` names the per-dimension-group tables a
        mixed-dimension Dataset splits into (see
        [resolve_table_names][xarray_sql.df.resolve_table_names]);
        adapters that split must honour it.
        """
        ...


_ADAPTERS: list[type[EngineAdapter[Any]]] = []

_A = TypeVar("_A", bound=type[EngineAdapter[Any]])


def register_adapter(cls: _A) -> _A:
    """Class decorator adding an adapter to the dispatch list."""
    _ADAPTERS.append(cls)
    return cls


def get_adapter(con: object) -> type[EngineAdapter[Any]]:
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
    con: ConT,
    name: str,
    ds: xr.Dataset,
    *,
    chunks: Chunks = None,
    table_names: TableNames = None,
    **kwargs: Any,
) -> ConT:
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

    A Dataset whose variables sit on different dimensions is split into
    one table per dimension group. Name those tables with
    ``table_names``, and the same SQL runs on every engine::

        xql.register(con, "era5", ds, table_names={
            ("time", "latitude", "longitude"): "surface",
            ("time", "level", "latitude", "longitude"): "atmosphere",
        })
        con.sql("SELECT AVG(temperature) FROM era5.atmosphere")

    Args:
        con: An engine connection: a ``datafusion.SessionContext`` (or
            [xarray_sql.XarrayContext][]) or a
            ``duckdb.DuckDBPyConnection``.
        name: The table name to register the Dataset under. Datasets
            whose variables have differing dimensions are split into one
            table per dimension group, addressed as ``name.group`` on
            every engine (DuckDB also keeps the flat ``name_group``
            spelling, since its registration namespace is flat).
        ds: An xarray Dataset.
        chunks: Xarray-like chunks specification controlling partition
            granularity. Defaults to the Dataset's existing chunks.
        table_names: Maps a dimension group's exact dim tuple to the name
            its table takes. Groups left unnamed take their dimensions
            joined by underscores (``time_latitude_longitude``); the
            group holding scalar variables, if any, takes ``scalar``.
            Keys matching no group in ``ds`` are ignored, so one naming
            map can be reused across Datasets holding different subsets
            of the same variables.
        **kwargs: Adapter-specific options, forwarded as-is — e.g.
            ``batch_size`` / ``prefetch`` on DuckDB.

    Returns:
        The connection, to allow chaining.
    """
    # The connection type is erased by the runtime dispatch; every adapter
    # returns the connection it was given.
    adapter: Any = get_adapter(con)
    return cast(
        ConT,
        adapter.register(
            con, name, ds, chunks=chunks, table_names=table_names, **kwargs
        ),
    )
