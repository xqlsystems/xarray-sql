"""DuckDB engine adapter.

Registers a lazy ``xarray.Dataset`` on a ``duckdb.DuckDBPyConnection``
as an [XarrayPushdownDataset][xarray_sql.backends.pyarrow.XarrayPushdownDataset]:
DuckDB classifies it with a real ``isinstance`` check against
``pyarrow.dataset.Dataset`` and calls ``scanner(columns=[...],
filter=<pyarrow.compute.Expression>)`` once per query, giving
projection pushdown, coordinate-range chunk pruning, and prefetched
parallel production (see [xarray_sql.backends.pyarrow][]).

This adapter never imports the ``duckdb`` package at runtime — detection
is by connection type, and registration is a method call on the
connection — so DuckDB stays a purely optional dependency
(``pip install xarray-sql[duckdb]``).

Zarr-native scanning inside DuckDB is what the [duckdb-zarr](https://github.com/xqlsystems/duckdb-zarr) extension provides; this
adapter instead covers everything xarray can open (NetCDF, GRIB, Xee, CF
decoding, in-memory) and pairs with [xarray_sql.to_dataset][] for
the labeled round-trip.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TypeGuard

import xarray as xr

from ..df import (
    Chunks,
    TableNames,
    group_vars_by_dims,
    resolve_table_names,
    shared_coord_arrays,
)
from .base import register_adapter
from .pyarrow import XarrayArrowStream, XarrayPushdownDataset

if TYPE_CHECKING:
    import duckdb

__all__ = ["DuckDBAdapter", "XarrayArrowStream", "XarrayPushdownDataset"]


def _quote(identifier: str) -> str:
    """Render *identifier* as a quoted SQL identifier."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _is_in_memory(con: duckdb.DuckDBPyConnection) -> bool:
    """Whether every database attached to *con* is memory-backed.

    ``con.register`` binds a Python object to the connection only for
    its lifetime — never to the catalog on disk. A view created over
    that binding is ordinary catalog DDL, so on a file-backed database
    it persists after the flat table it selects from is gone, and a
    later connection resolves it into "Table ... does not exist".
    In-memory databases have nothing to outlive, so mirroring there is
    safe.
    """
    try:
        rows = con.execute("PRAGMA database_list").fetchall()
    except Exception:  # noqa: BLE001 — conservative: assume persistent
        return False
    return all(path is None for _, _, path in rows)


def _mirror_as_schema(
    con: duckdb.DuckDBPyConnection, name: str, tables: dict[str, str]
) -> None:
    """Expose flat tables as views in a DuckDB schema named *name*.

    This translates a view to `name_group` can be queried as `name.group`
    in DuckDB.

    *tables* maps each group's table name to the flat name it was
    registered under.
    """
    if not _is_in_memory(con):
        warnings.warn(
            f"Registered the dimension groups of {name!r} as "
            f"{', '.join(sorted(tables.values()))}, but skipped mirroring "
            f"them as the {name!r} schema: this is a file-backed "
            f"connection, and {name}.<group> views would persist in it "
            f"after the flat tables they select from — registered only "
            f"for this connection — are gone. Query the flat table names "
            f"instead, or use an in-memory connection for the dotted "
            f"spelling.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote(name)}")
        existing = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM duckdb_tables() "
                "WHERE schema_name = ? AND NOT internal",
                [name],
            ).fetchall()
        }
        conflicts = set(tables) & existing
        if conflicts:
            raise ValueError(
                f"the {name!r} schema already has real tables named "
                f"{sorted(conflicts)} that were not registered by "
                f"xarray-sql; refusing to replace them with views"
            )
        for group, flat in tables.items():
            con.execute(
                f"CREATE OR REPLACE VIEW {_quote(name)}.{_quote(group)} "
                f"AS SELECT * FROM {_quote(flat)}"
            )
    except Exception as exc:  # noqa: BLE001 — degrade, don't fail the register
        # Creating a schema needs a writable catalog, and the conflict
        # check above can itself raise; either is not a reason to lose
        # the registration.
        warnings.warn(
            f"Registered the dimension groups of {name!r} as "
            f"{', '.join(sorted(tables.values()))}, but could not create "
            f"the {name!r} schema mirroring them as {name}.<group> "
            f"({exc}). Query the flat table names instead.",
            RuntimeWarning,
            stacklevel=3,
        )


@register_adapter
class DuckDBAdapter:
    """Registers Datasets on ``duckdb.DuckDBPyConnection`` connections."""

    @staticmethod
    def matches(con: object) -> TypeGuard[duckdb.DuckDBPyConnection]:
        # The connection class lives in ``duckdb`` or, in newer releases,
        # the ``_duckdb`` C-extension module.
        root = type(con).__module__.split(".")[0]
        return root in ("duckdb", "_duckdb")

    @staticmethod
    def register(
        con: duckdb.DuckDBPyConnection,
        name: str,
        ds: xr.Dataset,
        *,
        chunks: Chunks = None,
        table_names: TableNames = None,
        **kwargs: Any,
    ) -> duckdb.DuckDBPyConnection:
        """Register ``ds`` on a DuckDB connection.

        Datasets whose variables all share the same dimensions become a
        single table named ``name``. Mixed-dimension datasets are split
        into one table per dimension group, named after the group's
        dimensions joined by underscores unless ``table_names`` gives the
        group a name of its own::

            xql.register(con, 'era5', ds, table_names={
                ('time', 'latitude', 'longitude'): 'surface',
                ('time', 'level', 'latitude', 'longitude'): 'atmosphere',
            })
            con.sql('SELECT ... FROM era5.surface')   # or era5_surface

        Each group is registered under the flat name ``<name>_<group>``
        and mirrored as a view ``<name>.<group>`` in a DuckDB schema, so
        the dotted spelling the DataFusion adapter uses queries the same
        table here. Extra keyword arguments (``batch_size``,
        ``prefetch``) are forwarded to
        [XarrayPushdownDataset][xarray_sql.backends.pyarrow.XarrayPushdownDataset].
        """
        groups = group_vars_by_dims(ds)
        names = resolve_table_names(ds, table_names, case_insensitive=True)
        if len(groups) <= 1:
            con.register(name, XarrayPushdownDataset(ds, chunks, **kwargs))
            return con
        # Materialise dim coordinates once and share across sub-tables.
        coord_arrays = shared_coord_arrays(ds)
        flat_names = {}
        for dims, var_names in groups.items():
            group = names[dims]
            flat_names[group] = f"{name}_{group}"
            con.register(
                flat_names[group],
                XarrayPushdownDataset(
                    ds[var_names],
                    chunks,
                    coord_arrays=coord_arrays,
                    **kwargs,
                ),
            )
        _mirror_as_schema(con, name, flat_names)
        return con
