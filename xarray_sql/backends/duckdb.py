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

from typing import TYPE_CHECKING, Any, TypeGuard

import xarray as xr

from ..df import Chunks, group_vars_by_dims
from .base import register_adapter
from .pyarrow import XarrayArrowStream, XarrayPushdownDataset

if TYPE_CHECKING:
    import duckdb

__all__ = ["DuckDBAdapter", "XarrayArrowStream", "XarrayPushdownDataset"]


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
        **kwargs: Any,
    ) -> duckdb.DuckDBPyConnection:
        """Register ``ds`` on a DuckDB connection.

        Datasets whose variables all share the same dimensions become a
        single table named ``name``. Mixed-dimension datasets are split
        into one table per dimension group, named
        ``<name>_<dim1>_<dim2>_...`` (DuckDB registration has no schema
        namespace to mirror the DataFusion adapter's ``name.group``
        layout). Extra keyword arguments (``batch_size``, ``prefetch``)
        are forwarded to [XarrayPushdownDataset][xarray_sql.backends.pyarrow.XarrayPushdownDataset].
        """
        groups = group_vars_by_dims(ds)
        if len(groups) <= 1:
            con.register(name, XarrayPushdownDataset(ds, chunks, **kwargs))
            return con
        # Materialise dim coordinates once and share across sub-tables.
        coord_arrays = {
            str(dim): ds.coords[dim].values
            for dim in ds.dims
            if dim in ds.coords
        }
        for dims, var_names in groups.items():
            suffix = "_".join(dims) or "scalar"
            con.register(
                f"{name}_{suffix}",
                XarrayPushdownDataset(
                    ds[var_names],
                    chunks,
                    coord_arrays=coord_arrays,
                    **kwargs,
                ),
            )
        return con
