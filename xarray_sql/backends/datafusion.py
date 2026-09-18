"""DataFusion engine adapter.

DataFusion is xarray-sql's default engine and the richest adapter: the
Rust ``LazyArrowStreamTable`` table provider gives partition pruning on
dimension predicates, projection pushdown, and exact per-partition
statistics for the optimizer. This module only routes the generic
[xarray_sql.register][] seam onto that existing machinery.
"""

from __future__ import annotations

from typing import Any, TypeGuard

import xarray as xr
from datafusion import SessionContext
from datafusion.catalog import Schema

from ..df import (
    Chunks,
    TableNames,
    group_vars_by_dims,
    resolve_table_names,
    shared_coord_arrays,
)
from ..reader import read_xarray_table
from ..sql import XarrayContext
from .base import register_adapter


@register_adapter
class DataFusionAdapter:
    """Registers Datasets on ``datafusion.SessionContext`` connections."""

    @staticmethod
    def matches(con: object) -> TypeGuard[SessionContext]:
        return isinstance(con, SessionContext)

    @staticmethod
    def register(
        con: SessionContext,
        name: str,
        ds: xr.Dataset,
        *,
        chunks: Chunks = None,
        table_names: TableNames = None,
        **kwargs: Any,
    ) -> SessionContext:
        # XarrayContext.from_dataset adds cftime UDF registration and
        # round-trip metadata tracking on top of the split below; use it
        # when available.
        if isinstance(con, XarrayContext):
            return con.from_dataset(
                name, ds, chunks=chunks, table_names=table_names, **kwargs
            )

        groups = group_vars_by_dims(ds)
        names = resolve_table_names(ds, table_names)
        if len(groups) <= 1:
            con.register_table(name, read_xarray_table(ds, chunks, **kwargs))
            return con

        # A mixed-dimension Dataset becomes one table per dimension group
        # in a schema named after the Dataset, exactly as XarrayContext
        # registers it — so ``name.group`` is the same SQL either way.
        coord_arrays = shared_coord_arrays(ds)
        schema = Schema.memory_schema(con)
        con.catalog().register_schema(name, schema)
        for dims, var_names in groups.items():
            schema.register_table(
                names[dims],
                read_xarray_table(
                    ds[var_names],
                    chunks,
                    coord_arrays=coord_arrays,
                    **kwargs,
                ),
            )
        return con
