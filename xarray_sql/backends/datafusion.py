"""DataFusion engine adapter.

DataFusion is xarray-sql's default engine and the richest adapter: the
Rust ``LazyArrowStreamTable`` table provider gives partition pruning on
dimension predicates, projection pushdown, and exact per-partition
statistics for the optimizer. This module only routes the generic
[xarray_sql.register][] seam onto that existing machinery.

Like the other adapters, it never imports ``datafusion`` until a
DataFusion connection is registered, so the rest of xarray-sql works
where DataFusion's Python bindings are unavailable (e.g. Pyodide).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

import xarray as xr

from ..df import (
    Chunks,
    TableNames,
    group_vars_by_dims,
    resolve_table_names,
    shared_coord_arrays,
)
from ..reader import read_xarray_table
from .base import register_adapter

if TYPE_CHECKING:
    from datafusion import SessionContext


@register_adapter
class DataFusionAdapter:
    """Registers Datasets on ``datafusion.SessionContext`` connections."""

    @staticmethod
    def matches(con: object) -> TypeGuard[SessionContext]:
        # Context and any other subclass carry this in their MRO.
        return any(
            cls.__module__ == "datafusion.context"
            and cls.__qualname__ == "SessionContext"
            for cls in type(con).__mro__
        )

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
        from datafusion.catalog import Schema

        from ..sql import Context

        # Context.from_dataset adds cftime UDF registration and
        # round-trip metadata tracking on top of the split below; use it
        # when available.
        if isinstance(con, Context):
            return con.from_dataset(
                name, ds, chunks=chunks, table_names=table_names, **kwargs
            )

        groups = group_vars_by_dims(ds)
        names = resolve_table_names(ds, table_names)
        if len(groups) <= 1:
            con.register_table(name, read_xarray_table(ds, chunks, **kwargs))
            return con

        # A mixed-dimension Dataset becomes one table per dimension group
        # in a schema named after the Dataset, exactly as Context
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
