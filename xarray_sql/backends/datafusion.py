"""DataFusion engine adapter.

DataFusion is xarray-sql's default engine and the richest adapter: the
Rust ``LazyArrowStreamTable`` table provider gives partition pruning on
dimension predicates, projection pushdown, and exact per-partition
statistics for the optimizer. This module only routes the generic
:func:`xarray_sql.register` seam onto that existing machinery.
"""

from __future__ import annotations

from typing import Any

import xarray as xr
from datafusion import SessionContext

from ..df import Chunks
from ..reader import read_xarray_table
from ..sql import XarrayContext
from .base import register_adapter


@register_adapter
class DataFusionAdapter:
    """Registers Datasets on ``datafusion.SessionContext`` connections."""

    @staticmethod
    def matches(con: Any) -> bool:
        return isinstance(con, SessionContext)

    @staticmethod
    def register(
        con: SessionContext,
        name: str,
        ds: xr.Dataset,
        *,
        chunks: Chunks = None,
    ) -> SessionContext:
        # XarrayContext.from_dataset adds dim-group splitting, cftime UDF
        # registration, and round-trip metadata tracking on top of the
        # plain table registration; use it when available.
        if isinstance(con, XarrayContext):
            return con.from_dataset(name, ds, chunks=chunks)
        con.register_table(name, read_xarray_table(ds, chunks))
        return con
