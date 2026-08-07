"""Engine adapters — the *register* seam of xarray-sql.

xarray-sql translates data, not queries, across two seams — the two
boundaries between xarray and a query engine that neither side builds
for itself: *register* (a lazy ``xarray.Dataset`` becomes a table on
the engine's own connection; this package) and *round-trip* (an Arrow
result becomes a labeled Dataset again; [xarray_sql.to_dataset][]).
SQL dialects, geometry, H3, and optimizers belong to each engine and
its extension ecosystem.

Adapters register themselves on import via
[register_adapter][xarray_sql.backends.base.register_adapter];
[register][xarray_sql.backends.base.register] dispatches on the connection
type.
"""

from .base import EngineAdapter, get_adapter, register, register_adapter
from . import datafusion as _datafusion  # noqa: F401  (self-registers)
from . import duckdb as _duckdb  # noqa: F401  (self-registers)
from .pyarrow import (
    XarrayArrowStream,
    XarrayPushdownDataset,
    arrow_dataset,
)

__all__ = [
    "EngineAdapter",
    "XarrayArrowStream",
    "XarrayPushdownDataset",
    "arrow_dataset",
    "get_adapter",
    "register",
    "register_adapter",
]
