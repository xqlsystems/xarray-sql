"""Engine adapters — the *register* seam of xarray-sql.

xarray-sql translates data, not queries: it registers lazy
``xarray.Dataset`` objects as tables on a query engine's own connection
(seam 1, this package) and turns Arrow results back into labeled
Datasets (seam 2, :func:`xarray_sql.to_dataset`). SQL dialects,
geometry, H3, and optimizers belong to each engine and its extension
ecosystem.

Adapters register themselves on import via
:func:`~xarray_sql.backends.base.register_adapter`;
:func:`~xarray_sql.backends.base.register` dispatches on the connection
type.
"""

from .base import EngineAdapter, get_adapter, register, register_adapter
from . import datafusion as _datafusion  # noqa: F401  (self-registers)
from . import duckdb as _duckdb  # noqa: F401  (self-registers)
from .duckdb import XarrayArrowStream, XarrayPushdownDataset

__all__ = [
    "EngineAdapter",
    "XarrayArrowStream",
    "XarrayPushdownDataset",
    "get_adapter",
    "register",
    "register_adapter",
]
