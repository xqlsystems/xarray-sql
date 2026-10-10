from . import cftime
from .backends import (
    FlightSQLServer,
    arrow_dataset,
    arrow_datasets,
    register,
    serve,
)
from .geometry import bbox_conjuncts
from .df import from_map
from .reader import read_xarray, read_xarray_table
from .roundtrip import to_dataset

__all__ = [
    "cftime",
    "Context",
    "XarrayContext",
    "FlightSQLServer",
    "read_xarray_table",
    "read_xarray",
    "arrow_dataset",
    "arrow_datasets",
    "bbox_conjuncts",
    "register",
    "serve",
    "to_dataset",
    "from_map",  # deprecated
]

try:
    from .sql import Context, XarrayContext
except ModuleNotFoundError as e:
    # DataFusion's Python bindings have no wheels for some platforms
    # (e.g. Pyodide); every other engine adapter still works there.
    if e.name != "datafusion":
        raise
    __all__.remove("Context")
    __all__.remove("XarrayContext")

    def __getattr__(name: str):
        if name in ("Context", "XarrayContext"):
            raise AttributeError(
                f"{name} requires the datafusion package, which is "
                "not installed."
            )
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
