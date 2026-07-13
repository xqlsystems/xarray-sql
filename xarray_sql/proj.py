"""PROJ-backed CRS transforms for SQL — an optional pyproj extension.

Geospatial SQL dialects expose coordinate reference system (CRS)
transforms as a scalar function — PostGIS and DuckDB-spatial both call it
``ST_Transform`` — because a CRS transform is row-independent: each
point's new coordinate depends only on its own old coordinate. This
module brings the same capability to xarray-sql as a vectorized scalar
UDF over Arrow arrays::

    SELECT x, y,
           reproject(x, y, 'EPSG:32610', 'EPSG:4326')['x'] AS lon,
           reproject(x, y, 'EPSG:32610', 'EPSG:4326')['y'] AS lat
    FROM grid

The CRS pair is part of the *query*, not baked in at registration time,
so one registered UDF serves any transform — and, because the arguments
are ordinary SQL expressions, the CRS may even vary per row (e.g. a
``CASE`` expression selecting the UTM zone from the longitude).

Design notes:

* **Both output coordinates come from one call**, returned as an Arrow
  struct ``{x, y}`` (in ``always_xy`` order: easting/longitude first).
  Splitting the transform into two scalar UDFs would run PROJ twice per
  row and, worse, evaluate the two projections concurrently on separate
  expression trees.
* **All pyproj work runs on a dedicated pool of Python threads.**
  Constructing a transformer on a DataFusion runtime thread segfaults
  inside PROJ, while the identical work on Python-owned threads is
  stable — so the UDF hands each batch to the pool rather than calling
  pyproj in place. Each pool thread caches one transformer per CRS pair
  (transformers must not be shared across threads), which also
  amortizes the expensive PROJ database lookups of construction across
  record batches. Concurrent partitions still transform in parallel
  across the pool.
* Any CRS spelling ``pyproj.CRS`` accepts works: authority codes
  (``EPSG:4326``), WKT, PROJ strings (``+proj=utm +zone=10``), etc.
  An unknown CRS raises ``pyproj.exceptions.CRSError`` and fails the
  query loudly rather than returning wrong coordinates.
* Non-finite or NULL input coordinates yield NaN output (PROJ itself
  would return ``inf``); NULL CRS arguments yield NaN as well.

Requires ``pyproj`` (``pip install xarray-sql[geo]``). When pyproj is
installed, :class:`xarray_sql.XarrayContext` registers ``reproject()``
automatically; :func:`register` is the explicit hook for plain
DataFusion ``SessionContext`` objects or custom UDF names.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyproj
from datafusion import udf

__all__ = ["register"]

#: Arrow type returned by ``reproject()``: destination coordinates in
#: ``always_xy`` order — ``x`` is easting/longitude, ``y`` is
#: northing/latitude.
RETURN_TYPE = pa.struct([("x", pa.float64()), ("y", pa.float64())])


# ---------------------------------------------------------------------------
# The PROJ worker pool
# ---------------------------------------------------------------------------
#
# DataFusion evaluates UDFs on its runtime's worker threads. Plain Python
# threads run pyproj construction and transforms concurrently without
# incident (pyproj keeps its PROJ contexts thread-local), but constructing
# a ``Transformer`` *on a DataFusion runtime thread* segfaults inside
# PROJ's CRS machinery — Rust runtime threads are provisioned differently
# from Python threads (notably a much smaller stack). So the UDF never
# calls pyproj in place: every batch is handed to a small pool of
# Python-owned worker threads. pyproj releases the GIL during the
# transform loop, so concurrent partitions still run in parallel across
# the pool.

_local = threading.local()
_pool_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None


def _proj_pool() -> ThreadPoolExecutor:
    """Return the process-wide pool that runs all pyproj work."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(
                    max_workers=os.cpu_count() or 4,
                    thread_name_prefix="xarray-sql-proj",
                )
    return _pool


def _transformer(src_crs: str, dst_crs: str) -> pyproj.Transformer:
    """Return a cached ``Transformer`` owned by the calling pool thread.

    PROJ transformers are not safe to share across threads, so each pool
    thread keeps its own transformer per ``(src, dst)`` pair; the cache
    also amortizes construction (expensive PROJ database lookups) across
    record batches. ``always_xy=True`` fixes the argument order to
    (easting/longitude, northing/latitude) regardless of the CRS's
    declared axis order.
    """
    cache = getattr(_local, "transformers", None)
    if cache is None:
        cache = _local.transformers = {}
    key = (src_crs, dst_crs)
    transformer = cache.get(key)
    if transformer is None:
        transformer = cache[key] = pyproj.Transformer.from_crs(
            src_crs, dst_crs, always_xy=True
        )
    return transformer


def _transform_chunk(
    src_crs: str, dst_crs: str, xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform one coordinate chunk; runs on a PROJ pool thread."""
    return _transformer(src_crs, dst_crs).transform(xs, ys)


# ---------------------------------------------------------------------------
# The UDF
# ---------------------------------------------------------------------------


def _reproject(
    x: pa.Array, y: pa.Array, src_crs: pa.Array, dst_crs: pa.Array
) -> pa.Array:
    """Vectorized ``reproject`` kernel over one Arrow record batch.

    DataFusion broadcasts scalar arguments (the usual literal CRS
    strings) to full-length arrays before calling in, so all four
    arguments arrive with one value per row. The common case — one CRS
    pair for the whole batch — never touches the strings row by row:
    uniqueness is established with a vectorized Arrow kernel and the
    batch becomes a single PROJ call. (Materializing the CRS columns
    as Python strings costs two object allocations per row, which at
    billions of rows dwarfs the transform itself.) Only when the CRS
    genuinely varies within the batch are rows grouped by pair and
    transformed per group.
    """
    xs = np.asarray(x.to_numpy(zero_copy_only=False), dtype="float64")
    ys = np.asarray(y.to_numpy(zero_copy_only=False), dtype="float64")

    out_x = np.full(xs.shape, np.nan)
    out_y = np.full(ys.shape, np.nan)
    valid = np.isfinite(xs) & np.isfinite(ys)
    src_unique = pc.unique(src_crs)
    dst_unique = pc.unique(dst_crs)

    if len(src_unique) == 1 and len(dst_unique) == 1:
        src, dst = src_unique[0].as_py(), dst_unique[0].as_py()
        if src is not None and dst is not None and valid.any():
            tx, ty = (
                _proj_pool()
                .submit(_transform_chunk, src, dst, xs[valid], ys[valid])
                .result()
            )
            out_x[valid] = tx
            out_y[valid] = ty
    else:
        pairs = list(zip(src_crs.to_pylist(), dst_crs.to_pylist()))
        for src, dst in set(pairs):
            if src is None or dst is None:
                continue
            mask = valid & np.fromiter(
                (p == (src, dst) for p in pairs), dtype=bool, count=len(pairs)
            )
            if not mask.any():
                continue
            tx, ty = (
                _proj_pool()
                .submit(_transform_chunk, src, dst, xs[mask], ys[mask])
                .result()
            )
            out_x[mask] = tx
            out_y[mask] = ty

    # PROJ signals out-of-domain points with inf; normalize to NaN so
    # the result round-trips to xarray like any other missing value.
    invalid = ~(np.isfinite(out_x) & np.isfinite(out_y))
    out_x[invalid] = np.nan
    out_y[invalid] = np.nan

    return pa.StructArray.from_arrays(
        [pa.array(out_x), pa.array(out_y)], names=["x", "y"]
    )


def register(ctx, name: str = "reproject") -> None:
    """Register the ``reproject(x, y, src_crs, dst_crs)`` scalar UDF.

    Works on any DataFusion ``SessionContext`` (``XarrayContext``
    registers it automatically when pyproj is installed). The UDF
    returns a ``{x, y}`` struct of destination coordinates, so a query
    selects components with subscripts::

        SELECT reproject(x, y, 'EPSG:32610', 'EPSG:4326')['x'] AS lon
        FROM grid

    Args:
        ctx: The DataFusion session context to register the UDF on.
        name: SQL name for the function (default ``"reproject"``).
    """
    ctx.register_udf(
        udf(
            _reproject,
            [pa.float64(), pa.float64(), pa.utf8(), pa.utf8()],
            RETURN_TYPE,
            "immutable",
            name,
        )
    )
