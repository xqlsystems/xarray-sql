"""Exact table statistics reach the optimizer through the FFI boundary.

DataFusion 54 forwards ``Statistics`` across the ``datafusion-ffi`` boundary,
so the exact statistics ``XarrayScanExec`` reports are visible to the query
optimizer: num_rows (product of a chunk's dimension sizes), total byte size,
and per dimension-column min/max bounds. These tests pin that behaviour.
"""

import numpy as np
import xarray as xr

from xarray_sql import XarrayContext


def _explain(ctx: XarrayContext, query: str) -> str:
    ctx.sql("SET datafusion.explain.show_statistics = true").collect()
    rows = ctx.sql(f"EXPLAIN {query}").to_pandas()
    return "\n".join(rows["plan"].tolist())


def test_exact_rows_in_scan_statistics():
    """The scan reports exact row counts (forwarded across FFI)."""
    ds = xr.Dataset(
        {"air": (("time", "lat", "lon"), np.random.rand(100, 4, 5))},
        coords={
            "time": np.arange(100),
            "lat": np.arange(4),
            "lon": np.arange(5),
        },
    )
    ctx = XarrayContext()
    ctx.from_dataset("air", ds, chunks={"time": 50})
    plan = _explain(ctx, "SELECT lat, lon, air FROM air")
    total = 100 * 4 * 5
    assert f"Rows=Exact({total})" in plan


def test_exact_byte_size_in_scan_statistics():
    """The scan reports exact byte size (num_rows x fixed row width)."""
    ds = xr.Dataset(
        {"air": (("time", "lat", "lon"), np.random.rand(100, 4, 5))},
        coords={
            "time": np.arange(100),
            "lat": np.arange(4),
            "lon": np.arange(5),
        },
    )
    ctx = XarrayContext()
    ctx.from_dataset("air", ds, chunks={"time": 50})
    plan = _explain(ctx, "SELECT lat, lon, air FROM air")
    # 2000 rows x (lat int64 + lon int64 + air float64) = 2000 x 24 bytes, sized
    # by value type. lat/lon are dictionary-encoded (int32 keys), so the value
    # sizing is a safe upper bound and the total is reported Inexact.
    assert f"Bytes=Inexact({100 * 4 * 5 * 24})" in plan


def test_dimension_column_min_max_in_scan_statistics():
    """Dimension columns carry exact min/max and a zero null count.

    These are the join/filter key columns; the bounds come from the same
    coordinate metadata used for partition pruning (no data scan), and grid
    axes are always fully populated so the null count is exactly zero.
    """
    ds = xr.Dataset(
        {"air": (("time", "lat", "lon"), np.random.rand(100, 4, 5))},
        coords={
            "time": np.arange(100),
            "lat": np.arange(4),
            "lon": np.arange(5),
        },
    )
    ctx = XarrayContext()
    ctx.from_dataset("air", ds, chunks={"time": 50})
    plan = _explain(ctx, "SELECT lat, lon, air FROM air")
    # lat spans 0..3, lon spans 0..4, both never null.
    assert "Min=Exact(Int64(0)) Max=Exact(Int64(3)) Null=Exact(0)" in plan
    assert "Min=Exact(Int64(0)) Max=Exact(Int64(4)) Null=Exact(0)" in plan


def test_count_star_answered_from_statistics():
    """COUNT(*) returns the exact count from statistics (metadata only)."""
    ds = xr.Dataset(
        {"air": (("time", "lat", "lon"), np.random.rand(100, 4, 5))},
        coords={
            "time": np.arange(100),
            "lat": np.arange(4),
            "lon": np.arange(5),
        },
    )
    ctx = XarrayContext()
    ctx.from_dataset("air", ds, chunks={"time": 50})
    n = ctx.sql("SELECT COUNT(*) AS n FROM air").to_pandas()["n"][0]
    assert int(n) == 100 * 4 * 5


def test_join_picks_small_build_side():
    """With exact stats the optimizer broadcasts the smaller table (CollectLeft).

    Without statistics (the pre-54 FFI path) the optimizer could not know which
    side was smaller and fell back to a Partitioned hash join.
    """
    rng = np.random.default_rng(0)
    big = xr.Dataset(
        {"t": (("time", "lat", "lon"), rng.standard_normal((200, 8, 8)))},
        coords={
            "time": np.arange(200),
            "lat": np.arange(8),
            "lon": np.arange(8),
        },
    )
    small = xr.Dataset(
        {"w": (("lat", "lon"), rng.standard_normal((8, 8)))},
        coords={"lat": np.arange(8), "lon": np.arange(8)},
    )
    ctx = XarrayContext()
    ctx.from_dataset("big", big, chunks={"time": 50})
    ctx.from_dataset("small", small, chunks={"lat": 8})

    plan = _explain(
        ctx,
        "SELECT b.time, SUM(b.t * s.w) AS x FROM big b "
        "JOIN small s ON b.lat=s.lat AND b.lon=s.lon GROUP BY b.time",
    )
    assert "HashJoinExec: mode=CollectLeft" in plan
    # The small (build) side's exact cardinality crossed the FFI boundary.
    assert "Rows=Exact(64)" in plan
