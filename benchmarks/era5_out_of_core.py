"""Out-of-core benchmark + plan-shape assertions on public ARCO-ERA5.

Registers gs://gcp-public-data-arco-era5 (1,323,648 hourly time chunks;
nominally 1.37 trillion rows for one surface variable) without dask and
drives DuckDB, Polars, and the lazy round-trip against it, asserting
the *shape of the work* — exactly which source chunks each query reads
(via the scanner's iteration callback) and exact row counts — not just
the answers. A pruning or fast-path regression that silently falls back
to scanning everything fails these assertions long before it shows up
in wall-clock noise.

Needs network (anonymous GCS). Run: python benchmarks/era5_out_of_core.py
"""

import time

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import xarray as xr

import xarray_sql as xql
from xarray_sql.backends.pyarrow import XarrayPushdownDataset

URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

GRID = 721 * 1440

Q_DAY_BBOX = (
    'SELECT round(avg("2m_temperature") - 273.15, 2), count(*) FROM era5 '
    "WHERE time >= TIMESTAMP '2020-01-01' AND time < TIMESTAMP '2020-01-02' "
    "AND latitude BETWEEN 36 AND 44 AND longitude BETWEEN 350 AND 360"
)
Q_WEEK_GLOBE = (
    'SELECT round(avg("2m_temperature") - 273.15, 2), count(*) FROM era5 '
    "WHERE time >= TIMESTAMP '2020-01-03' AND time < TIMESTAMP '2020-01-10'"
)


def timed(label, fn, reads, expect_reads, expect_rows=None):
    reads.clear()
    t0 = time.time()
    out = fn()
    wall = time.time() - t0
    assert len(reads) == expect_reads, (
        f"{label}: read {len(reads)} blocks, expected {expect_reads} — "
        "pruning/coalescing regressed"
    )
    if expect_rows is not None:
        assert out[-1] == expect_rows, (label, out, expect_rows)
    print(f"{label:34s} {wall:6.2f}s  reads={len(reads):3d}  {out}")
    return out


def main() -> None:
    ds = xr.open_zarr(
        URL, chunks=None, storage_options={"token": "anon"}, consolidated=True
    )[["2m_temperature"]]

    reads: list = []
    dataset = XarrayPushdownDataset(
        ds,
        {"time": 1},
        prefetch=16,
        _iteration_callback=lambda b, n: reads.append(b),
    )
    con = duckdb.connect()
    con.register("era5", dataset)

    # 24 hourly chunks of 1,323,648; bbox trims rows exactly.
    timed(
        "duckdb day+bbox",
        lambda: con.execute(Q_DAY_BBOX).fetchone(),
        reads,
        expect_reads=24,
        expect_rows=24 * 33 * 40,
    )
    timed(
        "duckdb week globe",
        lambda: con.execute(Q_WEEK_GLOBE).fetchone(),
        reads,
        expect_reads=168,
        expect_rows=168 * GRID,
    )
    # count(*) over a chunk-aligned window: every surviving chunk is
    # provably inside the range, so the count is pure arithmetic.
    jan = (
        pc.field("time")
        >= pa.scalar(pd.Timestamp("2020-01-01"), type=pa.timestamp("ns"))
    ) & (
        pc.field("time")
        < pa.scalar(pd.Timestamp("2020-02-01"), type=pa.timestamp("ns"))
    )
    result = timed(
        "count_rows fast path (January)",
        lambda: (dataset.count_rows(filter=jan),),
        reads,
        expect_reads=0,
    )
    assert result[0] == 744 * GRID, result

    # Coalescing: same day+bbox in 4 merged reads instead of 24.
    coalesced = XarrayPushdownDataset(
        ds,
        {"time": 1},
        prefetch=16,
        coalesce_rows=8_000_000,
        _iteration_callback=lambda b, n: reads.append(b),
    )
    con2 = duckdb.connect()
    con2.register("era5", coalesced)
    timed(
        "duckdb day+bbox coalesced",
        lambda: con2.execute(Q_DAY_BBOX).fetchone(),
        reads,
        expect_reads=4,
        expect_rows=24 * 33 * 40,
    )

    # Lazy round-trip through Polars: construction reads nothing with
    # template coords; a 1-day window reads only its 4 coalesced blocks.
    lf = pl.scan_pyarrow_dataset(coalesced)
    reads.clear()
    lazy = xql.to_dataset(
        lf, template=ds, chunks={"time": 24}, coords="template"
    )
    assert reads == [], "lazy construction must not read the source"
    timed(
        "polars lazy 1-day window",
        lambda: (
            float(
                lazy["2m_temperature"]
                .sel(time=slice("2020-01-01", "2020-01-01 23:00"))
                .mean()
                .compute()
            ),
        ),
        reads,
        expect_reads=4,
    )
    print("all plan-shape assertions passed")


if __name__ == "__main__":
    main()
