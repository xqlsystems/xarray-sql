"""Plan-shape assertions on public ARCO-ERA5 (integration, needs network).

Registers gs://gcp-public-data-arco-era5 (1,323,648 hourly time chunks;
nominally 1.37 trillion rows for one surface variable) without dask and
drives DuckDB, Polars, and the lazy round-trip against it, asserting
the *shape of the work* — exactly which source chunks each query reads
(via the scanner's iteration callback) and exact row counts — not just
the answers. A pruning or fast-path regression that silently falls back
to scanning everything fails these assertions long before it shows up
in wall-clock noise.

Reads anonymously from GCS. Excluded from the CI unit run
(``pytest -m "not integration"``); run deliberately with
``pytest -m integration tests/test_era5_integration.py``.
"""

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import xarray as xr

import xarray_sql as xql
from xarray_sql.backends.pyarrow import XarrayPushdownDataset

pytestmark = pytest.mark.integration

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


@pytest.fixture(scope="module")
def era5() -> xr.Dataset:
    return xr.open_zarr(
        URL, chunks=None, storage_options={"token": "anon"}, consolidated=True
    )[["2m_temperature"]]


def _tracked_dataset(era5, **kwargs):
    """An XarrayPushdownDataset that records every block it reads."""
    reads: list = []
    dataset = XarrayPushdownDataset(
        era5,
        {"time": 1},
        prefetch=16,
        _iteration_callback=lambda b, n: reads.append(b),
        **kwargs,
    )
    return dataset, reads


def test_duckdb_day_bbox_prunes_to_24_chunks(era5):
    # 24 hourly chunks of 1,323,648 rows; the bbox trims rows exactly.
    dataset, reads = _tracked_dataset(era5)
    con = duckdb.connect()
    con.register("era5", dataset)
    out = con.execute(Q_DAY_BBOX).fetchone()
    assert len(reads) == 24, f"read {len(reads)} blocks — pruning regressed"
    assert out[-1] == 24 * 33 * 40


def test_duckdb_week_globe_prunes_to_168_chunks(era5):
    dataset, reads = _tracked_dataset(era5)
    con = duckdb.connect()
    con.register("era5", dataset)
    out = con.execute(Q_WEEK_GLOBE).fetchone()
    assert len(reads) == 168, f"read {len(reads)} blocks — pruning regressed"
    assert out[-1] == 168 * GRID


def test_dataset_is_rescannable_across_queries(era5):
    # One wrapper, two queries: the second scan must see fresh state,
    # not a consumed stream or stale pruning from the first.
    dataset, reads = _tracked_dataset(era5)
    con = duckdb.connect()
    con.register("era5", dataset)
    day = con.execute(Q_DAY_BBOX).fetchone()
    assert len(reads) == 24 and day[-1] == 24 * 33 * 40
    reads.clear()
    week = con.execute(Q_WEEK_GLOBE).fetchone()
    assert len(reads) == 168 and week[-1] == 168 * GRID


def test_count_rows_fast_path_reads_nothing(era5):
    # count(*) over a chunk-aligned window: every surviving chunk is
    # provably inside the range, so the count is pure arithmetic.
    dataset, reads = _tracked_dataset(era5)
    jan = (
        pc.field("time")
        >= pa.scalar(pd.Timestamp("2020-01-01"), type=pa.timestamp("ns"))
    ) & (
        pc.field("time")
        < pa.scalar(pd.Timestamp("2020-02-01"), type=pa.timestamp("ns"))
    )
    count = dataset.count_rows(filter=jan)
    assert reads == [], "count_rows fast path must not read data"
    assert count == 744 * GRID


def test_coalescing_merges_day_bbox_into_4_reads(era5):
    # Coalescing: the same day+bbox in 4 merged reads instead of 24.
    dataset, reads = _tracked_dataset(era5, coalesce_rows=8_000_000)
    con = duckdb.connect()
    con.register("era5", dataset)
    out = con.execute(Q_DAY_BBOX).fetchone()
    assert len(reads) == 4, f"read {len(reads)} blocks — coalescing regressed"
    assert out[-1] == 24 * 33 * 40


def test_polars_lazy_roundtrip_window_reads_4_blocks(era5):
    # Lazy round-trip through Polars: construction reads nothing with
    # template coords; a 1-day window reads only its 4 coalesced blocks.
    dataset, reads = _tracked_dataset(era5, coalesce_rows=8_000_000)
    lf = pl.scan_pyarrow_dataset(dataset)
    lazy = xql.to_dataset(
        lf, template=era5, chunks={"time": 24}, coords="template"
    )
    assert reads == [], "lazy construction must not read the source"
    value = float(
        lazy["2m_temperature"]
        .sel(time=slice("2020-01-01", "2020-01-01 23:00"))
        .mean()
        .compute()
    )
    assert len(reads) == 4, f"read {len(reads)} blocks — window regressed"
    assert value == pytest.approx(277.7, abs=5.0)  # plausible global-ish K
