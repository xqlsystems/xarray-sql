"""Tests for the engine-neutral pyarrow dataset view.

``xql.arrow_dataset`` returns a real ``pyarrow.dataset.Dataset``, so it
serves any consumer of that protocol — pyarrow itself and Polars are
exercised here; DuckDB has its own suite in ``test_duckdb_backend.py``.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import xarray as xr

import xarray_sql as xql


@pytest.fixture
def ds() -> xr.Dataset:
    np.random.seed(3)
    return xr.Dataset(
        {
            "temperature": (
                ["time", "lat"],
                20 + 5 * np.random.randn(20, 6),
            ),
            "humidity": (["time", "lat"], np.random.rand(20, 6)),
        },
        coords={
            "time": pd.date_range("2022-01-01", periods=20, freq="D"),
            "lat": np.linspace(-25.0, 25.0, 6),
        },
    ).chunk({"time": 5})


def test_to_table_projects_and_filters(ds):
    table = xql.arrow_dataset(ds).to_table(
        columns=["time", "temperature"],
        filter=pc.field("lat") > 0,
    )
    assert table.column_names == ["time", "temperature"]
    assert table.num_rows == 20 * 3  # lat > 0 keeps 3 of 6 latitudes


def test_count_rows_and_head(ds):
    dataset = xql.arrow_dataset(ds)
    assert dataset.count_rows() == 20 * 6
    assert dataset.head(7).num_rows == 7


def test_get_fragments_prunes_and_scans(ds):
    dataset = xql.arrow_dataset(ds)
    assert len(dataset.get_fragments()) == 4  # time chunked by 5

    # A time predicate covering the first chunk keeps one fragment.
    early = pc.field("time") < pa.scalar(
        pd.Timestamp("2022-01-06"), type=pa.timestamp("ns")
    )
    kept = dataset.get_fragments(filter=early)
    assert len(kept) == 1
    assert kept[0].to_table().num_rows == 5 * 6

    # An unsatisfiable predicate prunes everything.
    assert dataset.get_fragments(filter=pc.field("lat") > 100) == []


def test_datafusion_register_dataset_round_trips(ds):
    from datafusion import SessionContext

    ctx = SessionContext()
    ctx.register_dataset("t", xql.arrow_dataset(ds))
    out = ctx.sql(
        "SELECT time, AVG(temperature) AS temperature FROM t "
        "WHERE lat > 0 GROUP BY time ORDER BY time"
    ).to_pandas()
    expected = ds.temperature.sel(lat=ds.lat[ds.lat > 0]).mean("lat").compute()
    np.testing.assert_allclose(out["temperature"].values, expected.values)


def test_dask_from_map_over_fragments(ds):
    dd = pytest.importorskip("dask.dataframe")

    frags = xql.arrow_dataset(ds).get_fragments()
    ddf = dd.from_map(lambda f: f.to_table().to_pandas(), frags)
    assert len(ddf.compute()) == 20 * 6


def test_polars_scan_pushdown_round_trip(ds):
    pl = pytest.importorskip("polars")

    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(ds))
    out = (
        lf.filter(pl.col("lat") > 0)
        .group_by("time")
        .agg(pl.col("temperature").mean())
        .sort("time")
        .collect()
    )
    expected = (
        ds.temperature.sel(lat=ds.lat[ds.lat > 0]).mean("lat").compute().values
    )
    np.testing.assert_allclose(out["temperature"].to_numpy(), expected)


def test_polars_result_round_trips_to_xarray(ds):
    pl = pytest.importorskip("polars")

    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(ds))
    frame = (
        lf.group_by("time")
        .agg(pl.col("temperature").mean().alias("temperature"))
        .sort("time")
        .collect()
    )
    # Polars DataFrames export Arrow via the PyCapsule protocol, so the
    # engine-agnostic round-trip works unchanged.
    out = xql.to_dataset(frame, template=ds)
    assert list(out.dims) == ["time"]
    assert out.sizes["time"] == 20


def test_scanner_honors_batch_size(ds):
    dataset = xql.arrow_dataset(ds)
    batches = list(dataset.scanner(batch_size=7).to_batches())
    assert sum(b.num_rows for b in batches) == 20 * 6
    assert max(b.num_rows for b in batches) <= 7

    # The kwarg travels through the inherited to_batches path, which is
    # how Polars sizes its morsels.
    sizes = [b.num_rows for b in dataset.to_batches(batch_size=11)]
    assert sum(sizes) == 20 * 6
    assert max(sizes) <= 11


def test_schema_never_uses_view_types(ds):
    # A single view-typed column disables DuckDB's filter pushdown for
    # the whole table (duckdb-python#227); pin the schema to offset
    # layouts so a pyarrow upgrade cannot regress this silently.
    for field in xql.arrow_dataset(ds).schema:
        assert field.type not in (pa.string_view(), pa.binary_view())


class _ChunkCounter:
    def __init__(self):
        self.blocks = []

    def __call__(self, block, names):
        self.blocks.append(block)


@pytest.fixture
def counted():
    """A pushdown dataset over hourly data with a chunk-read counter."""
    from xarray_sql.backends.pyarrow import XarrayPushdownDataset

    source = xr.Dataset(
        {
            "t2m": (
                ["time", "lat"],
                np.arange(100.0 * 4).reshape(100, 4),
            )
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=100, freq="h"),
            "lat": np.linspace(-30.0, 30.0, 4),
        },
    )
    counter = _ChunkCounter()
    dataset = XarrayPushdownDataset(
        source, {"time": 10}, _iteration_callback=counter
    )
    return dataset, counter


def test_count_rows_unfiltered_is_pure_arithmetic(counted):
    dataset, counter = counted
    assert dataset.count_rows() == 100 * 4
    assert counter.blocks == []  # no chunk was read


def test_count_rows_strict_chunks_counted_without_reading(counted):
    dataset, counter = counted
    # [03:00, 27:00): chunks 0 and 2 are boundary, chunk 1 is provably
    # inside the range and must be counted arithmetically.
    lo = pa.scalar(pd.Timestamp("2020-01-01 03:00"), type=pa.timestamp("ns"))
    hi = pa.scalar(pd.Timestamp("2020-01-02 03:00"), type=pa.timestamp("ns"))
    predicate = (pc.field("time") >= lo) & (pc.field("time") < hi)
    assert dataset.count_rows(filter=predicate) == 24 * 4
    assert len(counter.blocks) == 2  # only the two boundary chunks


def test_count_rows_data_variable_filter_is_exact(counted):
    dataset, counter = counted
    # A filter on a data variable carries no coordinate guarantee: every
    # chunk is a boundary chunk, and the count must still be row-exact.
    assert dataset.count_rows(filter=pc.field("t2m") >= 200.0) == 200
    assert len(counter.blocks) == 10


def test_count_rows_unsatisfiable_filter(counted):
    dataset, counter = counted
    assert dataset.count_rows(filter=pc.field("lat") > 100.0) == 0
    assert counter.blocks == []


def test_empty_projection_is_a_real_projection(counted):
    dataset, _ = counted
    table = dataset.scanner(columns=[]).to_table()
    assert table.num_columns == 0
    assert table.num_rows == 100 * 4


def test_abandoned_scanner_does_not_wedge_later_scans(counted):
    dataset, counter = counted
    batches = dataset.scanner().to_batches()
    next(batches)
    del batches  # LIMIT-style early stop: consumer walks away mid-scan
    counter.blocks.clear()
    assert dataset.count_rows() == 400
    table = dataset.to_table(columns=["t2m"])
    assert table.num_rows == 400


@pytest.mark.parametrize("coalesce_rows", [None, 10 * 4, 30 * 4, 10_000])
def test_coalesce_results_identical(coalesce_rows):
    from xarray_sql.backends.pyarrow import XarrayPushdownDataset

    source = xr.Dataset(
        {"t2m": (["time", "lat"], np.arange(100.0 * 4).reshape(100, 4))},
        coords={
            "time": pd.date_range("2020-01-01", periods=100, freq="h"),
            "lat": np.linspace(-30.0, 30.0, 4),
        },
    )
    dataset = XarrayPushdownDataset(
        source, {"time": 10}, coalesce_rows=coalesce_rows
    )
    lo = pa.scalar(pd.Timestamp("2020-01-01 03:00"), type=pa.timestamp("ns"))
    hi = pa.scalar(pd.Timestamp("2020-01-03 07:00"), type=pa.timestamp("ns"))
    predicate = (pc.field("time") >= lo) & (pc.field("time") < hi)
    table = dataset.to_table(filter=predicate)
    assert table.num_rows == 52 * 4
    expected = source.t2m.isel(time=slice(3, 55)).values.ravel()
    np.testing.assert_array_equal(
        np.sort(table["t2m"].to_numpy()), np.sort(expected)
    )


def test_coalesce_merges_consecutive_chunk_runs():
    from xarray_sql.backends.pyarrow import XarrayPushdownDataset

    source = xr.Dataset(
        {"t2m": (["time", "lat"], np.arange(100.0 * 4).reshape(100, 4))},
        coords={
            "time": pd.date_range("2020-01-01", periods=100, freq="h"),
            "lat": np.linspace(-30.0, 30.0, 4),
        },
    )
    reads: list[dict] = []
    dataset = XarrayPushdownDataset(
        source,
        {"time": 10},
        coalesce_rows=30 * 4,  # up to 3 source chunks per read
        _iteration_callback=lambda b, n: reads.append(b),
    )
    # An unfiltered scan of 10 chunks arrives as ceil(10/3) = 4 reads.
    assert dataset.to_table().num_rows == 400
    assert len(reads) == 4
    spans = sorted((b["time"].start, b["time"].stop) for b in reads)
    assert spans == [(0, 30), (30, 60), (60, 90), (90, 100)]

    # Pruning still applies before merging: a filter keeping chunks
    # 0-2 and 7-9 yields one merged read per consecutive run.
    reads.clear()
    keep = (
        (pc.field("time") < pa.scalar(pd.Timestamp("2020-01-02 06:00"), type=pa.timestamp("ns")))
        | (pc.field("time") >= pa.scalar(pd.Timestamp("2020-01-03 22:00"), type=pa.timestamp("ns")))
    )
    table = dataset.to_table(filter=keep)
    assert table.num_rows == (30 + 30) * 4
    spans = sorted((b["time"].start, b["time"].stop) for b in reads)
    assert spans == [(0, 30), (70, 100)]


def test_coalesce_only_affects_scanner_not_fragments():
    from xarray_sql.backends.pyarrow import XarrayPushdownDataset

    source = xr.Dataset(
        {"t2m": (["time", "lat"], np.arange(100.0 * 4).reshape(100, 4))},
        coords={
            "time": pd.date_range("2020-01-01", periods=100, freq="h"),
            "lat": np.linspace(-30.0, 30.0, 4),
        },
    )
    dataset = XarrayPushdownDataset(
        source, {"time": 10}, coalesce_rows=10_000
    )
    # Fragment consumers (DataFusion, dask) keep one fragment per source
    # chunk for their own parallelism.
    assert len(dataset.get_fragments()) == 10


def test_count_rows_broad_filter_stays_arithmetic():
    from xarray_sql.backends.pyarrow import XarrayPushdownDataset

    # 100k single-element chunks with a filter keeping nearly all of
    # them: the hierarchical strictness analysis must prove whole
    # buckets at once instead of scanning every survivor.
    reads: list = []
    dataset = XarrayPushdownDataset(
        xr.Dataset(
            {"v": (["step"], np.arange(100_000.0))},
            coords={"step": np.arange(100_000.0)},
        ),
        {"step": 1},
        _iteration_callback=lambda b, n: reads.append(b),
    )
    assert dataset.count_rows(filter=pc.field("step") >= 100.0) == 99_900
    assert len(reads) <= 2  # at most the bucket-edge chunk


def test_count_rows_cross_dimension_refinement(counted):
    dataset, counter = counted
    # Paired ranges across dims: per-dim pruning keeps the union (the
    # cross combos too); the strictness pass must prune the crosses and
    # count exactly.
    lo = pa.scalar(pd.Timestamp("2020-01-01 00:00"), type=pa.timestamp("ns"))
    a = (pc.field("time") < pa.scalar(pd.Timestamp("2020-01-01 05:00"), type=pa.timestamp("ns"))) & (
        pc.field("lat") < -25.0
    )
    b = (pc.field("time") >= pa.scalar(pd.Timestamp("2020-01-04 22:00"), type=pa.timestamp("ns"))) & (
        pc.field("lat") > 25.0
    )
    n = dataset.count_rows(filter=a | b)
    assert n == 5 * 1 + 2 * 1  # 5 early hours x 1 lat + 2 late hours x 1 lat
    # only the genuinely mixed chunks were touched, not the crosses
    assert len(counter.blocks) <= 2
