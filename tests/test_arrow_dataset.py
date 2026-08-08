"""Contract tests for the engine-neutral pyarrow dataset view.

``xql.arrow_dataset`` returns a real ``pyarrow.dataset.Dataset`` for
consumers of ``schema``, ``scanner``, ``get_fragments`` and the scan
conveniences — pyarrow itself and Polars are exercised here; DuckDB has
its own suite in ``test_duckdb_backend.py``. DataFusion's native Rust
``TableProvider`` (``XarrayContext``) fills the same role through
DataFusion's own extension trait and sits outside this contract.

The contract, one section of this file per clause:

1. Exactness — the pushed filter is applied row-exactly after pruning;
   pruning never decides correctness.
2. Projection — exactly the requested columns come back; what gets read
   is the projected columns plus the filter's, nothing else.
3. Pruning/counting — provably impossible regions are never read,
   provable counts are pure arithmetic, and anything uncertain
   (boundary chunks, NaN/NaT coordinates, opaque expressions) is
   scanned conservatively.
4. Laziness — construction reads dimension coordinates only; scans
   repeat, survive mid-scan abandonment, and run concurrently.
5. Tuning (``batch_size``, ``prefetch``, ``prefetch_bytes``,
   ``coalesce_rows``) changes the shape of the work, never the result;
   non-positive ``batch_size`` is rejected at construction; fragments
   stay one per source chunk.
6. Schema stays on offset types (a view type disables DuckDB's
   pushdown); the prefetch pool is fully started at construction and
   shuts down when the dataset is collected.
"""

import threading

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import xarray as xr

import xarray_sql as xql
from xarray_sql.backends.pyarrow import XarrayPushdownDataset


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


class _ChunkCounter:
    def __init__(self):
        self.blocks = []
        self.column_sets = []

    def __call__(self, block, names):
        self.blocks.append(block)
        self.column_sets.append(tuple(names))


def _hourly_grid() -> xr.Dataset:
    """100 hourly steps x 4 latitudes with sequential values."""
    return xr.Dataset(
        {"t2m": (["time", "lat"], np.arange(100.0 * 4).reshape(100, 4))},
        coords={
            "time": pd.date_range("2020-01-01", periods=100, freq="h"),
            "lat": np.linspace(-30.0, 30.0, 4),
        },
    )


@pytest.fixture
def counted():
    """A pushdown dataset over hourly data with a chunk-read counter."""
    source = _hourly_grid()
    counter = _ChunkCounter()
    dataset = XarrayPushdownDataset(
        source, {"time": 10}, _iteration_callback=counter
    )
    return dataset, counter


# -- Dataset protocol surface ------------------------------------------------


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
    # the whole table; pin the schema to offset layouts so a pyarrow
    # upgrade cannot regress this silently.
    for field in xql.arrow_dataset(ds).schema:
        assert field.type not in (pa.string_view(), pa.binary_view())


# -- Consumer integrations ---------------------------------------------------


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


# -- Projection: what is returned vs what is read ----------------------------


def test_filter_only_columns_are_read_but_not_returned():
    src = xr.Dataset(
        {"a": (["i"], np.arange(100.0)), "b": (["i"], np.arange(100.0) * 2)},
        coords={"i": np.arange(100.0)},
    )
    counter = _ChunkCounter()
    dataset = XarrayPushdownDataset(src, {"i": 10}, _iteration_callback=counter)
    table = dataset.to_table(columns=["a"], filter=pc.field("b") >= 100.0)
    assert table.column_names == ["a"]
    assert table.num_rows == 50
    # The filter column is read alongside the projected one, nothing else.
    assert set(counter.column_sets) == {("a", "b")}


def test_empty_projection_is_a_real_projection(counted):
    dataset, counter = counted
    table = dataset.scanner(columns=[]).to_table()
    assert table.num_columns == 0
    assert table.num_rows == 100 * 4

    # With a filter, only the filter's column is read, still zero returned.
    counter.column_sets.clear()
    table = dataset.scanner(
        columns=[], filter=pc.field("t2m") < 40.0
    ).to_table()
    assert table.num_columns == 0
    assert table.num_rows == 40  # values 0..39: ten hours x four latitudes
    assert set(counter.column_sets) == {("t2m",)}


# -- Counting and pruning ----------------------------------------------------

_T0 = pd.Timestamp("2020-01-01 03:00")
_T1 = pd.Timestamp("2020-01-02 03:00")


def _ts(value):
    return pa.scalar(value, type=pa.timestamp("ns"))


@pytest.mark.parametrize(
    "predicate, rows, reads",
    [
        # No filter: pure arithmetic, no chunk is read.
        (None, 100 * 4, 0),
        # [03:00, 27:00): chunks 0 and 2 are boundary, chunk 1 is provably
        # inside the range and must be counted arithmetically.
        (
            (pc.field("time") >= _ts(_T0)) & (pc.field("time") < _ts(_T1)),
            24 * 4,
            2,
        ),
        # A data-variable filter carries no coordinate guarantee: every
        # chunk is a boundary chunk, and the count must still be row-exact.
        (pc.field("t2m") >= 200.0, 200, 10),
        # Unsatisfiable: everything pruned, nothing read.
        (pc.field("lat") > 100.0, 0, 0),
    ],
    ids=["unfiltered", "strict-chunks", "data-variable", "unsatisfiable"],
)
def test_count_rows_contract(counted, predicate, rows, reads):
    dataset, counter = counted
    assert dataset.count_rows(filter=predicate) == rows
    assert len(counter.blocks) == reads


def test_count_rows_broad_filter_stays_arithmetic():
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


def test_count_rows_cross_dimension_refinement():
    # Paired ranges across two chunked dims: per-dim pruning keeps the
    # union of each dim's survivors (so the cross combinations too);
    # the strictness pass must prune the crosses and count exactly.
    t = np.arange(200.0)
    lat = np.linspace(-45.0, 45.0, 20)
    reads: list = []
    dataset = XarrayPushdownDataset(
        xr.Dataset(
            {"v": (["t", "lat"], np.arange(200.0 * 20).reshape(200, 20))},
            coords={"t": t, "lat": lat},
        ),
        {"t": 10, "lat": 10},
        _iteration_callback=lambda b, n: reads.append(b),
    )
    predicate = ((pc.field("t") < 5.0) & (pc.field("lat") < -40.0)) | (
        (pc.field("t") >= 190.0) & (pc.field("lat") > 40.0)
    )
    n = dataset.count_rows(filter=predicate)
    expected = int(
        (
            ((t[:, None] < 5) & (lat[None, :] < -40))
            | ((t[:, None] >= 190) & (lat[None, :] > 40))
        ).sum()
    )
    assert n == expected
    # Per-dim pruning alone keeps 4 chunk combos (2 t-chunks x 2
    # lat-chunks); cross-dim refinement drops the 2 crosses.
    assert len(reads) <= 2


def test_poisoned_coordinates_prune_conservatively():
    # A NaN (or NaT) inside a coordinate chunk voids its range guarantee:
    # that chunk must be scanned, never pruned or counted arithmetically,
    # and the result must match the oracle (NaN compares False).
    x = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0])
    reads: list = []
    dataset = XarrayPushdownDataset(
        xr.Dataset({"v": (["x"], np.arange(6.0))}, coords={"x": x}),
        {"x": 2},
        _iteration_callback=lambda b, n: reads.append(b),
    )
    assert dataset.count_rows(filter=pc.field("x") > 0.5) == 4
    assert any(b["x"] == slice(2, 4) for b in reads)  # the NaN chunk

    t = pd.to_datetime(["2020-01-01", "2020-01-02", "NaT", "2020-01-04"])
    nat = XarrayPushdownDataset(
        xr.Dataset({"v": (["time"], np.arange(4.0))}, coords={"time": t}),
        {"time": 2},
    )
    lo = _ts(pd.Timestamp("2020-01-02"))
    assert nat.count_rows(filter=pc.field("time") >= lo) == 2


# -- Scan scheduling knobs: shape of the work, never the result ---------------


@pytest.mark.parametrize("coalesce_rows", [None, 10 * 4, 30 * 4, 10_000])
def test_coalesce_results_identical(coalesce_rows):
    source = _hourly_grid()
    dataset = XarrayPushdownDataset(
        source, {"time": 10}, coalesce_rows=coalesce_rows
    )
    predicate = (pc.field("time") >= _ts(_T0)) & (
        pc.field("time") < _ts(pd.Timestamp("2020-01-03 07:00"))
    )
    table = dataset.to_table(filter=predicate)
    assert table.num_rows == 52 * 4
    expected = source.t2m.isel(time=slice(3, 55)).values.ravel()
    np.testing.assert_array_equal(
        np.sort(table["t2m"].to_numpy()), np.sort(expected)
    )


def test_coalesce_merges_consecutive_chunk_runs():
    source = _hourly_grid()
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
    keep = (pc.field("time") < _ts(pd.Timestamp("2020-01-02 06:00"))) | (
        pc.field("time") >= _ts(pd.Timestamp("2020-01-03 22:00"))
    )
    table = dataset.to_table(filter=keep)
    assert table.num_rows == (30 + 30) * 4
    spans = sorted((b["time"].start, b["time"].stop) for b in reads)
    assert spans == [(0, 30), (70, 100)]


def test_coalesce_only_affects_scanner_not_fragments():
    dataset = XarrayPushdownDataset(
        _hourly_grid(), {"time": 10}, coalesce_rows=10_000
    )
    # Fragment consumers (DataFusion, dask) keep one fragment per source
    # chunk for their own parallelism.
    assert len(dataset.get_fragments()) == 10


def test_prefetch_bytes_scan_reads_every_block_once():
    source = xr.Dataset(
        {"v": (["step"], np.arange(10_000.0))},
        coords={"step": np.arange(10_000.0)},
    )
    reads: list = []
    # 100 chunks of 100 rows x 16 bytes/row = 1600 bytes per block; a
    # 4000-byte budget throttles admission well below the 8 threads
    # prefetch allows. The scan must still visit every block exactly
    # once and return the full table.
    dataset = XarrayPushdownDataset(
        source,
        {"step": 100},
        prefetch=8,
        prefetch_bytes=4_000,
        _iteration_callback=lambda b, n: reads.append(b),
    )
    table = dataset.to_table()
    assert table.num_rows == 10_000
    assert len(reads) == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefetch": 0},  # at most one worker: the pool-less path
        {"prefetch": -1},
        {"coalesce_rows": 0},
        {"prefetch_bytes": 0},
    ],
    ids=lambda kw: (
        next(iter(kw.items()))[0] + "=" + str(next(iter(kw.values())))
    ),
)
def test_degenerate_tuning_values_still_scan_exactly(kwargs):
    # Degenerate knob values may degrade the schedule (prefetch <= 1
    # takes the pool-less path) but never the answer, and never hang.
    src = xr.Dataset(
        {"v": (["i"], np.arange(100.0))}, coords={"i": np.arange(100.0)}
    )
    dataset = XarrayPushdownDataset(src, {"i": 10}, **kwargs)
    assert dataset.to_table().num_rows == 100
    assert dataset.count_rows(filter=pc.field("i") >= 50.0) == 50


@pytest.mark.parametrize("batch_size", [0, -5])
def test_non_positive_batch_size_fails_at_construction(batch_size):
    # batch_size cannot degrade gracefully: a zero size never advances
    # the zero-column scan's row loop, so it is rejected eagerly.
    src = xr.Dataset(
        {"v": (["i"], np.arange(100.0))}, coords={"i": np.arange(100.0)}
    )
    with pytest.raises(ValueError, match="batch_size"):
        XarrayPushdownDataset(src, {"i": 10}, batch_size=batch_size)


# -- Laziness, re-scannability, concurrency -----------------------------------


def test_abandoned_scanner_does_not_wedge_later_scans(counted):
    dataset, counter = counted
    batches = dataset.scanner().to_batches()
    next(batches)
    del batches  # LIMIT-style early stop: consumer walks away mid-scan
    counter.blocks.clear()
    assert dataset.count_rows() == 400
    table = dataset.to_table(columns=["t2m"])
    assert table.num_rows == 400


def test_concurrent_scans_are_isolated_and_exact():
    # Engines scan from their own worker threads; simultaneous filtered
    # scans over one dataset must not cross-talk.
    src = xr.Dataset(
        {"v": (["i"], np.arange(20_000.0))}, coords={"i": np.arange(20_000.0)}
    )
    dataset = XarrayPushdownDataset(src, {"i": 500}, prefetch=4)
    results: list = [None] * 8
    errors: list = []

    def worker(k):
        try:
            lo = k * 1000.0
            predicate = (pc.field("i") >= lo) & (pc.field("i") < lo + 3000.0)
            results[k] = dataset.to_table(filter=predicate).num_rows
        except Exception as exc:  # noqa: BLE001 — reported by the assert
            errors.append(f"{k}: {exc}")

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads), "a scan wedged"
    assert not errors
    assert results == [3000] * 8

    # Two batch iterators over the same dataset, consumed alternately,
    # stay exact and independent.
    a = dataset.scanner().to_batches()
    b = dataset.scanner().to_batches()
    rows_a = rows_b = 0
    exhausted_a = exhausted_b = False
    while not (exhausted_a and exhausted_b):
        batch = next(a, None)
        if batch is None:
            exhausted_a = True
        else:
            rows_a += batch.num_rows
        batch = next(b, None)
        if batch is None:
            exhausted_b = True
        else:
            rows_b += batch.num_rows
    assert rows_a == rows_b == 20_000


# -- Pool lifecycle ------------------------------------------------------------


def test_prefetch_pool_threads_all_started_at_construction(ds):
    dataset = XarrayPushdownDataset(ds, {"time": 5}, prefetch=6)
    # Every pool thread must exist before the first scan: a thread
    # spawned later, from inside an engine's scan callback, is exactly
    # the deadlock the pre-spawn exists to prevent. Thread accounting
    # is only visible on the executor's private state.
    assert len(dataset._pool._threads) == 6


def test_prefetch_pool_shut_down_when_dataset_dies(ds):
    import gc

    dataset = XarrayPushdownDataset(ds, {"time": 5}, prefetch=4)
    pool = dataset._pool
    del dataset
    gc.collect()
    # A shut-down executor refuses new work — the observable contract
    # that its threads have been told to exit.
    with pytest.raises(RuntimeError):
        pool.submit(lambda: None)
