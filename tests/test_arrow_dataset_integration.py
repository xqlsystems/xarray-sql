"""Integration tests: the arrow-dataset contract against real cloud stores.

The same contract ``test_arrow_dataset.py`` pins on synthetic data,
exercised at scale: real Zarr stores read over the network, consumed
through the real engines (DuckDB, Polars, DataFusion). Assertions are
plan-shape — exactly which source chunks each query reads, exact row
counts, values matched against a direct xarray read of the same window —
so a pruning or fast-path regression fails long before it shows up in
wall-clock noise.

Two axes, both extensible:

* ``CASES`` — one :class:`StoreCase` per dataset. Expectations are
  computed from the case's declared cadence and windows plus the store's
  own coordinates. A new dataset is a config entry, provided its
  temporal dimension is named ``time``, its cadence is regular, and its
  grid is dense; anything else needs test changes, not just a case.
* ``ENGINES`` — engine name to runner function; every runner executes
  the same windowed aggregation through its engine's idiomatic path
  (DuckDB SQL, Polars lazy expressions, DataFusion SQL).

Reads anonymously from public buckets. Excluded from the CI unit run
(``pytest -m "not integration"``); run deliberately with
``pytest -m integration tests/test_arrow_dataset_integration.py``.
"""

import threading
from dataclasses import dataclass, field

import pandas as pd
import pytest
import xarray as xr

import xarray_sql as xql
from xarray_sql.backends.pyarrow import XarrayPushdownDataset

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class StoreCase:
    id: str
    url: str
    variable: str  # the variable every scan queries
    other_variable: str  # registered alongside; must never be read
    time_chunk: int  # registration granularity, steps per chunk
    steps_per_day: int  # from the store's cadence
    window_start: str  # a day-window anchor, chunk-aligned
    month: str  # a chunk-aligned month for the arithmetic count
    bbox: dict = field(default_factory=dict)  # dim -> (lo, hi), inclusive
    storage_options: dict = field(default_factory=dict)


CASES = [
    StoreCase(
        id="arco-era5",
        url="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        variable="2m_temperature",
        other_variable="10m_u_component_of_wind",
        time_chunk=1,
        steps_per_day=24,
        window_start="2020-01-01",
        month="2020-01",
        bbox={"latitude": (36, 44), "longitude": (350, 360)},
        storage_options={"token": "anon"},
    ),
]


# -- Engine runners ------------------------------------------------------------
# One function per engine: run count(*) + avg(variable) over a half-open
# time window (plus an optional bbox) through the engine's idiomatic
# path, returning (rows, mean). Adding an engine is one function and one
# registry entry; tests parametrize over the registry.


def _sql(case, t0, t1, bbox) -> str:
    conds = [f"time >= TIMESTAMP '{t0}'", f"time < TIMESTAMP '{t1}'"]
    for dim, (lo, hi) in bbox.items():
        conds.append(f'"{dim}" BETWEEN {lo} AND {hi}')
    return (
        f'SELECT count(*), avg("{case.variable}") FROM t '
        f"WHERE {' AND '.join(conds)}"
    )


def _duckdb_scan(case, dataset, t0, t1, bbox):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.register("t", dataset)
    n, mean = con.execute(_sql(case, t0, t1, bbox)).fetchone()
    return int(n), float(mean)


def _datafusion_scan(case, dataset, t0, t1, bbox):
    from datafusion import SessionContext

    ctx = SessionContext()
    ctx.register_dataset("t", dataset)
    row = ctx.sql(_sql(case, t0, t1, bbox)).to_pandas().iloc[0]
    return int(row.iloc[0]), float(row.iloc[1])


def _polars_scan(case, dataset, t0, t1, bbox):
    pl = pytest.importorskip("polars")
    lf = pl.scan_pyarrow_dataset(dataset).filter(
        (pl.col("time") >= t0.to_pydatetime())
        & (pl.col("time") < t1.to_pydatetime())
    )
    for dim, (lo, hi) in bbox.items():
        lf = lf.filter((pl.col(dim) >= lo) & (pl.col(dim) <= hi))
    out = lf.select(
        pl.len().alias("n"), pl.col(case.variable).mean().alias("mean")
    ).collect()
    return int(out["n"][0]), float(out["mean"][0])


# DataFusion consumes the dataset through get_fragments() — one
# fragment per source chunk, which scanner-level coalescing
# deliberately leaves untouched (see the contract in
# test_arrow_dataset.py). The other engines scan the whole dataset.
_datafusion_scan.consumes_fragments = True  # type: ignore[attr-defined]

ENGINES = {
    "duckdb": _duckdb_scan,
    "polars": _polars_scan,
    "datafusion": _datafusion_scan,
}


@pytest.fixture(params=sorted(ENGINES), ids=str)
def scan(request):
    """The engine runner under test."""
    return ENGINES[request.param]


# -- Dataset cases -------------------------------------------------------------


@pytest.fixture(scope="module", params=CASES, ids=lambda c: c.id)
def case(request) -> StoreCase:
    c: StoreCase = request.param
    # The expectations below assume windows land on chunk boundaries;
    # reject a miswritten case loudly instead of failing tests obscurely.
    assert c.steps_per_day % c.time_chunk == 0, (
        f"{c.id}: steps_per_day must be a multiple of time_chunk"
    )
    steps_from_midnight = (
        pd.Timestamp(c.window_start) - pd.Timestamp(c.window_start).normalize()
    ) / (pd.Timedelta(days=1) / c.steps_per_day)
    assert steps_from_midnight % c.time_chunk == 0, (
        f"{c.id}: window_start is not chunk-aligned"
    )
    return c


@pytest.fixture(scope="module")
def source(case) -> xr.Dataset:
    ds = xr.open_zarr(
        case.url,
        chunks=None,
        storage_options=case.storage_options,
        consolidated=True,
    )[[case.variable, case.other_variable]]
    # Chunk boundaries are laid out from the store's own time origin;
    # both declared anchors must land on one or the fast-path/read
    # expectations below are silently wrong for this case.
    origin = pd.Timestamp(ds.time.values[0])
    chunk_span = pd.Timedelta(days=1) / case.steps_per_day * case.time_chunk
    for name in ("window_start", "month"):
        anchor = pd.Timestamp(getattr(case, name))
        assert (anchor - origin) % chunk_span == pd.Timedelta(0), (
            f"{case.id}: {name} is not aligned to the store's chunk grid"
        )
    return ds


def _tracked(case, source, variables=None, **kwargs):
    """A pushdown dataset over ``variables`` recording every block read."""
    reads: list = []
    column_sets: list = []
    dataset = XarrayPushdownDataset(
        source[variables or [case.variable]],
        {"time": case.time_chunk},
        prefetch=16,
        _iteration_callback=lambda b, names: (
            reads.append(b),
            column_sets.append(tuple(names)),
        ),
        **kwargs,
    )
    return dataset, reads, column_sets


def _grid_cells(case, source, use_bbox=True) -> int:
    """Cells per time step, inside the case's bbox unless disabled."""
    cells = 1
    for dim in source[case.variable].dims:
        if dim == "time":
            continue
        vals = source[dim].values
        if use_bbox and dim in case.bbox:
            lo, hi = case.bbox[dim]
            cells *= int(((vals >= lo) & (vals <= hi)).sum())
        else:
            cells *= len(vals)
    return cells


def _day_window(case, day=0) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(case.window_start) + pd.Timedelta(days=day)
    return start, start + pd.Timedelta(days=1)


def _day_chunks(case: StoreCase) -> int:
    return case.steps_per_day // case.time_chunk


# -- The contract, engine by engine ---------------------------------------------


def test_windowed_scan_prunes_and_is_exact(scan, case, source):
    # One day + bbox: every engine must push the window down so only the
    # day's chunks are read, and row count and mean must match a direct
    # xarray read of the same window.
    dataset, reads, _ = _tracked(case, source)
    t0, t1 = _day_window(case)
    n, mean = scan(case, dataset, t0, t1, case.bbox)

    assert len(reads) == _day_chunks(case), "the engine did not prune"
    assert n == case.steps_per_day * _grid_cells(case, source)

    window = source[case.variable].sel(time=slice(t0, t1 - pd.Timedelta("1ns")))
    for dim, (lo, hi) in case.bbox.items():
        keep = window[dim][(window[dim] >= lo) & (window[dim] <= hi)]
        window = window.sel({dim: keep})
    # rel=1e-5: the store is float32, so the two sides accumulate the
    # mean in different orders and dtypes.
    assert mean == pytest.approx(float(window.mean()), rel=1e-5)


def test_projection_reads_only_the_referenced_variable(scan, case, source):
    # Two variables registered; a query touching one must never read the
    # other, whichever engine decides the column set.
    dataset, _, column_sets = _tracked(
        case, source, variables=[case.variable, case.other_variable]
    )
    t0, t1 = _day_window(case)
    scan(case, dataset, t0, t1, {})
    read = {name for cols in column_sets for name in cols}
    assert case.variable in read
    assert case.other_variable not in read


def test_dataset_is_rescannable_across_queries(scan, case, source):
    # One wrapper, two disjoint-day queries: the second scan must see
    # fresh state, not a consumed stream or stale pruning from the first.
    dataset, reads, _ = _tracked(case, source)
    cells = _grid_cells(case, source, use_bbox=False)
    for day in range(2):
        reads.clear()
        t0, t1 = _day_window(case, day)
        n, _ = scan(case, dataset, t0, t1, {})
        assert n == case.steps_per_day * cells
        assert len(reads) == _day_chunks(case)


def test_coalescing_merges_consecutive_reads(scan, case, source):
    # The same day window in a handful of merged reads instead of one
    # per chunk; the answer must not change.
    cells = _grid_cells(case, source, use_bbox=False)
    merge_chunks = 6  # chunks per merged read
    dataset, reads, _ = _tracked(
        case,
        source,
        coalesce_rows=merge_chunks * case.time_chunk * cells,
    )
    t0, t1 = _day_window(case)
    n, _ = scan(case, dataset, t0, t1, {})
    assert n == case.steps_per_day * cells
    if getattr(scan, "consumes_fragments", False):
        # Fragment consumers read one source chunk per fragment.
        assert len(reads) == _day_chunks(case)
    else:
        assert len(reads) == -(-_day_chunks(case) // merge_chunks)  # ceil


def test_concurrent_queries_stay_exact(scan, case, source):
    # Engines scan from worker threads; two simultaneous queries over
    # disjoint days of one wrapper must both come back exact.
    dataset, _, _ = _tracked(case, source)
    cells = _grid_cells(case, source, use_bbox=False)
    results: list = [None, None]
    errors: list = []

    def worker(day):
        try:
            t0, t1 = _day_window(case, day)
            results[day] = scan(case, dataset, t0, t1, {})[0]
        except Exception as exc:  # noqa: BLE001 — reported by the assert
            errors.append(str(exc))

    # daemon: a genuinely wedged scan must fail the assert, not keep
    # the interpreter alive after pytest reports it.
    threads = [
        threading.Thread(target=worker, args=(d,), daemon=True) for d in (0, 1)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not any(t.is_alive() for t in threads), "a scan wedged"
    assert not errors
    assert results == [case.steps_per_day * cells] * 2


# -- Dataset-level fast paths (no engine in the loop) ---------------------------


def test_count_rows_fast_path_reads_nothing(case, source):
    # A chunk-aligned month: every surviving chunk is provably inside
    # the range, so the count is pure arithmetic.
    import pyarrow as pa
    import pyarrow.compute as pc

    dataset, reads, _ = _tracked(case, source)
    lo = pd.Timestamp(case.month)
    hi = lo + pd.offsets.MonthBegin(1)
    predicate = (pc.field("time") >= pa.scalar(lo, type=pa.timestamp("ns"))) & (
        pc.field("time") < pa.scalar(hi, type=pa.timestamp("ns"))
    )
    steps = (hi - lo) / pd.Timedelta(days=1) * case.steps_per_day
    count = dataset.count_rows(filter=predicate)
    assert reads == [], "count_rows fast path must not read data"
    assert count == int(steps) * _grid_cells(case, source, use_bbox=False)


def test_polars_lazy_roundtrip_window_reads_only_its_blocks(case, source):
    # Lazy round-trip: construction reads nothing with template coords;
    # a one-day window reads only its own coalesced blocks. Polars is
    # the one engine whose results re-execute over this dataset —
    # DuckDB relations cannot (see limitations.md) and DataFusion's
    # chunked round-trip lives on its native path.
    pl = pytest.importorskip("polars")
    if tuple(int(p) for p in pl.__version__.split(".")[:2]) >= (1, 43):
        # polars 1.43 regressed streaming re-execution over pyarrow
        # datasets: this window takes >10 minutes against 7s on 1.42.
        pytest.skip("polars >= 1.43 streaming re-execution regression")
    cells = _grid_cells(case, source, use_bbox=False)
    merge_chunks = 6
    dataset, reads, _ = _tracked(
        case,
        source,
        variables=[case.variable],
        coalesce_rows=merge_chunks * case.time_chunk * cells,
    )
    lf = pl.scan_pyarrow_dataset(dataset)
    reads.clear()
    lazy = xql.to_dataset(
        lf,
        template=source[[case.variable]],
        chunks={"time": case.steps_per_day},
        coords="template",
    )
    assert reads == [], "lazy construction must not read the source"
    t0, t1 = _day_window(case)
    value = float(
        lazy[case.variable]
        .sel(time=slice(t0, t1 - pd.Timedelta("1ns")))
        .mean()
        .compute()
    )
    assert len(reads) == -(-_day_chunks(case) // merge_chunks)
    oracle = float(
        source[case.variable]
        .sel(time=slice(t0, t1 - pd.Timedelta("1ns")))
        .mean()
    )
    assert value == pytest.approx(oracle, rel=1e-6)
