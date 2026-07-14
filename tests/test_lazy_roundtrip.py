"""Lazy chunked round-trip through engines beyond DataFusion.

``xql.to_dataset(result, chunks=...)`` re-executes the engine's query
per accessed window. These tests verify the reconstruction is correct
on Polars frames (DuckDB chunked reconstruction fails fast — see
DuckDBHandle.supports_chunked — while its eager path works), that
laziness is real, and that one-shot
streams are rejected with a clear error.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

import xarray_sql as xql
from xarray_sql.backends.pyarrow import XarrayPushdownDataset


@pytest.fixture
def source() -> xr.Dataset:
    np.random.seed(7)
    return xr.Dataset(
        {
            "t2m": (
                ["time", "lat"],
                np.random.rand(100, 6).astype(np.float64),
            ),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=100, freq="h"),
            "lat": np.linspace(-25.0, 25.0, 6),
        },
        attrs={"title": "synthetic"},
    )


@pytest.fixture
def registered(source):
    """A DuckDB connection with the source registered + a read counter."""
    duckdb = pytest.importorskip("duckdb")

    reads: list[dict] = []
    dataset = XarrayPushdownDataset(
        source, {"time": 10}, _iteration_callback=lambda b, n: reads.append(b)
    )
    con = duckdb.connect()
    con.register("t", dataset)
    return con, reads


def test_duckdb_chunked_fails_fast_with_guidance(source, registered):
    con, _ = registered
    rel = con.sql("SELECT * FROM t")
    # Re-executing a DuckDB relation from dask worker threads
    # intermittently deadlocks inside duckdb-python when the query
    # scans a Python-backed table; the library refuses instead of
    # hanging (see DuckDBHandle.supports_chunked).
    with pytest.raises(NotImplementedError, match="Polars"):
        xql.to_dataset(rel, template=source, chunks={"time": 10})


def test_duckdb_eager_round_trip_through_handle(source, registered):
    con, _ = registered
    rel = con.sql("SELECT * FROM t")
    out = xql.to_dataset(rel, template=source)
    assert not out.chunks
    assert out.attrs == source.attrs
    xr.testing.assert_allclose(out, source)


def test_duckdb_eager_filtered_and_aggregated(source, registered):
    con, _ = registered
    rel = con.sql(
        "SELECT time, avg(t2m) AS t2m FROM t "
        "WHERE lat > 0 GROUP BY time ORDER BY time"
    )
    out = xql.to_dataset(rel, template=source)
    expected = source.t2m.sel(lat=source.lat[source.lat > 0]).mean("lat")
    np.testing.assert_allclose(out.t2m.values, expected.values)


def test_polars_lazyframe_chunked_round_trip(source):
    pl = pytest.importorskip("polars")

    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(source, {"time": 10}))
    out = xql.to_dataset(lf, template=source, chunks={"time": 20})
    assert out.chunks
    xr.testing.assert_allclose(out.compute(), source)

    # Eager path through the same handle (LazyFrame has no stream
    # protocol; the handle executes it once).
    eager = xql.to_dataset(lf, template=source)
    xr.testing.assert_allclose(eager, source)


def test_polars_eager_frame_is_reexecutable(source):
    pl = pytest.importorskip("polars")

    frame = pl.DataFrame(
        {
            "time": np.repeat(source.time.values, 6),
            "lat": np.tile(source.lat.values, 100),
            "t2m": source.t2m.values.ravel(),
        }
    )
    out = xql.to_dataset(frame, template=source, chunks={"time": 50})
    xr.testing.assert_allclose(out.compute(), source)


def test_one_shot_stream_with_chunks_raises(source, registered):
    con, _ = registered
    table = con.sql("SELECT * FROM t").to_arrow_table()
    with pytest.raises(TypeError, match="re-executable"):
        xql.to_dataset(table, template=source, chunks={"time": 10})


def test_inherit_without_chunked_source_falls_back_to_eager(source, registered):
    con, _ = registered
    rel = con.sql("SELECT * FROM t")
    out = xql.to_dataset(rel, template=source, chunks="inherit")
    # The in-memory template has no multi-chunk dim: eager, dense.
    assert not out.chunks
    xr.testing.assert_allclose(out, source)


def test_stepped_indexer_uses_value_lists(source):
    pl = pytest.importorskip("polars")

    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(source, {"time": 10}))
    out = xql.to_dataset(lf, template=source, chunks={"time": 10})
    # A step-2 selection is not a contiguous coordinate range; the
    # values path must return exactly the requested rows.
    stepped = out.t2m.isel(time=slice(10, 30, 2)).compute()
    np.testing.assert_allclose(
        stepped.values, source.t2m.isel(time=slice(10, 30, 2)).values
    )


def test_descending_coordinate_windows():
    pl = pytest.importorskip("polars")

    desc = xr.Dataset(
        {"v": (["lat"], np.arange(8.0))},
        coords={"lat": np.linspace(70.0, 0.0, 8)},  # descending, like ERA5
    )
    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(desc, {"lat": 4}))
    out = xql.to_dataset(
        lf,
        template=desc,
        chunks={"lat": 4},
        coords="template",
    )
    xr.testing.assert_allclose(out.compute(), desc)
    window = out.v.isel(lat=slice(2, 6)).compute()
    np.testing.assert_allclose(window.values, desc.v.isel(lat=slice(2, 6)))


def test_unsorted_template_coords_window_exactly():
    pl = pytest.importorskip("polars")

    # Template coords are used verbatim, so the backend can see a
    # non-monotonic coordinate array. A contiguous positional window
    # like 1:3 then has monotonic values [7, 55], but the value range
    # [7, 55] also admits 23 at position 3 — the scatter would write
    # that unrequested row over a requested cell. Windows over a
    # non-monotonic coordinate must use explicit value lists.
    src = xr.Dataset(
        {"v": (["x"], np.arange(4.0))},
        coords={"x": np.array([102.0, 7.0, 55.0, 23.0])},
    )
    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(src, {"x": 4}))
    out = xql.to_dataset(lf, template=src, chunks={"x": 4}, coords="template")
    window = out.v.isel(x=slice(1, 3)).compute()
    np.testing.assert_array_equal(window.values, [1.0, 2.0])
    xr.testing.assert_allclose(out.compute(), src)


def test_polars_float_value_windows_are_exact():
    pl = pytest.importorskip("polars")

    # Non-representable float coordinates: upstream Polars is_in drops
    # them (silently matching nothing); the handle's degenerate-range
    # translation must return exactly the requested rows.
    src = xr.Dataset(
        {"v": (["lat", "t"], np.arange(38.0).reshape(19, 2))},
        coords={"lat": np.linspace(-45.0, 45.0, 19), "t": [0.0, 1.0]},
    )
    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(src, {"lat": 5}))
    out = xql.to_dataset(lf, template=src, chunks={"lat": 5})
    # A stepped (non-contiguous) selection forces the value-list path.
    picked = out.v.isel(lat=slice(1, 12, 2)).compute()
    np.testing.assert_allclose(
        picked.values, src.v.isel(lat=slice(1, 12, 2)).values
    )


def test_max_result_bytes_guards_stream_collection(source, registered):
    con, _ = registered
    rel = con.sql("SELECT * FROM t")
    with pytest.raises(ValueError, match="max_result_bytes"):
        xql.to_dataset(rel, template=source, max_result_bytes=1_000)
    # A generous budget passes untouched.
    out = xql.to_dataset(rel, template=source, max_result_bytes=10**9)
    xr.testing.assert_allclose(out, source)


def test_polars_large_float_value_lists_stay_flat():
    pl = pytest.importorskip("polars")

    # 5000 stepped float values in a single window: a left-deep OR
    # chain plans quadratically at this size (seconds per window); the
    # flat any_horizontal translation must stay exact and quick.
    n = 10_000
    src = xr.Dataset(
        {"v": (["x"], np.arange(float(n)))},
        coords={"x": np.linspace(-45.0, 45.0, n)},
    )
    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(src, {"x": n}))
    out = xql.to_dataset(lf, template=src, chunks={"x": n})
    picked = out.v.isel(x=slice(1, None, 2)).compute()
    np.testing.assert_allclose(
        picked.values, src.v.isel(x=slice(1, None, 2)).values
    )


def test_collect_streaming_falls_back_on_older_polars():
    from xarray_sql.lazyscan import _collect_streaming

    class OldLazyFrame:
        # Pre-1.25 collect(): no ``engine`` keyword.
        def collect(self):
            return "collected"

    assert _collect_streaming(OldLazyFrame()) == "collected"


def test_max_result_bytes_guards_polars_lazyframe(source):
    pl = pytest.importorskip("polars")

    # The LazyFrame eager fallback collects inside the engine before
    # any batch surfaces; with a budget set it must stream through
    # collect_batches so the guard fires before full materialization.
    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(source, {"time": 10}))
    with pytest.raises(ValueError, match="max_result_bytes"):
        xql.to_dataset(lf, template=source, max_result_bytes=1_000)
    out = xql.to_dataset(lf, template=source, max_result_bytes=10**9)
    xr.testing.assert_allclose(out, source)


def test_max_result_bytes_guards_table_only_results(source, registered):
    con, _ = registered
    table = con.sql("SELECT * FROM t").to_arrow_table()

    class TableOnly:
        # The narrowest result surface: to_arrow_table() materializes
        # in full before the budget can see a batch, so the guard runs
        # on the materialized size.
        def __init__(self, t):
            self._t = t

        def to_arrow_table(self):
            return self._t

    with pytest.raises(ValueError, match="max_result_bytes"):
        xql.to_dataset(
            TableOnly(table), template=source, max_result_bytes=1_000
        )
    out = xql.to_dataset(
        TableOnly(table), template=source, max_result_bytes=10**9
    )
    xr.testing.assert_allclose(out, source)


def test_max_result_bytes_guards_dense_blowup(registered):
    con, _ = registered
    # A sparse diagonal: tiny Arrow payload, huge dense grid (the
    # coordinate product), so the dense-size check must fire even
    # though the stream fits the budget.
    diag = pa.table(
        {
            "a": np.arange(3000.0),
            "b": np.arange(3000.0),
            "v": np.ones(3000),
        }
    )
    with pytest.raises(ValueError, match="dense reconstruction"):
        xql.to_dataset(diag, dims=["a", "b"], max_result_bytes=10_000_000)


def test_duckdb_spill_chunked_round_trip(source, registered, tmp_path):
    con, reads = registered
    rel = con.sql("SELECT * FROM t")
    reads.clear()
    out = xql.to_dataset(
        rel, template=source, chunks={"time": 10}, spill=tmp_path
    )
    spilled = list(tmp_path.glob("*.parquet"))
    assert len(spilled) == 1
    # The source was streamed exactly once (10 chunks), during the spill.
    assert len(reads) == 10
    assert out.chunks
    reads.clear()
    xr.testing.assert_allclose(out.compute(), source)
    # Windows re-execute against the Parquet file, not the source.
    assert reads == []


def test_duckdb_spill_filtered_aggregation(source, registered, tmp_path):
    con, _ = registered
    rel = con.sql(
        "SELECT time, avg(t2m) AS t2m FROM t "
        "WHERE lat > 0 GROUP BY time ORDER BY time"
    )
    out = xql.to_dataset(
        rel, template=source, chunks={"time": 25}, spill=tmp_path
    )
    expected = source.t2m.sel(lat=source.lat[source.lat > 0]).mean("lat")
    np.testing.assert_allclose(out.t2m.compute().values, expected.values)


def test_one_shot_table_spill_chunked(source, registered, tmp_path):
    con, _ = registered
    table = con.sql("SELECT * FROM t").to_arrow_table()
    out = xql.to_dataset(
        table, template=source, chunks={"time": 10}, spill=tmp_path
    )
    assert out.chunks
    xr.testing.assert_allclose(out.compute(), source)


def test_spill_file_removed_when_dataset_dies(source, registered, tmp_path):
    import gc

    con, _ = registered
    rel = con.sql("SELECT * FROM t")
    out = xql.to_dataset(
        rel, template=source, chunks={"time": 10}, spill=tmp_path
    )
    assert list(tmp_path.glob("*.parquet"))
    del out
    gc.collect()
    assert list(tmp_path.glob("*.parquet")) == []


def test_spill_requires_chunks(source, registered):
    con, _ = registered
    rel = con.sql("SELECT * FROM t")
    with pytest.raises(ValueError, match="spill= only applies"):
        xql.to_dataset(rel, template=source, spill=True)


def test_duckdb_handle_runner_stopped_when_handle_dies(source, registered):
    import gc

    from xarray_sql.lazyscan import DuckDBHandle

    con, _ = registered
    handle = DuckDBHandle(con.sql("SELECT * FROM t"))
    runner = handle._runner
    del handle
    gc.collect()
    # A shut-down executor refuses new work — the observable contract
    # that the handle's dedicated engine thread has been told to exit.
    with pytest.raises(RuntimeError):
        runner.submit(lambda: None)


def test_polars_spill_uses_streaming_sink(source, tmp_path):
    pl = pytest.importorskip("polars")

    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(source, {"time": 10}))
    out = xql.to_dataset(
        lf, template=source, chunks={"time": 20}, spill=tmp_path
    )
    xr.testing.assert_allclose(out.compute(), source)
