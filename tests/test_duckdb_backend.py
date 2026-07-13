"""Tests for the DuckDB engine adapter and the engine-agnostic round-trip.

Covers the two seams of the multi-engine design: ``xql.register`` puts a
lazy Dataset on a DuckDB connection, DuckDB executes its own SQL dialect
(including extensions), and ``xql.to_dataset`` rebuilds a labeled
Dataset from the Arrow result.
"""

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

import xarray_sql as xql
from xarray_sql.backends.duckdb import (
    XarrayArrowStream,
    XarrayPushdownDataset,
)


@pytest.fixture
def ds() -> xr.Dataset:
    np.random.seed(7)
    time = pd.date_range("2021-01-01", periods=8, freq="h")
    lat = np.linspace(-10.0, 10.0, 5)
    lon = np.linspace(0.0, 40.0, 6)
    temperature = 15 + 8 * np.random.randn(8, 5, 6)
    precipitation = 10 * np.random.rand(8, 5, 6)
    return xr.Dataset(
        data_vars=dict(
            temperature=(["time", "lat", "lon"], temperature),
            precipitation=(["time", "lat", "lon"], precipitation),
        ),
        coords=dict(time=time, lat=lat, lon=lon),
        attrs=dict(description="Synthetic weather."),
    ).chunk({"time": 4})


@pytest.fixture
def con(ds) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    xql.register(connection, "weather", ds)
    return connection


def test_full_scan_round_trips(con, ds):
    rel = con.sql(
        "SELECT time, lat, lon, temperature, precipitation FROM weather "
        "ORDER BY time, lat, lon"
    )
    out = xql.to_dataset(rel, template=ds)

    xr.testing.assert_allclose(out, ds.compute())
    assert out.attrs == ds.attrs


def test_aggregation_round_trips_on_surviving_dims(con, ds):
    rel = con.sql(
        "SELECT time, AVG(temperature) AS temperature FROM weather "
        "GROUP BY time ORDER BY time"
    )
    out = xql.to_dataset(rel, template=ds)

    expected = ds["temperature"].mean(["lat", "lon"]).compute()
    assert list(out.dims) == ["time"]
    np.testing.assert_allclose(out["temperature"].values, expected.values)


def test_registered_table_is_requeryable(con):
    first = con.sql("SELECT COUNT(*) AS n FROM weather").fetchone()[0]
    second = con.sql("SELECT COUNT(*) AS n FROM weather").fetchone()[0]
    assert first == second == 8 * 5 * 6


def test_registration_is_lazy(ds):
    reads: list = []
    stream = XarrayArrowStream(
        ds, _iteration_callback=lambda b, p: reads.append(b)
    )

    con = duckdb.connect()
    con.register("weather", stream)
    assert reads == []  # registration reads no data

    con.sql("SELECT AVG(temperature) FROM weather").fetchall()
    assert len(reads) > 0  # data was read during query execution


def test_where_filter_yields_sparse_result(con, ds):
    rel = con.sql(
        "SELECT time, lat, lon, temperature FROM weather "
        "WHERE lat > 0 ORDER BY time, lat, lon"
    )
    out = xql.to_dataset(rel, template=ds)

    expected = ds[["temperature"]].sel(lat=ds.lat[ds.lat > 0]).compute()
    xr.testing.assert_allclose(out, expected)


def test_template_sparsity_reindexes_to_full_extent(con, ds):
    rel = con.sql(
        "SELECT time, lat, lon, temperature FROM weather WHERE lat > 0"
    )
    out = xql.to_dataset(rel, template=ds, sparsity="template")

    assert out.sizes == {"time": 8, "lat": 5, "lon": 6}
    assert out["temperature"].isnull().sum() == 8 * 3 * 6  # lat <= 0 cells


def test_duckdb_dialect_and_join(con, ds):
    # Engine-native SQL: DuckDB's date_part plus a join against a local
    # relation — nothing xarray-sql has to understand.
    con.sql("CREATE TABLE labels AS SELECT 0 AS h, 'midnight' AS label")
    rel = con.sql(
        "SELECT w.time, AVG(w.temperature) AS temperature, ANY_VALUE(l.label) AS label "
        "FROM weather w JOIN labels l ON date_part('hour', w.time) = l.h "
        "GROUP BY w.time"
    )
    out = xql.to_dataset(rel, dims=["time"])
    assert out.sizes == {"time": 1}


def test_to_dataset_accepts_plain_arrow_table(ds):
    table = pa.table(
        {
            "time": pd.date_range("2021-01-01", periods=3, freq="h"),
            "temperature": [1.0, 2.0, 3.0],
        }
    )
    out = xql.to_dataset(table, dims=["time"])
    np.testing.assert_allclose(out["temperature"].values, [1.0, 2.0, 3.0])


def test_to_dataset_requires_dims_or_template():
    table = pa.table({"a": [1, 2], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="dims cannot be inferred"):
        xql.to_dataset(table)


def test_to_dataset_rejects_missing_dim_column():
    table = pa.table({"a": [1, 2], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="not columns of the result"):
        xql.to_dataset(table, dims=["z"])


def test_register_splits_mixed_dimension_variables(ds):
    mixed = ds.assign(surface=ds["temperature"].isel(time=0, drop=True))
    con = duckdb.connect()
    xql.register(con, "weather", mixed)

    n_full = con.sql("SELECT COUNT(*) FROM weather_time_lat_lon").fetchone()[0]
    n_surface = con.sql("SELECT COUNT(*) FROM weather_lat_lon").fetchone()[0]
    assert n_full == 8 * 5 * 6
    assert n_surface == 5 * 6


def test_pushdown_dataset_rejects_mixed_dimension_variables(ds):
    mixed = ds.assign(surface=ds["temperature"].isel(time=0, drop=True))
    with pytest.raises(ValueError, match="dimensions must be equal"):
        XarrayPushdownDataset(mixed)


def _tracked_connection(ds):
    """Register ds with an iteration callback; returns (con, reads)."""
    reads: list = []
    dataset = XarrayPushdownDataset(
        ds, _iteration_callback=lambda block, cols: reads.append((block, cols))
    )
    con = duckdb.connect()
    con.register("weather", dataset)
    return con, reads


def test_projection_pushdown_skips_unrequested_variables(ds):
    con, reads = _tracked_connection(ds)
    con.sql("SELECT AVG(temperature) FROM weather").fetchall()
    assert reads  # data was read
    for _, cols in reads:
        assert "precipitation" not in cols


def test_filter_pushdown_prunes_chunks(ds):
    # ds is chunked {"time": 4} -> 2 chunks; this predicate covers only
    # the first chunk, so the second is never loaded.
    con, reads = _tracked_connection(ds)
    n = con.sql(
        "SELECT COUNT(*) FROM weather WHERE time < '2021-01-01 04:00:00'"
    ).fetchone()[0]
    assert n == 4 * 5 * 6
    assert len(reads) == 1


def test_pushed_filter_is_applied_exactly(ds):
    # DuckDB trusts pushed comparison filters and does not re-apply
    # them, so the scan itself must enforce the predicate row-exactly —
    # including inside chunks that pruning keeps.
    con, _ = _tracked_connection(ds)
    out = con.sql(
        "SELECT COUNT(*) FROM weather "
        "WHERE time = '2021-01-01 02:00:00' AND lat > 0"
    ).fetchone()[0]
    expected = int(
        (ds.time == np.datetime64("2021-01-01T02:00:00")).sum()
        * (ds.lat > 0).sum()
        * ds.sizes["lon"]
    )
    assert out == expected


def test_filter_on_variable_outside_projection(ds):
    # The filter references `temperature`, the projection only `lat`;
    # the scan must widen its columns to evaluate the predicate.
    con, _ = _tracked_connection(ds)
    got = con.sql(
        "SELECT COUNT(DISTINCT lat) FROM weather WHERE temperature > 20"
    ).fetchone()[0]
    expected = len(
        np.unique(
            ds.lat.values[np.where((ds.temperature > 20).any(["time", "lon"]))]
        )
    )
    assert got == expected


def test_or_and_in_filters_round_trip(ds):
    con, _ = _tracked_connection(ds)
    rel = con.sql(
        "SELECT time, lat, lon, temperature FROM weather "
        "WHERE lat < -5 OR lat > 5 ORDER BY time, lat, lon"
    )
    out = xql.to_dataset(rel, template=ds)
    mask = (ds.lat < -5) | (ds.lat > 5)
    expected = ds[["temperature"]].sel(lat=ds.lat[mask]).compute()
    xr.testing.assert_allclose(out, expected)


def test_pushdown_dataset_rejects_unchunked_dataset(ds):
    with pytest.raises(ValueError, match="must be chunked"):
        XarrayPushdownDataset(ds.compute())


def test_fully_pruned_scan_returns_empty(con, ds):
    n = con.sql(
        "SELECT COUNT(*) FROM weather WHERE time >= '2022-01-01'"
    ).fetchone()[0]
    assert n == 0
    rel = con.sql(
        "SELECT time, lat, lon, temperature FROM weather "
        "WHERE time >= '2022-01-01'"
    )
    out = xql.to_dataset(rel, template=ds)
    assert out.sizes.get("time", 0) == 0


def test_limit_terminates_early(con):
    rows = con.sql("SELECT time, temperature FROM weather LIMIT 5").fetchall()
    assert len(rows) == 5


def test_descending_coordinate_pruning_is_correct(ds):
    # Latitude stored north→south, like most rasters and ERA5.
    flipped = ds.isel(lat=slice(None, None, -1)).chunk({"lat": 2})
    con = duckdb.connect()
    xql.register(con, "weather", flipped)
    got = con.sql("SELECT COUNT(*) FROM weather WHERE lat > 4").fetchone()[0]
    expected = int((ds.lat > 4).sum()) * ds.sizes["time"] * ds.sizes["lon"]
    assert got == expected


def test_integer_and_string_variables_round_trip():
    ds = xr.Dataset(
        {
            "klass": (["y", "x"], np.arange(12, dtype=np.uint8).reshape(3, 4)),
            "label": (
                ["y", "x"],
                np.array([["a"] * 4, ["b"] * 4, ["c"] * 4]),
            ),
        },
        coords={"y": np.arange(3), "x": np.arange(4)},
    ).chunk({"y": 2})
    con = duckdb.connect()
    xql.register(con, "grid", ds)
    rows = con.sql(
        "SELECT label, SUM(klass) AS total FROM grid "
        "WHERE klass >= 4 GROUP BY label ORDER BY label"
    ).fetchall()
    assert rows == [("b", 22), ("c", 38)]


def test_finely_chunked_dimension_uses_bucketed_pruning():
    # 5000 single-step time chunks exceeds the shadow fanout (1024), so
    # pruning goes through the coarse-then-refine path; an equality in
    # the middle of the axis must load exactly one chunk.
    n = 5000
    ds = xr.Dataset(
        {"v": (["time", "x"], np.random.rand(n, 2))},
        coords={
            "time": pd.date_range("2000-01-01", periods=n, freq="h"),
            "x": np.arange(2),
        },
    ).chunk({"time": 1})
    reads: list = []
    dataset = XarrayPushdownDataset(
        ds, _iteration_callback=lambda block, cols: reads.append(block)
    )
    con = duckdb.connect()
    con.register("t", dataset)

    got = con.sql(
        "SELECT COUNT(*) FROM t WHERE time = '2000-03-15 07:00:00'"
    ).fetchone()[0]
    assert got == 2
    assert len(reads) == 1

    # A range spanning most of the axis stays correct (refinement is
    # skipped when it cannot pay for itself).
    reads.clear()
    got = con.sql(
        "SELECT COUNT(*) FROM t WHERE time >= '2000-01-01 12:00:00'"
    ).fetchone()[0]
    assert got == (n - 12) * 2


def test_register_kwargs_are_forwarded(ds):
    con = duckdb.connect()
    xql.register(con, "weather", ds, prefetch=1, batch_size=7)
    n = con.sql("SELECT COUNT(*) FROM weather").fetchone()[0]
    assert n == 8 * 5 * 6


def test_register_dispatches_to_datafusion():
    # The same entry point serves the default engine.
    ctx = xql.XarrayContext()
    small = xr.Dataset(
        {"v": (["x"], np.arange(4.0))}, coords={"x": np.arange(4)}
    ).chunk({"x": 2})
    xql.register(ctx, "t", small)
    out = ctx.sql("SELECT x, v FROM t ORDER BY x").to_dataset()
    np.testing.assert_allclose(out["v"].values, np.arange(4.0))


def test_register_rejects_unknown_connection(ds):
    with pytest.raises(TypeError, match="No xarray-sql engine adapter"):
        xql.register(object(), "weather", ds)
