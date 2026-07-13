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
from xarray_sql.backends.duckdb import XarrayArrowStream


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


def test_register_rejects_mixed_dimension_variables(ds):
    mixed = ds.assign(surface=ds["temperature"].isel(time=0, drop=True))
    con = duckdb.connect()
    with pytest.raises(ValueError, match="dimensions must be equal"):
        xql.register(con, "weather", mixed)


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
