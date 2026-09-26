"""Tests for serving Datasets over Arrow Flight SQL.

A ``FlightSQLServer`` hosts lazily registered Datasets; the client here
is ADBC's Flight SQL driver, the same one remote users would connect
with, so the tests exercise the real wire protocol end to end.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xarray_sql as xql

flight_sql = pytest.importorskip("adbc_driver_flightsql.dbapi")

NAMES = {
    ("time", "lat", "lon"): "surface",
    ("time", "level", "lat", "lon"): "atmosphere",
}


@pytest.fixture
def ds() -> xr.Dataset:
    np.random.seed(5)
    return xr.Dataset(
        data_vars=dict(
            temperature=(["time", "lat", "lon"], np.random.randn(8, 5, 6)),
            precipitation=(["time", "lat", "lon"], np.random.rand(8, 5, 6)),
        ),
        coords=dict(
            time=pd.date_range("2021-01-01", periods=8, freq="h"),
            lat=np.linspace(-10.0, 10.0, 5),
            lon=np.linspace(0.0, 40.0, 6),
        ),
        attrs=dict(description="Synthetic weather."),
    ).chunk({"time": 4})


@pytest.fixture
def mixed_ds() -> xr.Dataset:
    np.random.seed(11)
    return xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], np.random.rand(6, 3, 4)),
            "temperature": (
                ["time", "level", "lat", "lon"],
                np.random.rand(6, 2, 3, 4),
            ),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=6, freq="D"),
            "lat": np.linspace(-90, 90, 3),
            "lon": np.linspace(-180, 180, 4),
            "level": [500, 1000],
        },
    ).chunk({"time": 2})


@pytest.fixture
def server(ds):
    with xql.serve({"weather": ds}) as running:
        yield running


@pytest.fixture
def con(server):
    connection = flight_sql.connect(server.uri)
    yield connection
    connection.close()


def _query(con, sql: str):
    cur = con.cursor()
    cur.execute(sql)
    return cur


def test_full_scan_round_trips(con, ds):
    cur = _query(
        con,
        "SELECT time, lat, lon, temperature, precipitation FROM weather "
        "ORDER BY time, lat, lon",
    )
    out = xql.to_dataset(cur, template=ds)

    xr.testing.assert_allclose(out, ds.compute())
    assert out.attrs == ds.attrs


def test_filtered_aggregation_round_trips(con, ds):
    cur = _query(
        con,
        "SELECT lat, lon, AVG(temperature) AS temperature FROM weather "
        "WHERE time >= '2021-01-01T04:00:00' "
        "GROUP BY lat, lon ORDER BY lat, lon",
    )
    out = xql.to_dataset(cur, template=ds)

    expected = ds.temperature.isel(time=slice(4, None)).mean("time")
    xr.testing.assert_allclose(out.temperature, expected.compute())


def test_mixed_dimensions_are_served_as_a_schema(mixed_ds):
    server = xql.FlightSQLServer()
    xql.register(server, "era5", mixed_ds, table_names=NAMES)
    with server.serve():
        con = flight_sql.connect(server.uri)
        avg = _query(con, "SELECT AVG(t2m) FROM era5.surface").fetchone()[0]
        count = _query(con, "SELECT COUNT(*) FROM era5.atmosphere")
        count = count.fetchone()[0]
        con.close()

    assert avg == pytest.approx(float(mixed_ds.t2m.mean()))
    assert count == 6 * 2 * 3 * 4


def test_datasets_registered_while_serving_are_visible(server, con, ds):
    server.register("later", ds[["precipitation"]])

    count = _query(con, "SELECT COUNT(*) FROM later").fetchone()[0]
    assert count == 8 * 5 * 6


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE EXTERNAL TABLE leak STORED AS CSV LOCATION '/etc/hosts'",
        "COPY (SELECT 1) TO '/tmp/xarray-sql-flight-leak.csv'",
        "CREATE TABLE copy AS SELECT * FROM weather",
        "SET datafusion.execution.batch_size = 1",
    ],
)
def test_only_queries_are_allowed(con, statement):
    with pytest.raises(flight_sql.Error):
        _query(con, statement).fetchall()


def test_tables_are_discoverable(con):
    objects = con.adbc_get_objects(depth="tables").read_all().to_pylist()
    tables = {
        table["table_name"]
        for catalog in objects
        for schema in catalog["catalog_db_schemas"] or []
        for table in schema["db_schema_tables"] or []
    }

    assert "weather" in tables


def test_shutdown_stops_the_server(ds):
    with xql.serve({"weather": ds}) as server:
        assert server.is_running
        assert server.uri == f"grpc://127.0.0.1:{server.port}"

    assert not server.is_running
