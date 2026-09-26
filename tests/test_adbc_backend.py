"""Tests for the ADBC engine adapter.

``xql.register`` ingests a Dataset into any database reachable through an
ADBC driver, and ``xql.to_dataset`` rebuilds a labeled Dataset from the
driver's Arrow cursor. Two drivers cover the two shapes of database:
SQLite, which has no schemas and stores timestamps as text, and DuckDB's
built-in ADBC driver, which has schemas and native temporal types.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

import xarray_sql as xql
from xarray_sql.backends.adbc import _clickhouse_ddl

dbapi = pytest.importorskip("adbc_driver_manager.dbapi")
sqlite_dbapi = pytest.importorskip("adbc_driver_sqlite.dbapi")

NAMES = {
    ("time", "lat", "lon"): "surface",
    ("time", "level", "lat", "lon"): "atmosphere",
}


def _duckdb_driver_path() -> str | None:
    """Path of the shared library holding DuckDB's ADBC entrypoint."""
    for module in ("_duckdb", "duckdb.duckdb"):
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            continue
        if spec is not None and spec.origin:
            return spec.origin
    return None


def _connect(driver: str):
    if driver == "sqlite":
        return sqlite_dbapi.connect()
    path = _duckdb_driver_path()
    if path is None:
        pytest.skip("duckdb is not installed")
    return dbapi.connect(driver=path, entrypoint="duckdb_adbc_init")


@pytest.fixture(params=["sqlite", "duckdb"])
def con(request):
    connection = _connect(request.param)
    yield connection
    connection.close()


@pytest.fixture
def sqlite_con():
    connection = _connect("sqlite")
    yield connection
    connection.close()


@pytest.fixture
def duckdb_con():
    connection = _connect("duckdb")
    yield connection
    connection.close()


@pytest.fixture
def ds() -> xr.Dataset:
    np.random.seed(3)
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


def _query(con, sql: str):
    cur = con.cursor()
    cur.execute(sql)
    return cur


def test_full_scan_round_trips(con, ds):
    xql.register(con, "weather", ds)

    cur = _query(
        con,
        "SELECT time, lat, lon, temperature, precipitation FROM weather "
        "ORDER BY time, lat, lon",
    )
    out = xql.to_dataset(cur, template=ds)

    xr.testing.assert_allclose(out, ds.compute())
    assert out.attrs == ds.attrs


def test_aggregation_round_trips_on_surviving_dims(con, ds):
    xql.register(con, "weather", ds)

    cur = _query(
        con,
        "SELECT lat, lon, AVG(temperature) AS temperature FROM weather "
        "GROUP BY lat, lon ORDER BY lat, lon",
    )
    out = xql.to_dataset(cur, template=ds)

    assert out.temperature.dims == ("lat", "lon")
    xr.testing.assert_allclose(
        out.temperature, ds.temperature.mean("time").compute()
    )


def test_chunked_round_trip_spills_the_cursor(duckdb_con, ds):
    # DuckDB keeps `time` a timestamp; SQLite returns it as text, which
    # only the eager round-trip recovers.
    xql.register(duckdb_con, "weather", ds)

    cur = _query(
        duckdb_con,
        "SELECT time, lat, lon, temperature FROM weather "
        "ORDER BY time, lat, lon",
    )
    out = xql.to_dataset(cur, template=ds, chunks={"time": 2}, spill=True)

    assert out.temperature.chunks is not None
    xr.testing.assert_allclose(
        out.temperature.compute(), ds.temperature.compute()
    )


def test_existing_table_is_not_overwritten_by_default(con, ds):
    xql.register(con, "weather", ds)

    with pytest.raises(dbapi.Error):
        xql.register(con, "weather", ds)


def test_replace_mode_recreates_the_table(con, ds):
    xql.register(con, "weather", ds)
    xql.register(con, "weather", ds.isel(time=slice(0, 4)), mode="replace")

    count = _query(con, "SELECT COUNT(*) FROM weather").fetchone()[0]
    assert count == 4 * 5 * 6


def test_append_mode_adds_rows(con, ds):
    xql.register(con, "weather", ds.isel(time=slice(0, 4)))
    xql.register(con, "weather", ds.isel(time=slice(4, 8)), mode="append")

    cur = _query(
        con,
        "SELECT time, lat, lon, temperature, precipitation FROM weather "
        "ORDER BY time, lat, lon",
    )
    xr.testing.assert_allclose(xql.to_dataset(cur, template=ds), ds.compute())


def test_temporary_table_is_queryable(con, ds):
    xql.register(con, "weather", ds, temporary=True)

    count = _query(con, "SELECT COUNT(*) FROM weather").fetchone()[0]
    assert count == 8 * 5 * 6


def test_mixed_dimensions_register_in_a_schema(duckdb_con, mixed_ds):
    xql.register(duckdb_con, "era5", mixed_ds, table_names=NAMES)

    cur = _query(duckdb_con, "SELECT AVG(t2m) FROM era5.surface")
    assert cur.fetchone()[0] == pytest.approx(float(mixed_ds.t2m.mean()))
    cur = _query(
        duckdb_con,
        "SELECT time, level, lat, lon, temperature FROM era5.atmosphere "
        "ORDER BY time, level, lat, lon",
    )
    out = xql.to_dataset(cur, template=mixed_ds[["temperature"]])
    xr.testing.assert_allclose(out.temperature, mixed_ds.temperature.compute())


def test_mixed_dimensions_fall_back_to_flat_names(sqlite_con, mixed_ds):
    with pytest.warns(RuntimeWarning, match="flat"):
        xql.register(sqlite_con, "era5", mixed_ds, table_names=NAMES)

    cur = _query(sqlite_con, "SELECT AVG(t2m) FROM era5_surface")
    assert cur.fetchone()[0] == pytest.approx(float(mixed_ds.t2m.mean()))
    count = _query(sqlite_con, "SELECT COUNT(*) FROM era5_atmosphere")
    assert count.fetchone()[0] == 6 * 2 * 3 * 4


def test_temporary_mixed_dimensions_use_flat_names(duckdb_con, mixed_ds):
    xql.register(
        duckdb_con, "era5", mixed_ds, table_names=NAMES, temporary=True
    )

    count = _query(duckdb_con, "SELECT COUNT(*) FROM era5_surface")
    assert count.fetchone()[0] == 6 * 3 * 4


def test_clickhouse_float_columns_are_nullable():
    # The scan writes NaN as an Arrow null, so aggregates skip it; a plain
    # Float64 column would store that null as 0.
    schema = pa.schema(
        [
            pa.field("time", pa.timestamp("ns")),
            pa.field("lat", pa.float64()),
            pa.field("t2m", pa.float64()),
            pa.field("sst", pa.float32()),
        ]
    )
    [ddl] = _clickhouse_ddl(
        "weather",
        schema,
        ("time", "lat"),
        mode="create",
        temporary=False,
        database=None,
    )
    assert '"t2m" Nullable(Float64)' in ddl
    assert '"sst" Nullable(Float32)' in ddl
    assert '"lat" Float64,' in ddl  # sort keys stay non-Nullable


@pytest.fixture
def postgres_con():
    uri = os.environ.get("XARRAY_SQL_TEST_POSTGRES_URI")
    if not uri:
        pytest.skip(
            "set XARRAY_SQL_TEST_POSTGRES_URI to run against PostgreSQL"
        )
    postgres = pytest.importorskip("adbc_driver_postgresql.dbapi")
    connection = postgres.connect(uri)
    yield connection
    connection.rollback()
    connection.close()


def test_postgres_schema_failure_explains_the_aborted_transaction(
    postgres_con, mixed_ds
):
    # PostgreSQL rejects schema names starting with `pg_`, so CREATE SCHEMA
    # fails here for any user, and a failed statement aborts the
    # transaction: no fallback ingest could run after it.
    with pytest.raises(RuntimeError, match="rollback"):
        xql.register(postgres_con, "pg_era5", mixed_ds, table_names=NAMES)


def test_postgres_uses_an_existing_schema(postgres_con, mixed_ds):
    with postgres_con.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS era5")
    xql.register(postgres_con, "era5", mixed_ds, table_names=NAMES)

    count = _query(postgres_con, "SELECT COUNT(*) FROM era5.surface")
    assert count.fetchone()[0] == 6 * 3 * 4


@pytest.fixture
def clickhouse_con():
    uri = os.environ.get("XARRAY_SQL_TEST_CLICKHOUSE_URI")
    driver = os.environ.get("XARRAY_SQL_TEST_CLICKHOUSE_DRIVER")
    if not (uri and driver):
        pytest.skip(
            "set XARRAY_SQL_TEST_CLICKHOUSE_URI and "
            "XARRAY_SQL_TEST_CLICKHOUSE_DRIVER to run against ClickHouse"
        )
    connection = dbapi.connect(driver=driver, db_kwargs={"uri": uri})
    for statement in [
        "DROP TABLE IF EXISTS weather",
        "DROP DATABASE IF EXISTS era5",
    ]:
        _query(connection, statement).close()
    yield connection
    connection.close()


def test_clickhouse_round_trips(clickhouse_con, ds):
    xql.register(clickhouse_con, "weather", ds)

    cur = _query(
        clickhouse_con,
        "SELECT time, lat, lon, temperature, precipitation FROM weather "
        "ORDER BY time, lat, lon",
    )
    xr.testing.assert_allclose(xql.to_dataset(cur, template=ds), ds.compute())


def test_clickhouse_time_literals_mean_utc(clickhouse_con, ds):
    xql.register(clickhouse_con, "weather", ds)

    cur = _query(
        clickhouse_con,
        "SELECT time, lat, lon, temperature FROM weather "
        "WHERE time >= '2021-01-01 04:00:00' ORDER BY time, lat, lon",
    )
    out = xql.to_dataset(cur, template=ds)

    expected = ds.temperature.isel(time=slice(4, None))
    xr.testing.assert_allclose(out.temperature, expected.compute())


def test_clickhouse_replace_then_append(clickhouse_con, ds):
    xql.register(clickhouse_con, "weather", ds)
    xql.register(
        clickhouse_con, "weather", ds.isel(time=slice(0, 4)), mode="replace"
    )
    xql.register(
        clickhouse_con, "weather", ds.isel(time=slice(4, 8)), mode="append"
    )

    cur = _query(
        clickhouse_con,
        "SELECT time, lat, lon, temperature, precipitation FROM weather "
        "ORDER BY time, lat, lon",
    )
    xr.testing.assert_allclose(xql.to_dataset(cur, template=ds), ds.compute())


def test_clickhouse_temporary_table_is_queryable(clickhouse_con, ds):
    xql.register(clickhouse_con, "weather", ds, temporary=True)

    count = _query(clickhouse_con, "SELECT COUNT(*) FROM weather")
    assert count.fetchone()[0] == 8 * 5 * 6


def test_clickhouse_mixed_dimensions_register_in_a_database(
    clickhouse_con, mixed_ds
):
    xql.register(clickhouse_con, "era5", mixed_ds, table_names=NAMES)

    cur = _query(
        clickhouse_con,
        "SELECT time, level, lat, lon, temperature FROM era5.atmosphere "
        "ORDER BY time, level, lat, lon",
    )
    out = xql.to_dataset(cur, template=mixed_ds[["temperature"]])
    xr.testing.assert_allclose(out.temperature, mixed_ds.temperature.compute())


def test_clickhouse_keeps_missing_values(clickhouse_con, ds):
    holed = ds.copy(deep=True)
    holed["temperature"][0, 0, 0] = np.nan
    xql.register(clickhouse_con, "weather", holed)

    with _query(clickhouse_con, "SELECT AVG(temperature) FROM weather") as avg:
        mean = avg.fetchone()[0]
    assert mean == pytest.approx(float(holed.temperature.mean()))
    cur = _query(
        clickhouse_con,
        "SELECT time, lat, lon, temperature FROM weather "
        "ORDER BY time, lat, lon",
    )
    out = xql.to_dataset(cur, template=holed)
    xr.testing.assert_allclose(out.temperature, holed.temperature.compute())
