"""Integration tests: realistic ARCO-ERA5 queries on every ADBC database.

A regional subset of ARCO-ERA5 — Europe at 0.25°, four 6-hourly steps,
surface fields plus temperature on three pressure levels — is registered
on each backend the ``db`` fixture reaches (``tests/_adbc.py``), then
queried the way a user would: area means, a grid-point series, derived
wind speed, threshold counts, per-level statistics, and a join between
the surface and pressure-level tables. Every answer is checked against
xarray computing the same thing from the same subset.

Reads anonymously from a public bucket. Excluded from the CI unit run
(``pytest -m "not integration"``); run deliberately with
``pytest -m integration tests/test_adbc_era5_integration.py``.
"""

import numpy as np
import pytest
import xarray as xr

import xarray_sql as xql

from ._adbc import BACKENDS, Database

pytestmark = pytest.mark.integration

ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
TIMES = slice("2020-07-01T00", "2020-07-01T18")
LATITUDE = slice(60, 35)  # stored north to south
LONGITUDE = slice(0, 30)
LEVELS = [500, 850, 1000]

T2M = "2m_temperature"
U10 = "10m_u_component_of_wind"
V10 = "10m_v_component_of_wind"

NAMES = {
    ("time", "latitude", "longitude"): "surface",
    ("time", "level", "latitude", "longitude"): "atmosphere",
}


@pytest.fixture(scope="module")
def era5() -> xr.Dataset:
    """The subset, read once and held in memory for every backend."""
    ds = xr.open_zarr(
        ERA5,
        chunks=None,
        storage_options={"token": "anon"},
        consolidated=True,
    )
    subset = ds[[T2M, U10, V10, "temperature"]].sel(
        time=ds.time.sel(time=TIMES)[::6],
        latitude=LATITUDE,
        longitude=LONGITUDE,
        level=LEVELS,
    )
    return subset.load().chunk({"time": 1})


@pytest.fixture(scope="module", params=BACKENDS, ids=[b.name for b in BACKENDS])
def db(request, era5):
    """Each backend, with the subset registered once for all its queries."""
    backend = request.param
    database = Database(backend, backend.connect())
    name = database.name("era5")
    xql.register(database.con, name, era5, table_names=NAMES)
    if backend.schemas:
        database.tables = (f"{name}.surface", f"{name}.atmosphere")
    else:
        database.tables = (f"{name}_surface", f"{name}_atmosphere")
    yield database
    database.cleanup()
    database.con.close()


def test_area_mean_time_series(db, era5):
    surface, _ = db.tables
    t2m = db.quoted(T2M)
    cur = db.query(
        f"SELECT time, AVG({t2m}) AS {t2m} FROM {surface} "
        "WHERE latitude BETWEEN 45 AND 55 AND longitude BETWEEN 5 AND 15 "
        "GROUP BY time ORDER BY time"
    )
    out = xql.to_dataset(cur, template=era5)

    expected = (
        era5[T2M]
        .sel(latitude=slice(55, 45), longitude=slice(5, 15))
        .mean(["latitude", "longitude"])
    )
    xr.testing.assert_allclose(out[T2M], expected.compute(), rtol=1e-6)


def test_grid_point_series(db, era5):
    surface, _ = db.tables
    t2m = db.quoted(T2M)
    cur = db.query(
        f"SELECT time, {t2m} FROM {surface} "
        "WHERE latitude = 51.5 AND longitude = 0 ORDER BY time"
    )
    out = xql.to_dataset(cur, template=era5)

    expected = era5[T2M].sel(latitude=51.5, longitude=0.0, drop=True)
    xr.testing.assert_allclose(out[T2M], expected.compute(), rtol=1e-6)


def test_max_wind_speed(db, era5):
    surface, _ = db.tables
    u, v = db.quoted(U10), db.quoted(V10)
    cur = db.query(
        f"SELECT time, MAX(SQRT({u} * {u} + {v} * {v})) AS wind "
        f"FROM {surface} GROUP BY time ORDER BY time"
    )
    out = xql.to_dataset(cur, template=era5)

    speed = np.hypot(era5[U10].astype("float64"), era5[V10].astype("float64"))
    expected = speed.max(["latitude", "longitude"]).rename("wind")
    xr.testing.assert_allclose(out["wind"], expected.compute(), rtol=1e-5)


def test_hot_cell_count(db, era5):
    surface, _ = db.tables
    t2m = db.quoted(T2M)
    cur = db.query(
        f"SELECT time, SUM(CASE WHEN {t2m} > 300 THEN 1 ELSE 0 END) AS hot "
        f"FROM {surface} GROUP BY time ORDER BY time"
    )
    out = xql.to_dataset(cur, template=era5)

    expected = (era5[T2M] > 300).sum(["latitude", "longitude"])
    np.testing.assert_array_equal(out["hot"].values, expected.values)


def test_mean_temperature_per_level(db, era5):
    _, atmosphere = db.tables
    temperature = db.quoted("temperature")
    cur = db.query(
        f"SELECT level, AVG({temperature}) AS {temperature} "
        f"FROM {atmosphere} GROUP BY level ORDER BY level"
    )
    out = xql.to_dataset(cur, template=era5)

    expected = era5["temperature"].mean(["time", "latitude", "longitude"])
    xr.testing.assert_allclose(
        out["temperature"], expected.compute(), rtol=1e-6
    )


def test_surface_to_850_hpa_join(db, era5):
    surface, atmosphere = db.tables
    t2m = db.quoted(T2M)
    temperature = db.quoted("temperature")
    cur = db.query(
        f"SELECT s.time, AVG(s.{t2m} - a.{temperature}) AS difference "
        f"FROM {surface} s JOIN {atmosphere} a "
        "ON s.time = a.time AND s.latitude = a.latitude "
        "AND s.longitude = a.longitude "
        "WHERE a.level = 850 GROUP BY s.time ORDER BY s.time"
    )
    out = xql.to_dataset(cur, template=era5)

    difference = era5[T2M].astype("float64") - era5["temperature"].sel(
        level=850, drop=True
    ).astype("float64")
    expected = difference.mean(["latitude", "longitude"]).rename("difference")
    xr.testing.assert_allclose(out["difference"], expected.compute(), rtol=1e-6)


def test_regional_field_round_trips(db, era5):
    surface, _ = db.tables
    t2m = db.quoted(T2M)
    cur = db.query(
        f"SELECT time, latitude, longitude, {t2m} FROM {surface} "
        "WHERE latitude BETWEEN 50 AND 52 AND longitude BETWEEN 0 AND 2 "
        "ORDER BY time, latitude, longitude"
    )
    out = xql.to_dataset(cur, template=era5)

    expected = era5[T2M].sel(latitude=slice(52, 50), longitude=slice(0, 2))
    xr.testing.assert_allclose(
        out[T2M].sortby("latitude", ascending=False),
        expected.compute(),
    )
    assert out[T2M].dtype == era5[T2M].dtype
    assert out[T2M].attrs == era5[T2M].attrs
