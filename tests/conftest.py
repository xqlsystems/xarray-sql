import pytest

import numpy as np
import pandas as pd
import xarray as xr

from ._adbc import BACKENDS, Database


def rand_wx(start: str, end: str) -> xr.Dataset:
    np.random.seed(42)
    lat = np.linspace(-90, 90, num=720)
    lon = np.linspace(-180, 180, num=1440)
    time = pd.date_range(start, end, freq="h")
    level = np.array([1000, 500], dtype=np.int32)
    reference_time = pd.Timestamp(start)
    temperature = 15 + 8 * np.random.randn(720, 1440, len(time), len(level))
    precipitation = 10 * np.random.rand(720, 1440, len(time), len(level))
    return xr.Dataset(
        data_vars=dict(
            temperature=(["lat", "lon", "time", "level"], temperature),
            precipitation=(["lat", "lon", "time", "level"], precipitation),
        ),
        coords=dict(
            lat=lat,
            lon=lon,
            time=time,
            level=level,
            reference_time=reference_time,
        ),
        attrs=dict(description="Random weather."),
    )


def create_large_dataset(time_steps=1000, lat_points=100, lon_points=100):
    """Create a large xarray dataset for memory testing."""
    np.random.seed(42)

    time = pd.date_range("2020-01-01", periods=time_steps, freq="h")
    lat = np.linspace(-90, 90, lat_points)
    lon = np.linspace(-180, 180, lon_points)

    temp_data = np.random.rand(time_steps, lat_points, lon_points) * 40 - 10
    precip_data = np.random.rand(time_steps, lat_points, lon_points) * 100

    return xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], temp_data),
            "precipitation": (["time", "lat", "lon"], precip_data),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


@pytest.fixture
def air():
    ds = xr.tutorial.open_dataset("air_temperature")
    chunks = {"time": 240}
    return ds.chunk(chunks)


@pytest.fixture
def air_small(air):
    return air.isel(
        time=slice(0, 12), lat=slice(0, 11), lon=slice(0, 10)
    ).chunk({"time": 240})


@pytest.fixture
def randwx():
    return rand_wx("1995-01-13T00", "1995-01-13T01")


@pytest.fixture
def large_ds():
    return create_large_dataset().chunk({"time": 25})


@pytest.fixture
def air_dataset_small():
    ds = xr.tutorial.open_dataset("air_temperature").chunk({"time": 240})
    return ds.isel(time=slice(0, 12), lat=slice(0, 11), lon=slice(0, 10))


@pytest.fixture
def air_dataset_large():
    return xr.tutorial.open_dataset("air_temperature").chunk({"time": 240})


@pytest.fixture
def rasm_ds():
    """rasm uses cftime.DatetimeNoLeap (noleap / 365_day) for time."""
    return xr.tutorial.open_dataset("rasm")


@pytest.fixture
def weather_dataset():
    ds = rand_wx("2023-01-01T00", "2023-01-01T12")
    return ds.isel(time=slice(0, 6), lat=slice(0, 10), lon=slice(0, 10)).chunk(
        {"time": 3}
    )


@pytest.fixture
def synthetic_dataset():
    return create_large_dataset(
        time_steps=50, lat_points=20, lon_points=20
    ).chunk({"time": 25})


@pytest.fixture
def station_dataset():
    return xr.Dataset(
        {
            "station_id": (["station"], [1, 2, 3, 4, 5]),
            "elevation": (["station"], [100, 250, 500, 750, 1000]),
            "name": (
                ["station"],
                [
                    "Station_A",
                    "Station_B",
                    "Station_C",
                    "Station_D",
                    "Station_E",
                ],
            ),
        }
    ).chunk({"station": 5})


@pytest.fixture
def air_and_stations():
    air = (
        xr.tutorial.open_dataset("air_temperature")
        .isel(time=slice(0, 12), lat=slice(0, 5), lon=slice(0, 8))
        .chunk({"time": 6})
    )
    stations = xr.Dataset(
        {
            "station_id": (["station"], [101, 102, 103]),
            "lat": (
                ["station"],
                [air.lat.values[0], air.lat.values[2], air.lat.values[4]],
            ),
            "lon": (
                ["station"],
                [air.lon.values[1], air.lon.values[3], air.lon.values[5]],
            ),
            "elevation": (["station"], [100, 250, 500]),
        }
    ).chunk({"station": 3})
    return air, stations


@pytest.fixture(params=BACKENDS, ids=[b.name for b in BACKENDS])
def db(request):
    """A connection to each available ADBC backend (see ``tests/_adbc.py``)."""
    backend = request.param
    database = Database(backend, backend.connect())
    yield database
    database.cleanup()
    database.con.close()
