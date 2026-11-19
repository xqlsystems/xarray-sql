"""A benchmark tracked over time in CI (see CONTRIBUTING.md).

It uses the ``benchmark`` fixture of pytest-benchmark
(https://github.com/ionelmc/pytest-benchmark), a pytest plugin that runs
a function repeatedly and reports its timings. On every push to main, CI
stores the timings with github-action-benchmark
(https://github.com/benchmark-action/github-action-benchmark), which
keeps them as a history and flags a commit that looks like a regression.

Starts small: one benchmark, over an in-memory Dataset that needs no
network, so a regression in xarray-sql, DataFusion or the in-memory
table adapter shows up in that history.
"""

import numpy as np
import pytest
import xarray as xr

from xarray_sql import Context

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="module")
def grid() -> xr.Dataset:
    """A synthetic (time, lat, lon) grid: 100 x 50 x 50 cells, no I/O."""
    rng = np.random.default_rng(0)
    time, lat, lon = 100, 50, 50
    return xr.Dataset(
        {
            "temperature": (
                ("time", "lat", "lon"),
                rng.normal(size=(time, lat, lon)),
            ),
        },
        coords={
            "time": np.arange(time),
            "lat": np.arange(lat),
            "lon": np.arange(lon),
        },
    )


def test_groupby_mean_over_time(benchmark, grid):
    """A climatology query: mean over time, grouped by lat and lon."""
    ctx = Context()
    ctx.from_dataset("grid", grid)

    def run():
        return ctx.sql(
            "SELECT lat, lon, AVG(temperature) AS mean_temp "
            'FROM "grid" GROUP BY lat, lon ORDER BY lat, lon'
        ).to_dataset()

    result = benchmark(run)
    assert result.sizes["lat"] == grid.sizes["lat"]
    assert result.sizes["lon"] == grid.sizes["lon"]
    expected = grid["temperature"].mean("time")
    np.testing.assert_allclose(result["mean_temp"].values, expected.values)


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> str:
    """A Zarr store of 20 variables, one time step per chunk, NaN on land.

    It's shaped like an ocean model's training data, whose loader reads a
    window of time steps of every variable.
    """
    rng = np.random.default_rng(0)
    time, lat, lon = 8, 90, 180
    land = rng.random((lat, lon)) < 0.3
    data = {}
    for v in range(20):
        values = rng.normal(size=(time, lat, lon)).astype("float32")
        values[:, land] = np.nan
        data[f"v{v:02d}"] = (("time", "lat", "lon"), values)
    ds = xr.Dataset(
        data,
        coords={
            "time": np.arange(time),
            "lat": np.arange(lat),
            "lon": np.arange(lon),
        },
    )
    path = str(tmp_path_factory.mktemp("zarr") / "ocean.zarr")
    ds.chunk({"time": 1}).to_zarr(path, zarr_format=3, consolidated=False)
    return path


@pytest.mark.parametrize("dask", [False, True], ids=["lazy", "dask"])
def test_read_a_window_of_every_variable_from_zarr(benchmark, store, dask):
    """A training loader's read: 4 time steps of all 20 variables."""
    ds = xr.open_zarr(store, chunks={} if dask else None, consolidated=False)
    ctx = Context()
    ctx.from_dataset("ocean", ds, chunks={"time": 1})

    def run():
        return ctx.sql(
            "SELECT * FROM ocean WHERE time BETWEEN 2 AND 5 "
            "ORDER BY time, lat, lon"
        ).to_arrow_table()

    rows = benchmark(run)
    window = ds.isel(time=slice(2, 6))
    assert rows.num_rows == window.sizes["time"] * 90 * 180
    np.testing.assert_allclose(
        np.asarray(rows["v07"].to_numpy(zero_copy_only=False), "float64"),
        window["v07"].values.ravel(),
        rtol=1e-6,
    )
