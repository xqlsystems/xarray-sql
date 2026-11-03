"""A benchmark tracked over time in CI (see CONTRIBUTING, issue #298).

Starts small: one benchmark, over an in-memory Dataset that needs no
network, so a regression in xarray-sql, DataFusion or the in-memory
table adapter shows up in the history github-action-benchmark stores
on every push to main.

Uses pytest-benchmark's ``benchmark`` fixture. pytest-codspeed
(https://codspeed.io/) provides a fixture of the same name, so a
benchmark written this way can run unmodified under either, if
CodSpeed's instruction-counting is added later.
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
        ).to_dataset(dims=["lat", "lon"])

    result = benchmark(run)
    assert result.sizes["lat"] == grid.sizes["lat"]
    assert result.sizes["lon"] == grid.sizes["lon"]
