"""The performance guide's SQL recipes, pinned on both engines.

The guide documents caching as plain engine SQL rather than wrapping
it in a helper; this test runs the documented statement on DuckDB and
DataFusion so the recipe cannot rot.
"""

import duckdb
import numpy as np
import pytest
import xarray as xr

import xarray_sql as xql


def _grid() -> xr.Dataset:
    np.random.seed(11)
    return xr.Dataset(
        {
            "klass": (
                ["y", "x"],
                np.random.randint(1, 6, (64, 64), dtype=np.uint8),
            )
        },
        coords={
            "y": np.linspace(-34.0, -30.0, 64),
            "x": np.linspace(-66.0, -62.0, 64),
        },
    ).chunk({"y": 32})


@pytest.fixture(params=["duckdb", "datafusion"])
def con(request):
    connection = (
        duckdb.connect() if request.param == "duckdb" else xql.XarrayContext()
    )
    xql.register(connection, "grid", _grid())
    return connection


def _rows(con, sql) -> list[tuple]:
    result = con.sql(sql)
    if hasattr(result, "fetchall"):
        return result.fetchall()
    frame = result.to_pandas()
    return [tuple(r) for r in frame.itertuples(index=False)]


def test_documented_caching_recipe(con):
    # The performance guide documents caching as plain engine SQL; this
    # pins the recipe on both engines — including that DataFusion DDL
    # is a lazy plan that must be collected to execute.
    ctas = (
        "CREATE OR REPLACE TABLE cube AS "
        "SELECT FLOOR(y) AS lat, klass, COUNT(*) AS n FROM grid "
        "GROUP BY 1, 2 ORDER BY lat, klass"
    )
    result = con.sql(ctas)
    if hasattr(result, "collect"):
        result.collect()
    assert _rows(con, "SELECT SUM(n) FROM cube")[0][0] == 64 * 64
