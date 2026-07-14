"""The performance guide's SQL recipes, pinned on both engines.

The guide documents caching and pyramid building as plain engine SQL
rather than wrapping them in helpers; these tests run the documented
statements on DuckDB and DataFusion so the recipes cannot rot.
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


def _run(con, sql):
    result = con.sql(sql)
    if hasattr(result, "collect"):
        result.collect()  # DataFusion DDL/DML is a lazy plan


def _build_pyramid(con, base_cell=0.5, levels=3):
    # The documented pyramid recipe: level 0 bins the source once into
    # integer cell indices; each coarser level halves the indices and
    # rolls up from the level below. Indices stay integers because
    # rebinning float origins level-over-level can alias boundary
    # points into the wrong parent cell.
    _run(
        con,
        f"""
        CREATE OR REPLACE TABLE pyr AS
        SELECT 0 AS level, x_idx, y_idx,
               x_idx * {base_cell} AS x_bin, y_idx * {base_cell} AS y_bin,
               count(*) AS n,
               sum(CASE WHEN klass >= 4 THEN 1 ELSE 0 END) AS class4_n,
               max(klass) AS max_class
        FROM (
            SELECT CAST(FLOOR(x / {base_cell}) AS BIGINT) AS x_idx,
                   CAST(FLOOR(y / {base_cell}) AS BIGINT) AS y_idx, *
            FROM grid
        )
        GROUP BY x_idx, y_idx
    """,
    )
    for level in range(1, levels):
        cell = base_cell * 2**level
        _run(
            con,
            f"""
            INSERT INTO pyr
            SELECT {level} AS level,
                   CAST(FLOOR(x_idx / 2.0) AS BIGINT) AS x_idx,
                   CAST(FLOOR(y_idx / 2.0) AS BIGINT) AS y_idx,
                   CAST(FLOOR(x_idx / 2.0) AS BIGINT) * {cell} AS x_bin,
                   CAST(FLOOR(y_idx / 2.0) AS BIGINT) * {cell} AS y_bin,
                   sum(n) AS n, sum(class4_n) AS class4_n,
                   max(max_class) AS max_class
            FROM pyr
            WHERE level = {level - 1}
            GROUP BY CAST(FLOOR(x_idx / 2.0) AS BIGINT),
                     CAST(FLOOR(y_idx / 2.0) AS BIGINT)
        """,
        )


def test_documented_pyramid_recipe(con):
    _build_pyramid(con)
    totals = _rows(
        con,
        "SELECT level, SUM(n), SUM(class4_n), MAX(max_class) FROM pyr "
        "GROUP BY level ORDER BY level",
    )
    assert len(totals) == 3
    # Every level preserves the decomposable totals exactly.
    for _, n, class4_n, max_class in totals:
        assert n == 64 * 64
        assert class4_n == totals[0][2]
        assert max_class == totals[0][3]
    # Coarser levels have fewer cells, and shares match a direct query.
    counts = _rows(
        con, "SELECT level, COUNT(*) FROM pyr GROUP BY level ORDER BY level"
    )
    assert counts[0][1] > counts[1][1] > counts[2][1]
    via_pyramid = _rows(
        con,
        "SELECT SUM(class4_n) * 1.0 / SUM(n) FROM pyr "
        "WHERE level = 0 AND x_bin >= -65 AND x_bin < -63",
    )[0][0]
    direct = _rows(
        con,
        "SELECT AVG(CASE WHEN klass >= 4 THEN 1.0 ELSE 0 END) FROM grid "
        "WHERE x >= -65 AND x < -63",
    )[0][0]
    assert via_pyramid == pytest.approx(direct)
