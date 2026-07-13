"""Tests for one-time materialization and pyramid cubes.

Parametrized over both supported engines: the helpers dispatch through
the adapter layer, so DuckDB connections and DataFusion contexts must
behave identically.
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


def test_materialize_caches_query(con):
    xql.materialize(
        con,
        "cube",
        "SELECT FLOOR(y) AS lat, klass, COUNT(*) AS n FROM grid GROUP BY 1, 2",
        order_by=["lat", "klass"],
    )
    assert _rows(con, "SELECT SUM(n) FROM cube")[0][0] == 64 * 64
    # Re-created on repeat (CREATE OR REPLACE), not duplicated.
    xql.materialize(con, "cube", "SELECT 1 AS one")
    assert _rows(con, "SELECT * FROM cube") == [(1,)]


def test_pyramid_levels_roll_up_exactly(con):
    xql.pyramid(
        con,
        "pyr",
        "grid",
        aggs={
            "n": ("count", "*"),
            "class4_n": ("sum", "CASE WHEN klass >= 4 THEN 1 ELSE 0 END"),
            "max_class": ("max", "klass"),
        },
        base_cell=0.5,
        levels=3,
    )
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
    # Coarser levels have fewer cells.
    counts = _rows(
        con, "SELECT level, COUNT(*) FROM pyr GROUP BY level ORDER BY level"
    )
    assert counts[0][1] > counts[1][1] > counts[2][1]


def test_pyramid_share_matches_direct_query(con):
    xql.pyramid(
        con,
        "pyr",
        "grid",
        aggs={
            "n": ("count", "*"),
            "class4_n": ("sum", "CASE WHEN klass >= 4 THEN 1 ELSE 0 END"),
        },
        base_cell=1.0,
        levels=1,
    )
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


def test_pyramid_rejects_non_decomposable_aggs(con):
    with pytest.raises(ValueError, match="unsupported kinds"):
        xql.pyramid(
            con,
            "pyr",
            "grid",
            aggs={"m": ("avg", "klass")},
            base_cell=1.0,
            levels=1,
        )


def test_pyramid_filter_prunes_source_scan(con):
    xql.pyramid(
        con,
        "pyr",
        "grid",
        aggs={"n": ("count", "*")},
        base_cell=1.0,
        levels=1,
        filter="y > -32",
    )
    n = _rows(con, "SELECT SUM(n) FROM pyr")[0][0]
    direct = _rows(con, "SELECT COUNT(*) FROM grid WHERE y > -32")[0][0]
    assert n == direct


def test_pyramid_cell_membership_is_exact_across_levels():
    # Rebinning float origins level-over-level can alias points across
    # cell boundaries; integer indices must halve exactly instead.
    import math

    con = duckdb.connect()
    con.execute("CREATE TABLE pts AS SELECT -130.943 AS x, 0.0005 AS y")
    xql.pyramid(
        con,
        "pyr",
        "pts",
        aggs={"n": ("count", "*")},
        base_cell=0.001,
        levels=2,
    )
    rows = con.sql("SELECT level, x_idx FROM pyr ORDER BY level").fetchall()
    assert rows[0][1] == math.floor(-130.943 / 0.001)
    assert rows[1][1] == math.floor(-130.943 / 0.002)
