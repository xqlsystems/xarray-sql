"""Naming the tables a mixed-dimension Dataset splits into, on every engine.

A Dataset whose variables sit on different dimensions (ARCO-ERA5: 262
surface fields on ``(time, latitude, longitude)``, 11 atmospheric ones
on ``(time, level, latitude, longitude)``) becomes one table per
dimension group. ``table_names`` is what lets the user call those
tables ``surface`` and ``atmosphere`` instead of
``time_latitude_longitude``.

The contract these tests hold:

1. ``table_names`` means the same thing wherever a Dataset is
   registered — ``XarrayContext``, a plain ``SessionContext``, DuckDB,
   or the pyarrow-dataset path Polars scans.
2. ``name.group`` is the portable spelling: the same SQL text runs on
   DataFusion and DuckDB. DuckDB additionally keeps the flat
   ``name_group`` tables it has always registered.
3. Naming changes only names. Pushdown, projection, and results are
   what they were.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datafusion import SessionContext

import xarray_sql as xql

duckdb = pytest.importorskip("duckdb")

# The ARCO-ERA5 shape in miniature: a surface group, an atmospheric
# group on an extra `level` dim, and a scalar metadata variable.
NAMES = {
    ("time", "lat", "lon"): "surface",
    ("time", "level", "lat", "lon"): "atmosphere",
    (): "meta",
}

SURFACE_QUERY = "SELECT AVG(t2m) AS avg_t FROM era5.surface"


@pytest.fixture
def ds() -> xr.Dataset:
    np.random.seed(11)
    return xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], np.random.rand(6, 3, 4)),
            "temperature": (
                ["time", "level", "lat", "lon"],
                np.random.rand(6, 2, 3, 4),
            ),
            "projection": ((), 42.0),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=6, freq="D"),
            "lat": np.linspace(-90, 90, 3),
            "lon": np.linspace(-180, 180, 4),
            "level": [500, 1000],
        },
    ).chunk({"time": 1})


def expected_avg(ds: xr.Dataset) -> float:
    return float(ds["t2m"].mean().compute())


# 1. The same naming map on every engine ------------------------------


def test_xarray_context_names_tables(ds):
    ctx = xql.XarrayContext()
    xql.register(ctx, "era5", ds, table_names=NAMES)

    result = ctx.sql(SURFACE_QUERY).to_pandas()["avg_t"][0]
    assert result == pytest.approx(expected_avg(ds))


def test_plain_session_context_splits_and_names(ds):
    # A bare SessionContext used to register the whole Dataset as one
    # table, which a mixed-dimension Dataset cannot be.
    con = SessionContext()
    xql.register(con, "era5", ds, table_names=NAMES)

    result = con.sql(SURFACE_QUERY).to_pandas()["avg_t"][0]
    assert result == pytest.approx(expected_avg(ds))


def test_plain_session_context_no_longer_flattens_mixed_dims(ds):
    # Pinning the break: the old single-table registration broadcast
    # `t2m` (3D) against `temperature` (4D) into one nonsensical table
    # (a real query against it returned a plausible-looking but wrong
    # row count). `era5` is now a schema, not a table — querying it
    # directly as one is the visible signal that the broadcast is gone,
    # not silently wrong.
    con = SessionContext()
    xql.register(con, "era5", ds)

    with pytest.raises(Exception):
        con.sql("SELECT COUNT(*) FROM era5").to_pandas()


def test_duckdb_names_tables(ds):
    con = duckdb.connect()
    xql.register(con, "era5", ds, table_names=NAMES)

    result = con.sql(SURFACE_QUERY).fetchone()[0]
    assert result == pytest.approx(expected_avg(ds))


def test_arrow_datasets_names_each_group(ds):
    tables = xql.arrow_datasets(ds, "era5", table_names=NAMES)

    assert set(tables) == {"era5_surface", "era5_atmosphere", "era5_meta"}
    assert set(tables["era5_surface"].schema.names) == {
        "time",
        "lat",
        "lon",
        "t2m",
    }


def test_arrow_datasets_accepts_the_same_kwargs_either_shape(ds):
    # The single-group branch used to go through arrow_dataset(), whose
    # signature is a fixed subset of what XarrayPushdownDataset accepts;
    # a kwarg the multi-group branch forwards fine (constructed
    # directly) would raise only when the Dataset happened to be
    # uniform.
    calls: list = []

    def record(block, columns) -> None:
        calls.append((block, columns))

    uniform = ds[["t2m"]]
    xql.arrow_datasets(uniform, "x", _iteration_callback=record)["x"].to_table()
    assert calls  # single-group branch actually took the callback

    calls.clear()
    tables = xql.arrow_datasets(
        ds, "x", table_names=NAMES, _iteration_callback=record
    )
    tables["x_surface"].to_table()
    assert calls  # multi-group branch already did; pinning it stays so


def test_arrow_datasets_geometry_skips_groups_without_the_dims(ds):
    # geometry=(x, y) is forwarded to every returned table; a
    # mixed-dimension Dataset has groups (here, the scalar `meta`
    # group) that don't have those dims at all.
    from xarray_sql.geometry import GEOMETRY_COLUMN

    tables = xql.arrow_datasets(
        ds, "era5", table_names=NAMES, geometry=("lat", "lon")
    )

    assert GEOMETRY_COLUMN in tables["era5_surface"].schema.names
    assert GEOMETRY_COLUMN in tables["era5_atmosphere"].schema.names
    assert GEOMETRY_COLUMN not in tables["era5_meta"].schema.names


def test_polars_queries_named_tables(ds):
    pl = pytest.importorskip("polars")
    ctx = pl.SQLContext()
    tables = xql.arrow_datasets(ds, "era5", table_names=NAMES)
    for table, dataset in tables.items():
        ctx.register(table, pl.scan_pyarrow_dataset(dataset))

    result = ctx.execute(
        "SELECT AVG(t2m) AS avg_t FROM era5_surface", eager=True
    )
    assert result["avg_t"][0] == pytest.approx(expected_avg(ds))


def test_every_engine_answers_the_same_query(ds):
    """One SQL string, three engines, one number."""
    xarray_ctx = xql.XarrayContext()
    xql.register(xarray_ctx, "era5", ds, table_names=NAMES)
    session_ctx = SessionContext()
    xql.register(session_ctx, "era5", ds, table_names=NAMES)
    con = duckdb.connect()
    xql.register(con, "era5", ds, table_names=NAMES)

    answers = [
        xarray_ctx.sql(SURFACE_QUERY).to_pandas()["avg_t"][0],
        session_ctx.sql(SURFACE_QUERY).to_pandas()["avg_t"][0],
        con.sql(SURFACE_QUERY).fetchone()[0],
    ]
    assert answers == pytest.approx([expected_avg(ds)] * 3)


# 2. Spellings: dotted everywhere, flat still on DuckDB ----------------


def test_duckdb_keeps_the_flat_spelling(ds):
    con = duckdb.connect()
    xql.register(con, "era5", ds, table_names=NAMES)

    dotted = con.sql("SELECT COUNT(*) FROM era5.atmosphere").fetchone()[0]
    flat = con.sql("SELECT COUNT(*) FROM era5_atmosphere").fetchone()[0]
    assert dotted == flat == 6 * 2 * 3 * 4


def test_duckdb_defaults_to_joined_dim_names(ds):
    con = duckdb.connect()
    xql.register(con, "era5", ds)

    assert con.sql("SELECT COUNT(*) FROM era5.time_lat_lon").fetchone()[0] == (
        6 * 3 * 4
    )
    assert con.sql("SELECT COUNT(*) FROM era5_time_lat_lon").fetchone()[0] == (
        6 * 3 * 4
    )


def test_scalar_group_can_be_named(ds):
    con = duckdb.connect()
    xql.register(con, "era5", ds, table_names=NAMES)

    assert con.sql("SELECT projection FROM era5.meta").fetchall() == [(42.0,)]


def test_uniform_dataset_keeps_the_bare_name(ds):
    # One dimension group is one table, named `name` — there is no
    # group to name, on any engine.
    surface_only = ds[["t2m"]]
    con = duckdb.connect()
    xql.register(con, "era5", surface_only, table_names=NAMES)

    assert con.sql("SELECT COUNT(*) FROM era5").fetchone()[0] == 6 * 3 * 4
    assert set(xql.arrow_datasets(surface_only, "era5")) == {"era5"}


def test_arrow_datasets_without_a_prefix(ds):
    # Engines registered table-by-table (Polars) may not want the
    # namespace prefix at all.
    tables = xql.arrow_datasets(ds, table_names=NAMES)

    assert set(tables) == {"surface", "atmosphere", "meta"}


# 3. Naming changes names, nothing else -------------------------------


def test_pushdown_survives_the_duckdb_schema_view(ds):
    # The dotted spelling is a view over the registered table; a view
    # that blocked pushdown would turn a pruned scan into a full one.
    reads: list = []
    projections: list = []
    con = duckdb.connect()
    xql.register(
        con,
        "era5",
        ds,
        table_names=NAMES,
        _iteration_callback=lambda block, proj: (
            reads.append(block),
            projections.append(proj),
        ),
    )

    con.sql(
        "SELECT AVG(t2m) FROM era5.surface WHERE time = TIMESTAMP '2020-01-03'"
    ).fetchall()

    assert len(reads) == 1  # 1 of 6 daily chunks
    assert projections[0] == ["time", "t2m"]


def test_named_result_round_trips_to_xarray(ds):
    con = duckdb.connect()
    xql.register(con, "era5", ds, table_names=NAMES)

    rel = con.sql(
        "SELECT time, lat, lon, t2m FROM era5.surface ORDER BY time, lat, lon"
    )
    out = xql.to_dataset(rel, template=ds)

    xr.testing.assert_allclose(out, ds[["t2m"]].compute())


def test_duplicate_names_are_rejected(ds):
    duplicated = {**NAMES, ("time", "level", "lat", "lon"): "surface"}
    con = duckdb.connect()

    with pytest.raises(ValueError, match="same table name 'surface'"):
        xql.register(con, "era5", ds, table_names=duplicated)


def test_names_for_absent_groups_are_ignored(ds):
    # One canonical naming map, reused over Datasets holding different
    # subsets of the same variables.
    con = duckdb.connect()
    xql.register(con, "era5", ds[["t2m", "projection"]], table_names=NAMES)

    assert con.sql("SELECT COUNT(*) FROM era5.surface").fetchone()[0] == (
        6 * 3 * 4
    )
    views = con.sql("SELECT view_name FROM duckdb_views()").fetchall()
    assert "atmosphere" not in {row[0] for row in views}


def test_read_only_duckdb_warns_but_still_registers(ds, tmp_path):
    # Mirroring the groups as `era5.<group>` needs a writable catalog.
    # A read-only connection loses the dotted spelling, not the tables.
    path = tmp_path / "ro.db"
    duckdb.connect(str(path)).close()
    con = duckdb.connect(str(path), read_only=True)

    with pytest.warns(RuntimeWarning, match="file-backed connection"):
        xql.register(con, "era5", ds, table_names=NAMES)

    assert con.sql("SELECT COUNT(*) FROM era5_surface").fetchone()[0] == (
        6 * 3 * 4
    )


def test_file_backed_duckdb_warns_and_skips_the_dangling_mirror(ds, tmp_path):
    # A writable file-backed connection *can* create the `era5.<group>`
    # views (unlike the read-only case above) — but the views are
    # catalog DDL that persists, while `con.register` only binds the
    # flat tables they select from for this connection's lifetime.
    # Left unchecked, that would leave a view dangling (and erroring
    # with "Table ... does not exist") the moment the database is
    # reopened; the adapter now skips creating it up front instead.
    path = tmp_path / "rw.db"
    con = duckdb.connect(str(path))

    with pytest.warns(RuntimeWarning, match="file-backed connection"):
        xql.register(con, "era5", ds, table_names=NAMES)

    assert con.sql("SELECT COUNT(*) FROM era5_surface").fetchone()[0] == (
        6 * 3 * 4
    )
    views = con.sql("SELECT view_name FROM duckdb_views()").fetchall()
    assert not {row[0] for row in views} & {"surface", "atmosphere", "meta"}

    # Nothing was left in the on-disk catalog for a later connection to
    # trip over — reopening the (still empty) database succeeds cleanly.
    con.close()
    duckdb.connect(str(path)).close()


def test_duckdb_rejects_case_insensitive_name_collision(ds):
    # DuckDB folds identifier case even when quoted, so 'surface' and
    # 'SURFACE' would silently collide there even though
    # resolve_table_names sees two distinct strings.
    collides = {**NAMES, ("time", "level", "lat", "lon"): "SURFACE"}
    con = duckdb.connect()

    with pytest.raises(ValueError, match="case-insensitive"):
        xql.register(con, "era5", ds, table_names=collides)


def test_duckdb_refuses_to_mirror_over_a_real_table(ds):
    # `CREATE OR REPLACE VIEW` would otherwise silently take over a
    # table the caller created for their own purposes.
    con = duckdb.connect()
    con.execute("CREATE SCHEMA era5")
    con.execute("CREATE TABLE era5.surface AS SELECT 1 AS mine")

    with pytest.warns(RuntimeWarning, match="real tables"):
        xql.register(con, "era5", ds, table_names=NAMES)

    # The caller's table survives untouched; the flat tables still work.
    assert con.sql("SELECT * FROM era5.surface").fetchall() == [(1,)]
    assert con.sql("SELECT COUNT(*) FROM era5_surface").fetchone()[0] == (
        6 * 3 * 4
    )
