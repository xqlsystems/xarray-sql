"""GeoArrow point-geometry columns derived at registration."""

import json

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr

import xarray_sql as xql
from xarray_sql.backends.pyarrow import XarrayPushdownDataset


@pytest.fixture
def grid() -> xr.Dataset:
    return xr.Dataset(
        {"risk": (["y", "x"], np.arange(8.0 * 6).reshape(8, 6))},
        coords={
            "y": np.linspace(-28.0, -29.4, 8),  # descending, like rasters
            "x": np.linspace(-58.0, -57.0, 6),
        },
    )


def test_geometry_field_annotation(grid):
    dataset = xql.arrow_dataset(grid, {"y": 4}, geometry=("x", "y"))
    field = dataset.schema.field("geometry")
    assert field.type == pa.binary()
    assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"
    meta = json.loads(field.metadata[b"ARROW:extension:metadata"])
    assert meta == {"crs": "OGC:CRS84"}


def test_wkb_points_decode_exactly(grid):
    dataset = xql.arrow_dataset(grid, {"y": 4}, geometry=("x", "y"))
    table = dataset.to_table(columns=["geometry", "x", "y"])
    blob = table["geometry"][0].as_py()
    assert len(blob) == 21 and blob[0] == 1
    x = np.frombuffer(blob, "<f8", count=1, offset=5)[0]
    y = np.frombuffer(blob, "<f8", count=1, offset=13)[0]
    assert (x, y) == (table["x"][0].as_py(), table["y"][0].as_py())


def test_native_points_are_the_coordinate_arrays(grid):
    dataset = xql.arrow_dataset(
        grid, {"y": 4}, geometry=("x", "y"), geometry_encoding="point"
    )
    table = dataset.to_table(columns=["x", "y", "geometry"])
    geom = table["geometry"].combine_chunks()
    np.testing.assert_array_equal(
        geom.field("x").to_numpy(), table["x"].to_numpy()
    )
    np.testing.assert_array_equal(
        geom.field("y").to_numpy(), table["y"].to_numpy()
    )


def test_geometry_without_projecting_coords(grid):
    # geometry can be requested alone; the dims it derives from are
    # scanned but projected away.
    dataset = xql.arrow_dataset(grid, {"y": 4}, geometry=("x", "y"))
    table = dataset.to_table(columns=["geometry", "risk"])
    assert table.column_names == ["geometry", "risk"]
    assert table.num_rows == 48


@pytest.fixture
def spatial_con(grid):
    """A spatial-loaded DuckDB connection with ``grid`` registered as ``t``.

    Returns ``(con, reads)`` where ``reads`` records each chunk read.
    """
    duckdb = pytest.importorskip("duckdb")

    reads: list = []
    dataset = XarrayPushdownDataset(
        grid,
        {"y": 4},
        geometry=("x", "y"),
        _iteration_callback=lambda b, n: reads.append(b),
    )
    con = duckdb.connect()
    try:
        con.execute("INSTALL spatial; LOAD spatial;")
    except duckdb.Error:
        pytest.skip("duckdb spatial extension unavailable")
    con.register("t", dataset)
    return con, reads


def test_duckdb_st_within_on_geometry_column(grid, spatial_con):
    con, reads = spatial_con

    described = dict(
        (row[0], row[1]) for row in con.execute("DESCRIBE t").fetchall()
    )
    assert described["geometry"].startswith("GEOMETRY")

    # bbox conjuncts prune chunks; the polygon refines row-exactly on
    # the ingested GEOMETRY column — no ST_Point construction needed.
    reads.clear()
    got = con.execute(
        "SELECT count(*), round(avg(risk), 3) FROM t "
        "WHERE y BETWEEN -28.7 AND -28.0 "
        "AND ST_Within(geometry, ST_GeomFromText("
        "'POLYGON ((-58.1 -28.75, -56.9 -28.75, -56.9 -27.9, "
        "-58.1 -27.9, -58.1 -28.75))'))"
    ).fetchone()
    assert len(reads) == 1  # y-range kept only the first chunk
    inside = (grid.y >= -28.7) & (grid.y <= -28.0)
    expected = grid.risk.values[inside.values, :]
    assert got == (expected.size, round(float(expected.mean()), 3))


_POLYGON = (
    "POLYGON ((-58.1 -28.75, -56.9 -28.75, -56.9 -27.9, "
    "-58.1 -27.9, -58.1 -28.75))"
)


def test_duckdb_spatial_join_needs_a_point_or_a_table(grid, spatial_con):
    """Joining the geometry column to a table of polygons fails inside
    DuckDB's Arrow scan, which can't take the filter the spatial join
    pushes into it. Building the point in the join, or copying the table
    into DuckDB first, works. When DuckDB fixes this, the first assertion
    fails: drop the workarounds from the docs then."""
    duckdb = pytest.importorskip("duckdb")
    con, _ = spatial_con
    con.execute(
        "CREATE TABLE regions AS "
        f"SELECT ST_SetCRS(ST_GeomFromText('{_POLYGON}'), 'OGC:CRS84') AS geom"
    )
    want = int(((grid.y >= -28.75) & (grid.y <= -27.9)).sum()) * grid.sizes["x"]

    with pytest.raises(duckdb.NotImplementedException, match="PyArrow"):
        con.execute(
            "SELECT count(*) FROM t JOIN regions r ON ST_Within(t.geometry, r.geom)"
        ).fetchone()
    got = con.execute(
        "SELECT count(*) FROM t JOIN regions r ON "
        "ST_Within(ST_SetCRS(ST_Point(t.x, t.y), 'OGC:CRS84'), r.geom)"
    ).fetchone()
    assert got == (want,)
    con.execute("CREATE TABLE copied AS FROM t")
    got = con.execute(
        "SELECT count(*) FROM copied c JOIN regions r ON ST_Within(c.geometry, r.geom)"
    ).fetchone()
    assert got == (want,)


def test_duckdb_compares_only_matching_crs_tags(grid):
    """DuckDB's ST_Read tags GeoJSON as EPSG:4326, and ST_Within refuses
    to compare it with the points' OGC:CRS84 tag. Retagging the polygon
    OGC:CRS84, the longitude-first order GeoJSON uses, makes them compare."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    try:
        con.execute("INSTALL spatial; LOAD spatial;")
    except duckdb.Error:
        pytest.skip("duckdb spatial extension unavailable")
    xql.register(con, "t", grid, geometry=("x", "y"))
    polygon = f"ST_SetCRS(ST_GeomFromText('{_POLYGON}'), 'EPSG:4326')"
    query = "SELECT count(*) FROM t WHERE ST_Within(geometry, {})"

    with pytest.raises(duckdb.BinderException, match="coordinate reference"):
        con.execute(query.format(polygon)).fetchone()
    retagged = f"ST_SetCRS({polygon}, 'OGC:CRS84')"
    inside = (grid.y >= -28.75) & (grid.y <= -27.9)
    want = int(inside.sum()) * grid.sizes["x"]
    assert con.execute(query.format(retagged)).fetchone() == (want,)


def test_geopandas_consumes_native_points(grid):
    gpd = pytest.importorskip("geopandas")

    dataset = xql.arrow_dataset(
        grid, {"y": 4}, geometry=("x", "y"), geometry_encoding="point"
    )
    gdf = gpd.GeoDataFrame.from_arrow(dataset.to_table())
    assert gdf.geometry.iloc[0].x == float(grid.x[0])
    assert str(gdf.crs).endswith("CRS84")


def test_geometry_name_collision_raises():
    clash = xr.Dataset(
        {"geometry": (["x"], np.arange(3.0))},
        coords={"x": np.arange(3.0)},
    )
    with pytest.raises(ValueError, match="shadow"):
        xql.arrow_dataset(clash, {"x": 3}, geometry=("x", "x"))


def test_bbox_conjuncts_prunes_and_pairs_with_st_within(grid, spatial_con):
    con, reads = spatial_con

    bounds = (-58.1, -28.75, -56.9, -27.9)  # xmin, ymin, xmax, ymax
    conjuncts = xql.bbox_conjuncts(bounds, x="x", y="y")
    assert '"x" BETWEEN' in conjuncts and '"y" BETWEEN' in conjuncts
    reads.clear()
    got = con.execute(
        f"SELECT count(*) FROM t WHERE {conjuncts} "
        "AND ST_Within(geometry, ST_GeomFromText("
        "'POLYGON ((-58.1 -28.75, -56.9 -28.75, -56.9 -27.9, "
        "-58.1 -27.9, -58.1 -28.75))'))"
    ).fetchone()
    assert len(reads) == 1  # the y-range pruned to one chunk
    inside = (grid.y >= -28.75) & (grid.y <= -27.9)
    assert got[0] == int(inside.sum()) * grid.sizes["x"]


def test_bbox_conjuncts_accepts_bounds_objects():
    class Boxy:
        bounds = (1.0, 2.0, 3.0, 4.0)

    sql = xql.bbox_conjuncts(Boxy(), x="lon", y="lat", pad=0.5)
    assert sql == '"lon" BETWEEN 0.5 AND 3.5 AND "lat" BETWEEN 1.5 AND 4.5'


def test_wkb_points_guards_int32_offset_overflow():
    from xarray_sql.geometry import _wkb_points

    # Stride-0 broadcast views: len() reports ~103M points without
    # allocating them, and the guard must fire before any buffer is
    # built (pa.binary() offsets are int32; n * 21 would overflow).
    n = 103_000_000
    x = np.broadcast_to(np.float64(0.0), (n,))
    with pytest.raises(ValueError, match="int32"):
        _wkb_points(x, x)
