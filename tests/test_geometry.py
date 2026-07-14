"""GeoArrow point-geometry columns derived at registration."""

import json

import numpy as np
import pandas as pd
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


def test_duckdb_st_within_on_geometry_column(grid):
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
