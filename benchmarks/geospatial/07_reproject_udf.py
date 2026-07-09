#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "numpy",
#   "pyproj",
#   "pyarrow",
#   "xee",
#   "earthengine-api",
#   "shapely",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Reprojection — a per-pixel CRS transform is a scalar UDF (à la ST_Transform).

Reprojection moves coordinates from one CRS to another (here UTM zone 10N,
EPSG:32610, → lon/lat, EPSG:4326). Crucially it is **row-independent**: each
pixel's new coordinate depends only on its own old coordinate. That is exactly
the shape of a SQL *scalar UDF*, and it is precisely how the geospatial SQL
world already does it — PostGIS ``ST_Transform`` and DuckDB-spatial
``ST_Transform`` are scalar PROJ wrappers.

xarray-sql ships that UDF as its pyproj extension (``xarray_sql.proj``):
with pyproj installed, every ``XarrayContext`` speaks CRS out of the box,
and the CRS pair is part of the query rather than baked into the UDF::

    SELECT x, y,
           reproject(x, y, 'EPSG:32610', 'EPSG:4326')['x'] AS lon,
           reproject(x, y, 'EPSG:32610', 'EPSG:4326')['y'] AS lat
    FROM grid

**The reference is Earth Engine itself.** There is *one* dataset: a single UTM
grid opened through [Xee](https://github.com/google/Xee) carrying
``ee.Image.pixelLonLat()``. Each pixel arrives with two things — its UTM ``x``/
``y`` (the grid coordinates, our SQL input) and Earth Engine's *own* per-pixel
``longitude``/``latitude`` (data variables, the reference). So we are not
opening the same image twice in two CRS; we feed the UTM coordinates to the PROJ
UDF and check the lon/lat it returns against EE's independently-computed lon/lat
for the *same* pixels. The reference is a different geodesy engine, not PROJ
again, and they agree to sub-metre precision.

The extension returns *both* coordinates from one struct-returning call
(one PROJ transform per row) and runs all PROJ work on its own worker
pool, so the query parallelizes across partitions safely.

Requires Earth Engine access: ``earthengine authenticate`` once, then an
initialized project (set ``EARTHENGINE_PROJECT``). Skips cleanly otherwise.
"""

from __future__ import annotations

import xarray as xr

import xarray_sql as xql

from _harness import (
    CaseSkipped,
    assert_grid_close,
    initialize_earth_engine,
    measured,
    run_case,
    show_result,
    show_sql,
)

_SRC_CRS, _DST_CRS = "EPSG:32610", "EPSG:4326"  # UTM zone 10N → lon/lat
# A 1° box over the San Francisco Bay area, well inside UTM zone 10N.
_AOI = (-122.6, 37.4, -121.6, 38.4)
_SCALE_M = 2_000  # 2 km pixels → a ~50×60 grid


def _open_ee_lonlat_grid() -> xr.Dataset:
    """Open ``ee.Image.pixelLonLat()`` on a UTM grid via Xee.

    Earth Engine evaluates ``pixelLonLat`` on the requested UTM grid, so each
    pixel carries its UTM ``x``/``y`` (coordinates) and EE's own ``longitude`` /
    ``latitude`` (data variables) — the independent reprojection reference.
    """
    try:
        import shapely.geometry as sgeom
        from xee import helpers
    except ImportError as exc:  # pragma: no cover
        raise CaseSkipped(
            "Earth Engine support needs `pip install earthengine-api xee`"
        ) from exc

    ee = initialize_earth_engine()

    # fit_geometry builds the pixel grid (crs, crs_transform, shape_2d) Xee's
    # backend expects — here a UTM grid at _SCALE_M metres covering the AOI.
    grid = helpers.fit_geometry(
        sgeom.box(*_AOI),
        geometry_crs="EPSG:4326",
        grid_crs=_SRC_CRS,
        grid_scale=(float(_SCALE_M), float(_SCALE_M)),
    )
    ic = ee.ImageCollection([ee.Image.pixelLonLat()])
    ds = xr.open_dataset(ic, engine="ee", **grid)
    # One image → a length-1 time axis; drop it. Xee gives x/y coordinates (UTM
    # metres) and longitude/latitude data variables (EE's per-pixel geodesy).
    return ds.isel(time=0).load()


def main() -> None:
    ds = _open_ee_lonlat_grid()
    n = ds.sizes["y"] * ds.sizes["x"]
    print(
        f"  EE pixelLonLat on UTM grid {dict(ds.sizes)}  ({n:,} pixels)  "
        f"{_SRC_CRS} → {_DST_CRS}"
    )

    # XarrayContext registers reproject() automatically (the pyproj
    # extension). Several partitions: the extension runs PROJ on its own
    # worker pool, so the UDF is safe under DataFusion's parallelism.
    ctx = xql.XarrayContext()
    ctx.from_dataset(
        "grid", ds, chunks={"y": max(1, ds.sizes["y"] // 4), "x": ds.sizes["x"]}
    )

    sql = f"""
        SELECT x, y,
               reproject(x, y, '{_SRC_CRS}', '{_DST_CRS}')['x'] AS lon,
               reproject(x, y, '{_SRC_CRS}', '{_DST_CRS}')['y'] AS lat
        FROM grid
        ORDER BY y, x
    """
    show_sql(sql)

    for _ in measured("SQL reprojection (PROJ scalar UDF)"):
        got = ctx.sql(sql).to_dataset(dims=["y", "x"])

    # Reference: Earth Engine's own per-pixel lon/lat (independent of PROJ).
    # EE and PROJ are separate implementations, so compare at ~1e-5° (~1 m).
    assert_grid_close(
        "reprojected longitude", got.lon, ds.longitude, rtol=0, atol=1e-5
    )
    assert_grid_close(
        "reprojected latitude", got.lat, ds.latitude, rtol=0, atol=1e-5
    )

    show_result(got)

    corner = got.isel(x=0, y=0)
    print(
        f"\n  Corner check: UTM ({float(corner.x):.0f}, {float(corner.y):.0f}) → "
        f"lon {float(corner.lon):.4f}, lat {float(corner.lat):.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Reprojection: PROJ scalar UDF vs Earth Engine")
    )
