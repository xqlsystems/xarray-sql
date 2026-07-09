#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "numpy",
#   "pyproj",
#   "pyarrow",
#   "scipy",
#   "xee",
#   "earthengine-api",
#   "shapely",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Warp — reprojecting *and* resampling a raster is case 07's UDF + case 08's JOIN.

A *warp* moves a raster from one CRS onto a grid in another — the everyday GDAL/
rasterio ``reproject`` that GIS runs constantly. It is exactly the composition of
the two "hard" cases:

* **case 07** — reproject coordinates with a scalar PROJ **UDF**; and
* **case 08** — resample values with a sparse-weight **JOIN**.

The pipeline reads as that composition, and it shows the division of labor cleanly:

1. **SQL reprojects the target grid** (the 07 UDF): for every target ``(lon, lat)``
   cell, ``reproject()`` returns where it falls in the source CRS (UTM ``x``/``y``).
2. **Arrays build the bilinear weights** (the geometry): each reprojected target
   point lands between four source pixels; we compute its four bilinear weights —
   the genuinely geometric step the array world owns. (This is the same
   "arrays compute the weights, SQL applies them" boundary as case 08, except the
   target points are *scattered* in source space because they were reprojected,
   so the weights are a per-point stencil rather than a separable lat×lon grid.)
3. **SQL applies the weights** (the 08 JOIN): join the source values onto the
   weight table and ``SUM(value * weight)`` per target cell.

**References.** The exact check is the array paradigm doing the same warp — plain
``pyproj`` + xarray ``.interp`` at the reprojected points — which the SQL result
matches to floating-point tolerance. As an *independent* real-world cross-check we
also open the **same SRTM** directly on the lon/lat grid through Xee (Earth
Engine's own warp) and report the agreement; it is close (a few metres median) but
not bit-exact, because EE resamples from native 30 m while our source is the coarse
UTM grid — which is exactly why the deterministic warp, not EE, is the tolerance
reference.

Data: real **SRTM elevation** (Northern California terrain) via [Xee](https://github.com/google/Xee),
opened once on a UTM grid (the source) and once on a lon/lat grid (the EE
cross-check). Requires Earth Engine access; skips cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pyproj
import shapely.geometry as sgeom
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
    timed,
)

_SRC_CRS = "EPSG:32610"  # UTM zone 10N — the source raster's CRS
_DST_CRS = "EPSG:4326"  # lon/lat — the target grid's CRS
_AOI = (-122.6, 37.4, -121.6, 38.4)  # ~1° box of Northern California terrain
_SRC_SCALE_M = 2_000.0  # ~2 km source pixels
_DST_SCALE_DEG = 0.02  # ~2 km target cells


def _open_srtm(
    grid_crs: str, scale: tuple[float, float], xy_names
) -> xr.DataArray:
    """Open SRTM elevation over the AOI on the requested grid via Xee (lazy)."""
    try:
        from xee import helpers
    except ImportError as exc:  # pragma: no cover
        raise CaseSkipped(
            "Earth Engine support needs `pip install earthengine-api xee`"
        ) from exc

    ee = initialize_earth_engine()
    grid = helpers.fit_geometry(
        sgeom.box(*_AOI),
        geometry_crs="EPSG:4326",
        grid_crs=grid_crs,
        grid_scale=scale,
    )
    ic = ee.ImageCollection([ee.Image("USGS/SRTMGL1_003")])
    da = xr.open_dataset(ic, engine="ee", **grid)["elevation"].isel(time=0)
    a, b = xy_names
    rename = {}
    for d in da.dims:
        dl = d.lower()
        if dl in ("y", "lat", "latitude"):
            rename[d] = a
        elif dl in ("x", "lon", "longitude"):
            rename[d] = b
    da = da.rename(rename).sortby(a).sortby(b)
    return da.assign_coords(
        {a: da[a].astype("float64"), b: da[b].astype("float64")}
    )


def _warp_weight_table(
    sx: np.ndarray,
    sy: np.ndarray,
    dst_lon: np.ndarray,
    dst_lat: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
) -> xr.Dataset:
    """Bilinear weights for reprojected target points — the geometry step.

    Each target cell ``(dst_lat, dst_lon)`` was reprojected to source coordinates
    ``(px, py)``; here we find the four surrounding source pixels and their
    bilinear weights. One row per (target cell, source corner). Targets that fall
    outside the source footprint contribute no rows (and are dropped).
    """
    dst_lats, dst_lons, src_xs, src_ys, weights = [], [], [], [], []
    for k in range(len(px)):
        x, y = px[k], py[k]
        if not (sx[0] <= x <= sx[-1] and sy[0] <= y <= sy[-1]):
            continue
        i = int(np.clip(np.searchsorted(sx, x) - 1, 0, len(sx) - 2))
        j = int(np.clip(np.searchsorted(sy, y) - 1, 0, len(sy) - 2))
        tx = (x - sx[i]) / (sx[i + 1] - sx[i])
        ty = (y - sy[j]) / (sy[j + 1] - sy[j])
        for ii, wx in ((i, 1.0 - tx), (i + 1, tx)):
            for jj, wy in ((j, 1.0 - ty), (j + 1, ty)):
                dst_lats.append(dst_lat[k])
                dst_lons.append(dst_lon[k])
                src_xs.append(sx[ii])
                src_ys.append(sy[jj])
                weights.append(wx * wy)
    n = len(weights)
    return xr.Dataset(
        {
            "dst_lat": (["pair"], np.array(dst_lats, "float64")),
            "dst_lon": (["pair"], np.array(dst_lons, "float64")),
            "src_x": (["pair"], np.array(src_xs, "float64")),
            "src_y": (["pair"], np.array(src_ys, "float64")),
            "weight": (["pair"], np.array(weights, "float64")),
        },
        coords={"pair": np.arange(n)},
    ).chunk({"pair": n})


def main() -> None:
    with timed("open SRTM on UTM + lon/lat grids via Xee"):
        src = _open_srtm(_SRC_CRS, (_SRC_SCALE_M, _SRC_SCALE_M), ("y", "x"))
        ref_ee = _open_srtm(
            _DST_CRS, (_DST_SCALE_DEG, _DST_SCALE_DEG), ("lat", "lon")
        )
    sx, sy = src.x.values, src.y.values

    # Target lon/lat grid strictly inside the source UTM footprint, so every
    # target cell reprojects to a point with four source corners (no edge cells
    # to drop). Inscribe a lon/lat box in the reprojected UTM rectangle.
    inv = pyproj.Transformer.from_crs(_SRC_CRS, _DST_CRS, always_xy=True)
    cx = [sx[0], sx[-1], sx[0], sx[-1]]
    cy = [sy[0], sy[0], sy[-1], sy[-1]]
    clon, clat = inv.transform(cx, cy)
    lon0, lon1 = max(clon[0], clon[2]) + 0.01, min(clon[1], clon[3]) - 0.01
    lat0, lat1 = max(clat[0], clat[1]) + 0.01, min(clat[2], clat[3]) - 0.01
    tlon = np.linspace(lon0, lon1, 60)
    tlat = np.linspace(lat0, lat1, 60)
    print(
        f"  source UTM grid {len(sy)}×{len(sx)}  →  target lon/lat grid "
        f"{len(tlat)}×{len(tlon)}  ({_SRC_CRS} → {_DST_CRS})"
    )

    # XarrayContext registers reproject() automatically (the pyproj
    # extension) — the direction is spelled in the query itself.
    ctx = xql.XarrayContext()

    # The target grid as a (dst_lat, dst_lon) table.
    LON, LAT = np.meshgrid(tlon, tlat)
    target = xr.Dataset(
        {
            "dst_lon": (["cell"], LON.ravel()),
            "dst_lat": (["cell"], LAT.ravel()),
        },
        coords={"cell": np.arange(LON.size)},
    ).chunk({"cell": LON.size})
    ctx.from_dataset("target", target, chunks={"cell": LON.size})

    # 1) SQL reprojects the target grid into the source CRS (case 07's UDF).
    reproj_sql = f"""
        SELECT dst_lat, dst_lon,
               reproject(dst_lon, dst_lat, '{_DST_CRS}', '{_SRC_CRS}')['x'] AS sx,
               reproject(dst_lon, dst_lat, '{_DST_CRS}', '{_SRC_CRS}')['y'] AS sy
        FROM target
    """
    show_sql(reproj_sql, label="SQL — reproject target grid (PROJ UDF)")
    rp = ctx.sql(reproj_sql).to_pandas()
    px, py = rp["sx"].to_numpy(), rp["sy"].to_numpy()

    # 2) Arrays turn the reprojected points into a bilinear weight table.
    weights = _warp_weight_table(
        sx, sy, rp["dst_lon"].to_numpy(), rp["dst_lat"].to_numpy(), px, py
    )
    ctx.from_dataset(
        "src", src.to_dataset(name="value"), chunks={"y": len(sy), "x": len(sx)}
    )
    ctx.from_dataset("weights", weights, chunks={"pair": weights.sizes["pair"]})

    # 3) SQL applies the weights (case 08's JOIN).
    apply_sql = """
        SELECT w.dst_lat AS lat, w.dst_lon AS lon,
               SUM(s.value * w.weight) AS warped
        FROM weights w
        JOIN src s ON s.x = w.src_x AND s.y = w.src_y
        GROUP BY w.dst_lat, w.dst_lon
        ORDER BY w.dst_lat, w.dst_lon
    """
    show_sql(apply_sql, label="SQL — apply bilinear weights (JOIN)")
    for _ in measured("SQL warp (reproject UDF + regrid JOIN)"):
        got = ctx.sql(apply_sql).to_dataset(dims=["lat", "lon"]).warped

    # Reference: the array paradigm doing the same warp — pyproj reproject of the
    # target grid, then xarray's own bilinear .interp at those source points.
    for _ in measured("xarray reference (pyproj + .interp)"):
        tr = pyproj.Transformer.from_crs(_DST_CRS, _SRC_CRS, always_xy=True)
        rx, ry = tr.transform(LON.ravel(), LAT.ravel())
        warped = src.interp(
            x=xr.DataArray(rx, dims="cell"),
            y=xr.DataArray(ry, dims="cell"),
            method="linear",
        ).values.reshape(len(tlat), len(tlon))
        ref = xr.DataArray(
            warped, dims=["lat", "lon"], coords={"lat": tlat, "lon": tlon}
        )

    assert_grid_close("warped elevation (m)", got, ref, rtol=1e-6, atol=1e-4)
    show_result(got)

    # Independent cross-check: EE's own SRTM on the lon/lat grid (a real warp).
    ee_on_grid = ref_ee.interp(lat=got.lat, lon=got.lon, method="linear").values
    a, b = got.values.ravel(), ee_on_grid.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    corr = float(np.corrcoef(a[m], b[m])[0, 1])
    print(
        f"\n  vs Earth Engine's own lon/lat SRTM: median |Δ| "
        f"{np.nanmedian(np.abs(a[m] - b[m])):.1f} m, correlation {corr:.4f} "
        f"(EE resamples native 30 m; ours warps the {_SRC_SCALE_M:.0f} m UTM grid)"
    )


if __name__ == "__main__":
    raise SystemExit(run_case(main, "Warp: reproject UDF + regrid JOIN (SRTM)"))
