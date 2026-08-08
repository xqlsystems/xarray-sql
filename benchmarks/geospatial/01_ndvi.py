#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "aiohttp",
#   "requests",
#   "pystac-client",
#   "zarr>=3",
#   "numpy",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""NDVI — "apply_ufunc over a raster" is just column arithmetic.

The Normalized Difference Vegetation Index is the workhorse of optical remote
sensing: ``NDVI = (NIR - Red) / (NIR + Red)``, computed per pixel. The array
paradigm reaches for ``xarray.apply_ufunc`` (the coiled/benchmarks #1545
"vectorized operations" case) to broadcast this over a whole scene.

But a per-pixel formula over two bands is just *column arithmetic over two
columns*::

    SELECT x, y, (nir - red) / (nir + red) AS ndvi
    FROM scene
    ORDER BY y, x

Each pixel is one row; the ufunc is the SELECT expression. Invalid pixels are
already NaN (xarray decodes the band's ``_FillValue`` on open), and NaN
propagates through the arithmetic on both sides — so the masking is free, no
``CASE`` required.

Dataset: a real Sentinel-2 L2A scene in **Zarr** from the ESA EOPF sample
service, discovered with ``pystac-client`` and opened the canonical way with
``xarray`` — ``xr.open_datatree`` yields the reflectance bands (B04=red,
B08=NIR at 10 m) already scaled to reflectance and carrying their ``x``/``y``
coordinates. We read one window so the case stays bounded. Requires network;
skips cleanly if the service is offline.
"""

from __future__ import annotations

import xarray as xr

from _engines import EngineContext
from _harness import (
    CaseSkipped,
    assert_grid_close,
    measured,
    run_case,
    show_result,
    show_sql,
)

# EOPF sample-service STAC catalog; an agricultural AOI near Torino, Italy, in
# early May (peak spring growth). The search is deterministic — it resolves to
# a specific archived Sentinel-2 product.
_STAC = "https://stac.core.eopf.eodc.eu"
_BBOX = [7.2, 44.5, 7.4, 44.7]
_DATETIME = "2025-04-25/2025-05-05"

# A 1024×1024 (~105 km²) window over vegetated valley floor.
_Y0, _X0, _N = 4_000, 6_000, 1_024


def _load_scene() -> tuple[xr.Dataset, str]:
    """Discover a Sentinel-2 L2A product and open its 10 m red/NIR bands.

    Idiomatic end to end: ``pystac-client`` finds the product, ``open_datatree``
    opens the hierarchical EOPF Zarr, and the ``reflectance/r10m`` node already
    carries B04/B08 scaled to reflectance (nodata decoded to NaN) with
    ``x``/``y`` coordinates — no manual scaling or coordinate reconstruction.
    """
    try:
        from pystac_client import Client

        catalog = Client.open(_STAC)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=_BBOX,
            datetime=_DATETIME,
            max_items=1,
        )
        item = next(search.items())
        tree = xr.open_datatree(
            item.assets["product"].href, engine="zarr", chunks={}
        )
    except StopIteration as exc:
        raise CaseSkipped("no Sentinel-2 product found for the query") from exc
    except Exception as exc:  # noqa: BLE001 — any failure → skip, not crash
        raise CaseSkipped(f"EOPF Sentinel-2 unavailable ({exc})") from exc

    r10m = tree["measurements/reflectance/r10m"].to_dataset()
    scene = (
        r10m[["b04", "b08"]]
        .rename(b04="red", b08="nir")
        .isel(y=slice(_Y0, _Y0 + _N), x=slice(_X0, _X0 + _N))
    )
    return scene, item.id


def main() -> None:
    scene, item_id = _load_scene()
    n = scene.sizes["y"] * scene.sizes["x"]
    print(f"  Sentinel-2 L2A {item_id}")
    print(
        f"  scene window: {dict(scene.sizes)}  ({n:,} pixels, B04=red/B08=NIR)"
    )

    ctx = EngineContext()
    print(f"  engine: {ctx.flavor}")
    ctx.from_dataset("scene", scene, chunks={"y": 256, "x": 256})

    sql = """
        SELECT x, y, (nir - red) / (nir + red) AS ndvi
        FROM scene
        ORDER BY y, x
    """
    show_sql(sql)

    for _ in measured("SQL NDVI"):
        got = ctx.sql_to_dataset(sql, dims=["y", "x"]).ndvi

    # Array reference: the same formula in pure xarray. ``.compute()`` reads the
    # window and evaluates it here (the scene is lazy), so this measures the same
    # read-and-compute the SQL side does — not just graph construction.
    for _ in measured("xarray reference"):
        ref = ((scene.nir - scene.red) / (scene.nir + scene.red)).compute()

    # Compare the xarray way — aligned by coordinate label, so the ORDER BY
    # above is enough and neither side needs an explicit sort.
    assert_grid_close("NDVI (per-pixel)", got, ref, rtol=1e-6)

    show_result(got)

    valid = ref.notnull()
    print(
        f"\n  NDVI over {int(valid.sum()):,} valid pixels: "
        f"min {float(ref.min()):.3f}, "
        f"mean {float(ref.mean()):.3f}, "
        f"max {float(ref.max()):.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(run_case(main, "NDVI: per-pixel column arithmetic"))
