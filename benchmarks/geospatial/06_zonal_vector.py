# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "numpy",
#   "gcsfs",
#   "zarr>=3",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Zonal statistics over regions — "rasterize the polygons, then mask" is a JOIN.

"What is the average temperature inside each region?" is the canonical
*raster × vector* operation. The array paradigm rasterizes each region to a
mask and reduces the raster under it, one region at a time. But a region is
just a row in a table of bounds, and "pixel falls inside region" is a **range
predicate** — so zonal statistics is a JOIN between the raster table and the
regions table, plus a GROUP BY::

    SELECT r.region, AVG(a."2m_temperature") - 273.15 AS avg_c
    FROM era5.surface a JOIN regions r
      ON  a.latitude  BETWEEN r.lat_min AND r.lat_max
      AND a.longitude BETWEEN r.lon_min AND r.lon_max
    GROUP BY r.region

This is exactly the README's promise — *joining tabular data with raster data* —
made concrete: the raster is the full **ARCO-ERA5** archive (``WHERE time …``
prunes it to one day), the regions are a second SQL table, and the spatial
relationship is an ordinary ``BETWEEN``.

Dataset: the full ARCO-ERA5 archive opened *lazily* — the table spans the whole
record, but the query aggregates only one day's window (the ``WHERE`` prunes the
read; it is not a scan of the full archive) — plus a handful of continental-scale
bounding boxes (longitudes in ERA5's 0–360°E convention).
"""

from __future__ import annotations

import datetime

import numpy as np
import xarray as xr

from _engines import EngineContext
from _harness import (
    CaseSkipped,
    assert_grid_close,
    measured,
    run_case,
    show_result,
    show_sql,
    timed,
)

_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_DAY = "2020-06-01"
_START, _END = (
    datetime.datetime(2020, 6, 1, 0),
    datetime.datetime(2020, 6, 1, 23),
)

# Continental-scale boxes (name, lat_min, lat_max, lon_min, lon_max), lon 0–360°E.
_REGIONS = [
    ("Sahara", 18.0, 30.0, 0.0, 30.0),
    ("Amazon", -10.0, 5.0, 290.0, 310.0),
    ("Australia_Outback", -30.0, -20.0, 125.0, 140.0),
    ("Greenland", 65.0, 80.0, 300.0, 340.0),
    ("SE_Asia", 5.0, 20.0, 95.0, 110.0),
]


def _regions_dataset() -> xr.Dataset:
    """A vector layer as an xarray Dataset: one row per region, bounds as vars."""
    bounds = np.array([r[1:] for r in _REGIONS], dtype="float64")
    return xr.Dataset(
        {
            "lat_min": (["region"], bounds[:, 0]),
            "lat_max": (["region"], bounds[:, 1]),
            "lon_min": (["region"], bounds[:, 2]),
            "lon_max": (["region"], bounds[:, 3]),
        },
        coords={"region": np.arange(len(_REGIONS))},
    ).chunk({"region": len(_REGIONS)})


def main() -> None:
    try:
        import gcsfs  # noqa: F401 — required by the gs:// protocol

        ds = xr.open_zarr(_URL, chunks=None, storage_options={"token": "anon"})
    except Exception as exc:  # noqa: BLE001 — any failure → skip, not crash
        raise CaseSkipped(f"ARCO-ERA5 unavailable ({exc})") from exc

    print(
        f"  raster: full ARCO-ERA5 ({ds.sizes['time']:,} timesteps, "
        f"{ds.sizes['latitude']}×{ds.sizes['longitude']})   "
        f"vector: {len(_REGIONS)} continental boxes"
    )

    ctx = EngineContext()
    print(f"  engine: {ctx.flavor}")
    with timed("register full ERA5 + regions"):
        ctx.from_dataset(
            "era5",
            ds,
            chunks={"time": 6},
            table_names={
                ("time", "latitude", "longitude"): "surface",
                ("time", "level", "latitude", "longitude"): "atmosphere",
            },
        )
        ctx.from_dataset(
            "regions", _regions_dataset(), chunks={"region": len(_REGIONS)}
        )

    sql = """
        SELECT r.region AS region_id,
               AVG(a."2m_temperature") - 273.15 AS avg_c,
               COUNT(*) AS n_obs
        FROM era5.surface a
        JOIN regions r
          ON  a.latitude  BETWEEN r.lat_min AND r.lat_max
          AND a.longitude BETWEEN r.lon_min AND r.lon_max
        WHERE a.time BETWEEN $start AND $end
        GROUP BY r.region
        ORDER BY r.region
    """
    show_sql(sql)

    for _ in measured("SQL zonal stats (raster × vector range JOIN)"):
        got = ctx.sql_to_dataset(
            sql,
            dims=["region_id"],
            param_values={"start": _START, "end": _END},
        )

    # Array reference: one lazy pass — stack the region masks and reduce. No
    # .load(): the day's field is read inside this timed block (exactly like the
    # SQL side), and reading it once for all regions is a single masked reduction
    # rather than a Python loop that would re-read the field per region.
    for _ in measured("xarray reference"):
        day = xr.open_zarr(
            _URL, chunks=None, storage_options={"token": "anon"}
        )["2m_temperature"].sel(time=_DAY)
        in_region = xr.concat(
            [
                (day.latitude >= lat_min)
                & (day.latitude <= lat_max)
                & (day.longitude >= lon_min)
                & (day.longitude <= lon_max)
                for _, lat_min, lat_max, lon_min, lon_max in _REGIONS
            ],
            dim="region_id",
        )
        ref = (
            day.where(in_region).mean(["time", "latitude", "longitude"])
            - 273.15
        ).assign_coords(region_id=got.region_id)

    assert_grid_close(
        "zonal mean per region (°C)", got.avg_c, ref, rtol=1e-4, atol=1e-2
    )

    show_result(got)

    print("\n  Region                 avg °C      n_obs")
    for (name, *_), avg, n in zip(_REGIONS, got.avg_c.values, got.n_obs.values):
        print(f"  {name:<20} {avg:7.2f}  {int(n):>10,}")


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Zonal stats: raster × vector range JOIN (ARCO-ERA5)")
    )
