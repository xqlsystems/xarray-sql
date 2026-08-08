# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "gcsfs",
#   "zarr>=3",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Diurnal climatology — the "rechunk + grouped reduction" that is a GROUP BY.

A *climatology* is the average value for each time-of-cycle, computed
independently at every location: "what is the typical temperature here at
06:00?" In the array paradigm (and in the coiled/benchmarks #1545 write-up)
this is the canonical painful workload — load native Zarr chunks, *rechunk* to
put all of time in one chunk ("pencils"), run a grouped reduction over the
calendar, then rechunk back to "pancakes" for output.

The rechunking exists only to serve the array layout. The *operation* is::

    SELECT latitude, longitude, hour_of_day, AVG("2m_temperature")
    GROUP BY latitude, longitude, hour_of_day

Group by location and time-of-cycle, average the rest — the same answer as
``da.groupby("time.hour").mean()``. ERA5 is hourly, so grouping by hour of day
gives a clean 24-bin **diurnal cycle**, one sample per day in the window.

We register the full ARCO-ERA5 archive as a lazy table, but the climatology here
is computed over a *bounded window* — a few summer days over a CONUS-ish box. The
``WHERE`` prunes the read, so the query touches only ``2m_temperature`` over that
window and never scans the rest of the archive. The point is not that we reduce
the whole record; it is that you can aim a query at a multi-decade archive and pay
only for the slice it asks for.
"""

from __future__ import annotations

import datetime

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
# A few days over a CONUS-ish box (ERA5 latitude descends; lon is 0–360°E).
_START, _END = datetime.datetime(2020, 6, 1), datetime.datetime(2020, 6, 3, 23)
_LAT_N, _LAT_S = 50.0, 25.0
_LON_W, _LON_E = 235.0, 290.0
_PARAMS = {
    "start": _START,
    "end": _END,
    "lat_s": _LAT_S,
    "lat_n": _LAT_N,
    "lon_w": _LON_W,
    "lon_e": _LON_E,
}


def main() -> None:
    # Open the full ARCO-ERA5 archive lazily — no data is read here. ERA5 mixes
    # surface (time, lat, lon) and atmospheric (… level …) variables, so register
    # it as two tables under an ``era5`` schema; the query below touches only the
    # surface table's 2m_temperature.
    try:
        import gcsfs  # noqa: F401 — required by the gs:// protocol

        ds = xr.open_zarr(_URL, chunks=None, storage_options={"token": "anon"})
    except Exception as exc:  # noqa: BLE001 — any failure → skip, not crash
        raise CaseSkipped(f"ARCO-ERA5 unavailable ({exc})") from exc

    ctx = EngineContext()
    print(f"  engine: {ctx.flavor}")
    with timed("register full ERA5 (lazy)"):
        ctx.from_dataset(
            "era5",
            ds,
            chunks={"time": 6},
            table_names={
                ("time", "latitude", "longitude"): "surface",
                ("time", "level", "latitude", "longitude"): "atmosphere",
            },
        )

    sql = """
        SELECT latitude,
               longitude,
               date_part('hour', time) AS hour,
               AVG("2m_temperature") - 273.15 AS clim_c
        FROM era5.surface
        WHERE time      BETWEEN $start AND $end
          AND latitude  BETWEEN $lat_s AND $lat_n
          AND longitude BETWEEN $lon_w AND $lon_e
        GROUP BY latitude, longitude, date_part('hour', time)
        ORDER BY latitude DESC, longitude, hour
    """
    show_sql(sql)

    # A climatology is a gridded product: round-trip the result back to an
    # xarray Dataset keyed by (latitude, longitude, hour) — how it is used.
    for _ in measured("SQL diurnal climatology (lazy read)"):
        got = ctx.sql_to_dataset(
            sql,
            dims=["latitude", "longitude", "hour"],
            param_values=_PARAMS,
        )

    # Array reference: the textbook groupby-over-the-cycle reduction, in °C —
    # the same lazy window, materialized only on demand.
    for _ in measured("xarray reference"):
        window = ds["2m_temperature"].sel(
            time=slice(_START, _END),
            latitude=slice(_LAT_N, _LAT_S),
            longitude=slice(_LON_W, _LON_E),
        )
        ref = window.groupby("time.hour").mean("time") - 273.15

    assert_grid_close(
        "diurnal climatology (°C)", got.clim_c, ref, rtol=1e-4, atol=1e-2
    )

    show_result(got)


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Climatology: GROUP BY lat, lon, hour (ARCO-ERA5)")
    )
