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
"""Temperature anomaly — "broadcast-subtract the climatology" is a self-JOIN.

An *anomaly* is the departure of each observation from its climatological
normal: ``anomaly(t) = T(t) − climatology(hour-of-day(t))`` at each cell. The
array paradigm computes the climatology, then leans on xarray's grouped
broadcasting to line it back up with every timestep:
``ds.groupby("time.hour") - climatology``.

That broadcast — "attach each cell's normal back onto every matching timestep" —
is exactly a relational **JOIN** on the grouping key. So the anomaly is a
climatology CTE joined back to the raw observations::

    WITH clim AS (SELECT latitude, longitude, hour, AVG(T) ... GROUP BY ...)
    SELECT a.T - c.clim_t AS anomaly
    FROM era5 a JOIN clim c
      ON (a.latitude, a.longitude, hour(a.time)) = (c.latitude, c.longitude, c.hour)

We register the full ARCO-ERA5 archive as a lazy table, but the anomaly here is
computed over a *bounded window* (a few summer days over a CONUS-ish box): both
the climatology CTE and the outer scan read only ``2m_temperature``, and only
over the window the ``WHERE`` asks for — never the rest of the archive. You can
aim a query at the whole archive and pay only for the slice it asks for.
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
        WITH clim AS (
            SELECT latitude, longitude,
                   date_part('hour', time) AS hour,
                   AVG("2m_temperature") AS clim_t
            FROM era5.surface
            WHERE time      BETWEEN $start AND $end
              AND latitude  BETWEEN $lat_s AND $lat_n
              AND longitude BETWEEN $lon_w AND $lon_e
            GROUP BY latitude, longitude, date_part('hour', time)
        )
        SELECT a.time, a.latitude, a.longitude,
               a."2m_temperature" - c.clim_t AS anomaly
        FROM era5.surface a
        JOIN clim c
          ON a.latitude = c.latitude
         AND a.longitude = c.longitude
         AND date_part('hour', a.time) = c.hour
        WHERE a.time      BETWEEN $start AND $end
          AND a.latitude  BETWEEN $lat_s AND $lat_n
          AND a.longitude BETWEEN $lon_w AND $lon_e
        ORDER BY a.time, a.latitude DESC, a.longitude
    """
    show_sql(sql)

    # The anomaly is a gridded field; round-trip it to (time, lat, lon).
    for _ in measured("SQL anomaly (climatology CTE self-join, lazy read)"):
        got = ctx.sql_to_dataset(
            sql,
            dims=["time", "latitude", "longitude"],
            param_values=_PARAMS,
        )

    # Array reference: grouped broadcast-subtract, in pure xarray (lazy window).
    for _ in measured("xarray reference"):
        window = ds["2m_temperature"].sel(
            time=slice(_START, _END),
            latitude=slice(_LAT_N, _LAT_S),
            longitude=slice(_LON_W, _LON_E),
        )
        grouped = window.groupby("time.hour")
        ref = grouped - grouped.mean("time")

    assert_grid_close(
        "anomaly (T − diurnal climatology)",
        got.anomaly,
        ref,
        rtol=1e-3,
        atol=1e-2,
    )

    show_result(got)


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Anomaly: climatology CTE self-JOIN (ARCO-ERA5)")
    )
