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
"""Zonal mean — the array reduction that is secretly a GROUP BY.

A *zonal mean* averages a field around each circle of latitude (over all
longitudes, and here over a day of hours too), collapsing a 3-D field to a 1-D
profile of value-vs-latitude — the classic pole-to-pole temperature curve. In
the array paradigm this is ``da.mean(dim=["longitude", "time"])``, a reduction
over two axes.

Relationally it is nothing more than::

    SELECT latitude, AVG("2m_temperature") GROUP BY latitude

The "axes" we reduce over are just the columns we *don't* group by. Same answer,
and the SQL reads like the plain-English definition of a zonal mean.

Dataset: the full **ARCO-ERA5** archive (0.25° global, 1.3M hourly timesteps).
The table is the whole reanalysis; ``WHERE time …`` prunes it to one day, and
the GROUP BY produces a 721-point global temperature profile.
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
# One day of hourly data, global; the WHERE below prunes ERA5 to this window.
_DAY = "2020-06-01"
_START, _END = (
    datetime.datetime(2020, 6, 1, 0),
    datetime.datetime(2020, 6, 1, 23),
)


def main() -> None:
    # Open the full ARCO-ERA5 archive (lazy, dask off) — no slicing here; the
    # SQL WHERE clause prunes it to the window we ask for.
    try:
        import gcsfs  # noqa: F401 — required by the gs:// protocol

        ds = xr.open_zarr(_URL, chunks=None, storage_options={"token": "anon"})
    except Exception as exc:  # noqa: BLE001 — any failure → skip, not crash
        raise CaseSkipped(f"ARCO-ERA5 unavailable ({exc})") from exc

    print(
        f"  ARCO-ERA5: {ds.sizes['time']:,} hourly timesteps, "
        f"{ds.sizes['latitude']}×{ds.sizes['longitude']} grid, "
        f"{len(ds.data_vars)} variables (no pre-slicing)"
    )

    # ERA5 mixes surface (time, lat, lon) and atmospheric (… level …) variables,
    # so register it as two tables under an ``era5`` schema.
    ctx = EngineContext()
    print(f"  engine: {ctx.flavor}")
    with timed("register full ERA5"):
        ctx.from_dataset(
            "era5",
            ds,
            chunks={"time": 6},
            table_names={
                ("time", "latitude", "longitude"): "surface",
                ("time", "level", "latitude", "longitude"): "atmosphere",
            },
        )

    # Pass the day's bounds as query parameters; the query still reads only that
    # one day out of the whole archive.
    sql = """
        SELECT latitude,
               AVG("2m_temperature") - 273.15 AS air_mean_c
        FROM era5.surface
        WHERE time BETWEEN $start AND $end
        GROUP BY latitude
        ORDER BY latitude DESC
    """
    show_sql(sql)

    # Round-trip the profile back to an xarray Dataset keyed by latitude.
    for _ in measured("SQL zonal mean (reads one day)"):
        got = ctx.sql_to_dataset(
            sql,
            dims=["latitude"],
            param_values={"start": _START, "end": _END},
        )

    # Array reference: reduce the same day over the two un-grouped axes.
    for _ in measured("xarray reference"):
        ref = (
            ds["2m_temperature"].sel(time=_DAY).mean(["longitude", "time"])
            - 273.15
        )

    assert_grid_close(
        "zonal mean (2m_temp vs latitude, °C)",
        got.air_mean_c,
        ref,
        rtol=1e-4,
        atol=1e-3,
    )

    show_result(got)

    print("\n  Global temperature profile (every 72nd parallel, °C):")
    print(got.air_mean_c.isel(latitude=slice(None, None, 72)).to_series())


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Zonal mean: GROUP BY latitude (ARCO-ERA5)")
    )
