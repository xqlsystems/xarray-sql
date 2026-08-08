# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "numpy",
#   "pandas",
#   "gcsfs",
#   "zarr>=3",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Forecast skill — scoring ML weather models against ERA5 is a JOIN + aggregate.

Scoring the **Pangu-Weather** and **GraphCast** machine-learning forecast models
against ERA5 ground truth is the headline workload of
[WeatherBench 2](https://weatherbench2.readthedocs.io/). A forecast is indexed by
*initialization time* and *lead time* (``prediction_timedelta``); the truth is
indexed by *valid time*. Evaluation aligns them by ``valid_time = init + lead``
and reduces the error to RMSE as a function of lead — the classic "error grows
with forecast horizon" curve.

That alignment is a relational **JOIN**, and ``valid_time = init + lead`` is just
timestamp + duration arithmetic the engine does natively::

    SELECT f.model, f.prediction_timedelta AS lead,
           SQRT(AVG(POWER(f.t - e.t, 2))) AS rmse
    FROM forecasts f
    JOIN era5 e
      ON  e.time = f.time + f.prediction_timedelta   -- valid_time = init + lead
      AND e.latitude  = f.latitude
      AND e.longitude = f.longitude
    GROUP BY f.model, f.prediction_timedelta

We stack the two models along a ``model`` dimension into a single forecast
table, so one query scores them together, grouped by the ``model`` column. The
forecasts and ERA5 are opened lazily, and the JOIN reads only what it needs.

Datasets: WeatherBench 2 **Pangu**, **GraphCast**, and **ERA5** at a coarse
64×32 grid (so the demo is small and fast), read from the public ``gs://
weatherbench2`` bucket. Requires network; skips cleanly offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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

_GRID = "64x32_equiangular_conservative"
_ERA5 = f"gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-{_GRID}.zarr"
_PANGU = f"gs://weatherbench2/datasets/pangu/2018-2022_0012_{_GRID}.zarr"
_GRAPHCAST = (
    "gs://weatherbench2/datasets/graphcast/2020/"
    f"date_range_2019-11-16_2021-02-01_12_hours-{_GRID}.zarr"
)
_VAR = "2m_temperature"
_INIT = slice("2020-01-01", "2020-01-10")  # 20 init times (12-hourly)


def _open(url: str) -> xr.Dataset:
    try:
        import gcsfs  # noqa: F401

        # decode_timedelta=True: forecasts store prediction_timedelta as a
        # real duration (and it silences xarray's decode-timedelta warning).
        return xr.open_zarr(
            url,
            chunks=None,
            storage_options={"token": "anon"},
            decode_timedelta=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise CaseSkipped(f"WeatherBench2 unavailable ({exc})") from exc


def _reference_rmse(forecasts: xr.Dataset, truth: xr.Dataset) -> xr.DataArray:
    """xarray reference: per (model, lead), align truth at valid_time, take RMSE.

    The 64×32 windows are tiny, so the reference reads them into memory and
    reduces there; the SQL side above stays lazy. We use ``.compute()`` rather
    than ``.load()`` deliberately: ``.load()`` caches the data *in place* on the
    shared ``forecasts``/``truth`` objects (which the SQL table also reads from),
    which would let a profiled reference serve a warm read — ``.compute()``
    returns a fresh array and leaves the inputs lazy, so each measurement is cold.
    """
    f = forecasts[_VAR].compute()
    e = truth[_VAR].compute()
    leads = f.prediction_timedelta.values
    per_lead = []
    for lead in leads:
        e_at_valid = e.sel(time=f.time.values + lead)  # (init, lat, lon)
        diff = f.sel(prediction_timedelta=lead) - e_at_valid.values
        per_lead.append(
            np.sqrt((diff**2).mean(["time", "latitude", "longitude"]))
        )
    return (
        xr.concat(per_lead, dim="lead")
        .assign_coords(lead=leads)
        .transpose("model", "lead")
    )


def main() -> None:
    # Open everything lazily — no .load() here.
    era5 = _open(_ERA5)

    # The two models store different pressure-level sets (Pangu 13, GraphCast
    # 37), so we keep the common surface field 2m_temperature and stack the
    # models along a `model` dimension into one forecast table. Snap the grid
    # onto ERA5's exact coordinates (same 64×32 grid) so the join on latitude and
    # longitude lines up exactly across the two Zarr stores.
    pangu = _open(_PANGU)[[_VAR]].sel(time=_INIT)
    graphcast = _open(_GRAPHCAST)[[_VAR]].sel(time=_INIT)
    forecasts = xr.concat([pangu, graphcast], dim="model").assign_coords(
        model=["pangu", "graphcast"],
        latitude=era5.latitude.values,
        longitude=era5.longitude.values,
    )

    # ERA5 truth must span every valid time (last init + longest lead); bound it
    # lazily so the JOIN does not scan the whole 1959–2023 record.
    valid_max = (
        pangu.time.values.max() + pangu.prediction_timedelta.values.max()
    )
    truth = era5[[_VAR]].sel(time=slice(_INIT.start, pd.Timestamp(valid_max)))

    print(
        f"  64×32 2m_temperature | init {_INIT.start}…{_INIT.stop} "
        f"({pangu.sizes['time']} inits × {pangu.sizes['prediction_timedelta']} "
        f"leads × 2 models)"
    )

    ctx = EngineContext()
    print(f"  engine: {ctx.flavor}")
    # chunks here is the Arrow batch (partition) size each table streams in, not a
    # filter — no data is dropped. Both windows are small, so one partition each is
    # fastest (fewer partitions = fewer Python→Arrow round-trips for the same
    # rows); time:100 covers both the ~40 forecast inits and the ~79 truth steps.
    # Empirically the truth chunk is what matters — splitting it small costs ~3×,
    # while the forecasts chunk is in the noise — and a chunk *mismatch* costs
    # nothing, so there is no need to keep them different.
    ctx.from_dataset("forecasts", forecasts, chunks={"time": 100})
    ctx.from_dataset("era5", truth, chunks={"time": 100})

    sql = """
        SELECT f.model,
               f.prediction_timedelta AS lead,
               SQRT(AVG(POWER(
                   CAST(f."2m_temperature" AS DOUBLE) - e."2m_temperature", 2
               ))) AS rmse
        FROM forecasts f
        JOIN era5 e
          ON  e.time = f.time + f.prediction_timedelta  -- valid = init + lead
          AND e.latitude  = f.latitude
          AND e.longitude = f.longitude
        GROUP BY f.model, f.prediction_timedelta
        ORDER BY f.model, lead
    """
    show_sql(sql)

    for _ in measured("SQL RMSE by (model, lead) — lazy JOIN"):
        got = ctx.sql_to_dataset(sql, dims=["model", "lead"]).rmse

    for _ in measured("xarray reference"):
        ref = _reference_rmse(forecasts, truth)

    assert_grid_close("RMSE(model, lead)", got, ref, rtol=1e-4, atol=1e-3)

    show_result(got)

    # Headline: error growth with forecast horizon, both models. The gridded SQL
    # result round-trips to a pandas table directly — index is lead (in days),
    # one column per model.
    table = (
        got.assign_coords(lead=got["lead"].values / np.timedelta64(1, "D"))
        .to_pandas()
        .T
    )
    table.index.name = "lead (days)"
    print("\n  2m-temperature RMSE (K) vs lead — lower is better:\n")
    print(table.iloc[::4].round(3).to_string())


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Forecast skill: Pangu vs GraphCast vs ERA5 (WB2)")
    )
