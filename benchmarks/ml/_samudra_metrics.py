# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "samudra @ https://github.com/m2lines/Samudra/archive/6da59b02741d96d0656ef6ec90df833e370c3468.tar.gz",
#   "torch",
#   # Samudra reads and writes Zarr format 2 (zarr<3), which xarray
#   # stopped opening over HTTP in 2025.9.
#   "xarray>=2025.1.2,<2025.9",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Score saved ocean rollouts with Samudra's own observation metrics.

Samudra (https://github.com/m2lines/Samudra) is an emulator of the global
ocean, and ``samudra.metrics`` is the package its evaluation scores a
finished multi-year rollout with. It compares the rollout with
observations, not with the simulation it was trained on:

* sea surface temperature against OISST, NOAA's daily analysis of
  satellite and in-situ measurements;
* surface currents, derived from sea surface height (``zos``) as
  geostrophic flow, and their eddy kinetic energy, against DUACS, the
  Copernicus satellite altimetry product;
* ocean heat content from the surface to 700 m and from 700 m to
  2,000 m, against ARGO-IAP, a gridded analysis of Argo floats.

Samudra's authors publish the three products, already prepared, beside
their training data (https://m2lines.github.io/Samudra/docs/data/).
Errors are area-weighted RMSEs over the scoring window, 2015 to 2022 as
Samudra's configuration sets it, with uncertainty from resampling whole
calendar years.

This script runs that code on the rollouts ``04_samudra_sparse.py
--save DIR`` writes: ``DIR/om4.zarr`` (the OM4 simulation itself, the
best an emulator of it can hope to do), and one store per trained model.
A rollout with all 19 of OM4's depths is scored by
``samudra.metrics.report.compute_observation_metrics``, the function
Samudra's evaluation calls. Heat content integrates temperature over
every depth to 2,000 m, so a rollout with fewer depths is scored on sea
surface temperature and currents only, through the same module's
per-product functions. Heat content uses ``samudra.metrics``' heat
capacity, 3,850 J kg⁻¹ K⁻¹, not the 3,992 of OM4's ocean model, MOM6, so
its numbers compare with Samudra's metrics but not with heat content
computed elsewhere.

    uv run benchmarks/ml/_samudra_metrics.py DIR

writes ``DIR/observation_metrics.csv``, a row per metric, model, depth
and period, and prints the primary rows. The products are streamed over
HTTP; at full resolution they are tens of gigabytes, so a first run can
take hours.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import xarray as xr
from samudra.metrics import kernels, observations, report

OBSERVATIONS = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-07/obs/"
# Samudra's scoring window (its configs/data/obs.yaml).
WINDOW = ("2015-01-01", "2022-12-31")
PRIMARY = ["primary_complete_years", "full_overlap"]


class _Layout:
    """The one part of Samudra's data layout its metrics read here: OM4's
    processed releases are on a rectilinear (Gaussian) grid."""

    grid_type = "gaussian"


def product(name: str) -> xr.Dataset:
    """Open one observation product, as ``samudra.metrics`` standardizes
    it."""
    ds = xr.open_zarr(OBSERVATIONS + name, chunks={})
    return observations.with_cell_area(observations.standardize(ds))


def score(path: Path, label: str, window, bootstrap: int, products) -> list:
    """Score the rollout at ``path`` against the products."""
    rollout = observations.model_on_latlon_grid(xr.open_zarr(path), _Layout())
    if rollout.sizes.get("lev") == 19:
        frame = report.compute_observation_metrics(
            {label: rollout},
            duacs=products["duacs"],
            oisst=products["oisst"],
            argo=products["argo"],
            model_dz={
                label: observations.model_depth_thickness(rollout, _Layout())
            },
            window=window,
            bootstrap_samples=bootstrap,
        )
        return list(frame.to_dict("records"))
    rows: list = report._sst_metrics(
        label, rollout, products["oisst"], window, bootstrap
    )
    rows += report._velocity_metrics(
        label, rollout, products["duacs"], window, bootstrap, "absolute"
    )
    return list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "rollouts", type=Path, help="the directory --save wrote"
    )
    parser.add_argument(
        "labels",
        nargs="*",
        help="which rollouts to score (default: every .zarr in the directory)",
    )
    parser.add_argument("--start", default=WINDOW[0])
    parser.add_argument("--end", default=WINDOW[1])
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=kernels.DEFAULT_BOOTSTRAP_SAMPLES,
        help="calendar-year bootstrap draws for the intervals (0 disables)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    labels = args.labels or sorted(p.stem for p in args.rollouts.glob("*.zarr"))
    window = (pd.Timestamp(args.start), pd.Timestamp(args.end))
    products = {
        "duacs": product("duacs.zarr"),
        "oisst": product("oisst.zarr"),
        "argo": product("argo-iap.zarr"),
    }
    rows: list = []
    for label in labels:
        started = time.time()
        rows += score(
            args.rollouts / f"{label}.zarr",
            label,
            window,
            args.bootstrap,
            products,
        )
        print(f"Scored {label} in {time.time() - started:.0f} s.", flush=True)
        # Written after each rollout, so a long run keeps what it finished.
        pd.DataFrame(rows).to_csv(
            args.rollouts / "observation_metrics.csv", index=False
        )
    frame = pd.DataFrame(rows)
    primary = frame[frame["period_kind"].isin(PRIMARY)].fillna({"depth": ""})
    table = primary.pivot_table(
        index=["metric", "depth"], columns="model", values="value", dropna=False
    )
    print(table.dropna(how="all").to_string())


if __name__ == "__main__":
    main()
