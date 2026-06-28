# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "xarray_sql",
#   "xarray[io]",
#   "gcsfs",
#   "numpy",
# ]
#
# [tool.uv.sources]
# xarray_sql = { path = "..", editable = true }
# ///
"""Differentiable SQL over ARCO-ERA5.

A minimal demonstration of xarray-sql's autograd: take a real climate archive
(ARCO-ERA5, read anonymously from GCS), express a physical quantity as an
*analytic* SQL formula over its variables, and let ``grad(...)`` differentiate
that formula symbolically — evaluated per grid cell, which is the relational
equivalent of ``jax.vmap(jax.grad(f))`` (each row is an independent point).

Note this is *symbolic* differentiation of an expression, not a finite-
difference spatial gradient: ``grad(f(u, v), u)`` is the exact partial
derivative of the formula ``f``, evaluated at every cell's values.

Two cases:

1. Wind-speed magnitude ``speed = sqrt(u^2 + v^2)``. Its sensitivity to the
   eastward wind is ``d(speed)/du = u / speed`` — checked exactly.

2. Saturation vapour pressure ``e_s(T)`` (August-Roche-Magnus form of the
   Clausius-Clapeyron relation). ``d(e_s)/dT`` governs how fast the atmosphere's
   moisture capacity grows with temperature — checked against the closed-form
   slope.

Run standalone (builds the local extension on first use):

    uv run benchmarks/grad_era5.py
"""

from __future__ import annotations

import time

import numpy as np
import xarray as xr

import xarray_sql as xql

ARCO_ERA5 = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

# ERA5 variable names start with a digit, so they must be double-quoted in SQL.
U = '"10m_u_component_of_wind"'
V = '"10m_v_component_of_wind"'
T = '"2m_temperature"'


def load_era5_block() -> xr.Dataset:
    """Open ARCO-ERA5 and pull one timestamp over a small region.

    Lazy open of the whole archive; only the requested block is read. We keep
    it to a few thousand cells so the demo runs in seconds.
    """
    full = xr.open_zarr(
        ARCO_ERA5, chunks=None, storage_options={"token": "anon"}
    )
    block = (
        full[
            [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]
        .sel(time="2020-01-01T00")
        # A ~North-America box (index-based to avoid lat-orientation pitfalls).
        .isel(latitude=slice(120, 200), longitude=slice(900, 1000))
        .load()
    )
    # One partition, so a SQL `ORDER BY latitude DESC` survives the round-trip
    # back to xarray (across multiple partitions, to_dataset reconstructs
    # coordinates in ascending order regardless of ORDER BY).
    return block.chunk()


def wind_speed_sensitivity(ctx: xql.XarrayContext, ref: xr.Dataset) -> None:
    """grad(sqrt(u^2 + v^2)) checked against the exact u / speed, v / speed."""
    speed = f"sqrt(power({U}, 2) + power({V}, 2))"
    out = ctx.sql(
        f"""
        SELECT
          latitude,
          longitude,
          {speed}            AS wind_speed,
          grad({speed}, {U}) AS d_speed_d_u,
          grad({speed}, {V}) AS d_speed_d_v
        FROM era5
        ORDER BY latitude DESC, longitude
        """
    ).to_dataset(dims=["latitude", "longitude"])

    u = ref["10m_u_component_of_wind"]
    v = ref["10m_v_component_of_wind"]
    speed_ref = np.sqrt(u**2 + v**2)

    xr.testing.assert_allclose(
        out["wind_speed"], speed_ref.rename("wind_speed")
    )
    xr.testing.assert_allclose(
        out["d_speed_d_u"], (u / speed_ref).rename("d_speed_d_u")
    )
    xr.testing.assert_allclose(
        out["d_speed_d_v"], (v / speed_ref).rename("d_speed_d_v")
    )
    print("  wind-speed sensitivity matches u/|w|, v/|w| exactly")
    print(out)


def clausius_clapeyron(ctx: xql.XarrayContext, ref: xr.Dataset) -> None:
    """grad(e_s(T)) checked against the closed-form Clausius-Clapeyron slope."""
    # August-Roche-Magnus: e_s(T) = A * exp(B * tc / (tc + C)), tc = T - 273.15.
    a, b, c = 6.1094, 17.625, 243.04
    tc = f"({T} - 273.15)"
    es = f"{a} * exp({b} * {tc} / ({tc} + {c}))"
    out = ctx.sql(
        f"""
            SELECT
              latitude,
              longitude,
              {es}            AS e_s,
              grad({es}, {T}) AS de_s_dt
            FROM era5
            ORDER BY latitude DESC, longitude
            """
    ).to_dataset(dims=["latitude", "longitude"])

    # Reference in float64 (the columns are float32): the exact derivative is
    #   d(e_s)/dT = e_s * B*C / (tc + C)^2.
    temp = ref["2m_temperature"].astype("float64")
    tc_ref = temp - 273.15
    es_ref = a * np.exp(b * tc_ref / (tc_ref + c))
    des_dt_ref = es_ref * (b * c) / (tc_ref + c) ** 2

    xr.testing.assert_allclose(out["e_s"], es_ref.rename("e_s"), rtol=1e-5)
    xr.testing.assert_allclose(
        out["de_s_dt"], des_dt_ref.rename("de_s_dt"), rtol=1e-5
    )
    print("  d(e_s)/dT matches the closed-form Clausius-Clapeyron slope")
    print(out)


def main() -> None:
    t0 = time.time()
    ds = load_era5_block()
    print(f"loaded ERA5 block {dict(ds.sizes)} in {time.time() - t0:.1f}s")

    ctx = xql.XarrayContext()
    ctx.from_dataset("era5", ds)

    print("\n== wind-speed sensitivity: grad(sqrt(u^2 + v^2)) ==")
    wind_speed_sensitivity(ctx, ds)

    print("\n== Clausius-Clapeyron: grad(e_s(T)) ==")
    clausius_clapeyron(ctx, ds)

    print("\nOK: symbolic SQL gradients match the analytic references.")


if __name__ == "__main__":
    main()
