"""Tests for the pyproj CRS-transform extension (`xarray_sql.proj`)."""

import numpy as np
import pyproj
import pytest
import xarray as xr
from datafusion import SessionContext

from xarray_sql import XarrayContext, proj

UTM10 = "EPSG:32610"  # UTM zone 10N (metres)
UTM11 = "EPSG:32611"  # UTM zone 11N (metres)
WGS84 = "EPSG:4326"  # lon/lat degrees
WEBMERC = "EPSG:3857"  # Web Mercator (metres)


@pytest.fixture
def utm_grid():
    """A 60x50 UTM zone 10N grid over the San Francisco Bay Area.

    Chunked into several partitions so DataFusion evaluates the UDF
    concurrently — exercising the per-thread transformer cache.
    """
    x = np.linspace(530_000.0, 630_000.0, 50)
    y = np.linspace(4_140_000.0, 4_250_000.0, 60)
    xx, yy = np.meshgrid(x, y)
    return xr.Dataset(
        {"value": (["y", "x"], np.hypot(xx, yy))},
        coords={"y": y, "x": x},
    ).chunk({"y": 15, "x": 50})


def test_reproject_matches_pyproj(utm_grid):
    ctx = XarrayContext()
    ctx.from_dataset("grid", utm_grid)
    result = ctx.sql(
        f"""
        SELECT x, y,
               reproject(x, y, '{UTM10}', '{WGS84}')['x'] AS lon,
               reproject(x, y, '{UTM10}', '{WGS84}')['y'] AS lat
        FROM grid
        ORDER BY y, x
        """
    ).to_pandas()

    transformer = pyproj.Transformer.from_crs(UTM10, WGS84, always_xy=True)
    ref_lon, ref_lat = transformer.transform(
        result["x"].to_numpy(), result["y"].to_numpy()
    )
    np.testing.assert_allclose(result["lon"], ref_lon, rtol=0, atol=1e-9)
    np.testing.assert_allclose(result["lat"], ref_lat, rtol=0, atol=1e-9)


def test_reproject_roundtrip(utm_grid):
    ctx = XarrayContext()
    ctx.from_dataset("grid", utm_grid)
    result = ctx.sql(
        f"""
        WITH lonlat AS (
          SELECT x, y,
                 reproject(x, y, '{UTM10}', '{WGS84}')['x'] AS lon,
                 reproject(x, y, '{UTM10}', '{WGS84}')['y'] AS lat
          FROM grid
        )
        SELECT x, y,
               reproject(lon, lat, '{WGS84}', '{UTM10}')['x'] AS rx,
               reproject(lon, lat, '{WGS84}', '{UTM10}')['y'] AS ry
        FROM lonlat
        ORDER BY y, x
        """
    ).to_pandas()
    # A metre-based CRS round-trips to well under a millimetre.
    np.testing.assert_allclose(result["rx"], result["x"], rtol=0, atol=1e-4)
    np.testing.assert_allclose(result["ry"], result["y"], rtol=0, atol=1e-4)


def test_per_row_crs():
    """The CRS arguments are expressions, so they may vary row by row."""
    lon = np.linspace(-125.9, -114.1, 24)  # spans UTM zones 10N and 11N
    lat = np.linspace(32.5, 41.5, 10)
    LON, LAT = np.meshgrid(lon, lat)
    pts = xr.Dataset(
        {
            "lon": (["i"], LON.ravel()),
            "lat": (["i"], LAT.ravel()),
        },
        coords={"i": np.arange(LON.size)},
    ).chunk({"i": LON.size})

    ctx = XarrayContext()
    ctx.from_dataset("pts", pts)
    result = ctx.sql(
        f"""
        SELECT lon, lat,
               reproject(lon, lat, '{WGS84}',
                         CASE WHEN lon < -120.0
                              THEN '{UTM10}' ELSE '{UTM11}' END)['x'] AS e,
               reproject(lon, lat, '{WGS84}',
                         CASE WHEN lon < -120.0
                              THEN '{UTM10}' ELSE '{UTM11}' END)['y'] AS n
        FROM pts
        ORDER BY i
        """
    ).to_pandas()

    for zone, mask in [
        (UTM10, result["lon"] < -120.0),
        (UTM11, result["lon"] >= -120.0),
    ]:
        transformer = pyproj.Transformer.from_crs(WGS84, zone, always_xy=True)
        ref_e, ref_n = transformer.transform(
            result.loc[mask, "lon"].to_numpy(),
            result.loc[mask, "lat"].to_numpy(),
        )
        np.testing.assert_allclose(
            result.loc[mask, "e"], ref_e, rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(
            result.loc[mask, "n"], ref_n, rtol=0, atol=1e-6
        )


def test_null_and_out_of_domain_yield_nan():
    ctx = XarrayContext()
    result = ctx.sql(
        f"""
        SELECT
          reproject(CAST(NULL AS DOUBLE), 45.0,
                    '{WGS84}', '{WEBMERC}')['x'] AS null_coord,
          reproject(0.0, 100.0, '{WGS84}', '{WEBMERC}')['y'] AS bad_lat,
          reproject(0.0, 45.0, CAST(NULL AS VARCHAR),
                    '{WEBMERC}')['x'] AS null_crs
        """
    ).to_pandas()
    assert np.isnan(result["null_coord"].iloc[0])
    assert np.isnan(result["bad_lat"].iloc[0])
    assert np.isnan(result["null_crs"].iloc[0])


def test_invalid_crs_raises():
    ctx = XarrayContext()
    with pytest.raises(Exception):
        ctx.sql(
            "SELECT reproject(0.0, 0.0, 'EPSG:999999', 'EPSG:4326')"
        ).to_pandas()


def test_register_on_plain_session_context_with_custom_name():
    ctx = SessionContext()
    proj.register(ctx, name="st_transform")
    result = ctx.sql(
        f"""
        SELECT st_transform(-122.0, 37.0, '{WGS84}', '{WEBMERC}')['x'] AS gx
        """
    ).to_pandas()
    transformer = pyproj.Transformer.from_crs(WGS84, WEBMERC, always_xy=True)
    ref_x, _ = transformer.transform(-122.0, 37.0)
    np.testing.assert_allclose(result["gx"].iloc[0], ref_x, rtol=0, atol=1e-6)
