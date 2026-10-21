# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql[duckdb]",
#   "xarray",
#   "numpy",
#   "pandas",
#   "zarr>=3",
#   "fsspec",
#   "aiohttp",
#   "h3>=4",
#   "shapely>=2",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Rasters, points and polygons in one query: exact geometry, then H3.

Geospatial data comes in different shapes: a forecast is a raster, cities
are points, countries are polygons. Joining them answers everyday
questions, like "how hot will each country be tomorrow?" and "how many
people live where it will pass 35 °C?". This case asks both in DuckDB, an
embedded SQL database, two ways:

1. **Exact geometry**, with DuckDB's ``spatial`` extension.
   ``xql.register(..., geometry=("longitude", "latitude"))`` gives the
   raster a point ``geometry`` column, so ``ST_Within(geometry, country)``
   is a point-in-polygon test. A single country's query pairs it with
   ``xql.bbox_conjuncts``, so the lazy raster reads only the chunks over
   that country.
2. **H3**, Uber's hierarchical grid of hexagons, with DuckDB's community
   ``h3`` extension. Every shape becomes hexagon IDs: a grid cell or a city
   is the hexagon containing it, and a country is the hexagons that cover
   it. Each spatial relationship is then an equality ``JOIN`` on the ID,
   which any SQL engine runs as a hash join.

Exact geometry is exact, and slow on detailed outlines: every grid point
is tested against a country's thousands of vertices. H3 pays that cost
once, to cover the countries with hexagons, and answers every query after
that in milliseconds, within a small error. The case checks each step
against xarray, shapely and h3-py references, and reports how far H3's
answers are from the exact ones.

Data:

* the latest NOAA GFS forecast (0.25°, global), from dynamical.org's public
  Zarr store; the first 24 hours of 2 m temperature, read lazily;
* Natural Earth's countries (1:10m) and populated places (7,342 cities with
  their metropolitan populations), read by DuckDB straight from GeoJSON on
  GitHub.

The forecast changes every six hours, so the numbers do too.
"""

from __future__ import annotations

import duckdb
import h3
import numpy as np
import pandas as pd
import shapely
import xarray as xr

import xarray_sql as xql
from _harness import (
    CaseSkipped,
    assert_grid_close,
    measured,
    run_case,
    show_sql,
    timed,
)

_GFS = "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr"
_NATURAL_EARTH = (
    "/vsicurl/https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/"
)
_HOT = 35.0  # °C
# H3 resolution 4 hexagons average 1,770 km², about two GFS grid cells at
# the equator, so every hexagon holds at least one grid cell's center.
_RES = 4
# Countries whose exact means are checked against shapely: large and small,
# tropical and polar, one-piece and many islands.
_CHECKED = ["India", "Brazil", "Norway", "Indonesia", "Egypt"]


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    try:
        con.execute("INSTALL spatial; LOAD spatial")
        con.execute("INSTALL h3 FROM community; LOAD h3")
        con.execute(f"""
            CREATE TABLE countries AS
            SELECT NAME AS name, geom
            FROM ST_Read('{_NATURAL_EARTH}ne_10m_admin_0_countries.geojson');
            CREATE TABLE cities AS
            SELECT name, pop_max AS population, latitude, longitude, geom
            FROM ST_Read('{_NATURAL_EARTH}ne_10m_populated_places_simple.geojson');
        """)
    except duckdb.Error as exc:
        raise CaseSkipped(
            f"DuckDB extensions or Natural Earth ({exc})"
        ) from exc
    return con


def _outline(con: duckdb.DuckDBPyConnection, name: str):
    """A country's outline as WKT, and its bounds (xmin, ymin, xmax, ymax)."""
    wkt, *bounds = con.execute(
        "SELECT ST_AsText(geom), ST_XMin(geom), ST_YMin(geom),"
        " ST_XMax(geom), ST_YMax(geom) FROM countries WHERE name = ?",
        [name],
    ).fetchone()
    return wkt, bounds


def _inside(field: xr.DataArray, wkt: str, bounds) -> xr.DataArray:
    """The reference: ``field`` inside the outline, NaN elsewhere, cropped
    to the rows and columns with any cell inside."""
    xmin, ymin, xmax, ymax = bounds
    box = field.sel(latitude=slice(ymax, ymin), longitude=slice(xmin, xmax))
    lon, lat = np.meshgrid(box.longitude, box.latitude)
    inside = shapely.contains_xy(shapely.from_wkt(wkt), lon, lat)
    return (
        box.where(inside)
        .dropna("latitude", how="all")
        .dropna("longitude", how="all")
    )


def main() -> None:
    try:
        gfs = xr.open_zarr(_GFS, chunks=None)
    except Exception as exc:  # noqa: BLE001 — any failure → skip, not crash
        raise CaseSkipped(f"GFS unavailable ({exc})") from exc
    # The latest forecast's first 24 hours: (lead_time, latitude, longitude).
    gfs = gfs[["temperature_2m"]].isel(init_time=-1, lead_time=slice(0, 25))
    print(
        f"  raster: GFS forecast from {gfs.init_time.values}, "
        f"{gfs.sizes['lead_time']} hours × {gfs.sizes['latitude']}"
        f"×{gfs.sizes['longitude']}"
    )

    with timed("load DuckDB extensions + Natural Earth"):
        con = _connect()
    n_countries, n_cities = con.execute(
        "SELECT (SELECT count(*) FROM countries), (SELECT count(*) FROM cities)"
    ).fetchone()
    print(f"  vector: {n_countries} countries (polygons), {n_cities:,} cities")

    # DuckDB's ST_Read tags GeoJSON's longitude/latitude as EPSG:4326, and
    # ST_Within refuses to compare geometries with different tags, so tag
    # the raster's points the same way. (Both name WGS84 longitude and
    # latitude; xarray-sql's default tag is OGC:CRS84.)
    xql.register(
        con,
        "gfs",
        gfs,
        geometry=("longitude", "latitude"),
        geometry_crs="EPSG:4326",
    )

    # ── 1. One country, exact geometry, read lazily ──────────────────────
    name = "India"
    wkt, bounds = _outline(con, name)
    # ST_Within alone can't skip chunks: the engine doesn't push functions
    # into the scan. The bounding box's BETWEENs can, so the query reads
    # only the chunks over the country, and ST_Within refines the box.
    sql = f"""
        SELECT latitude, longitude, MAX(temperature_2m) AS t_max
        FROM gfs
        WHERE {xql.bbox_conjuncts(bounds, x="longitude", y="latitude")}
          AND ST_Within(geometry,
                        ST_SetCRS(ST_GeomFromText('{wkt}'), 'EPSG:4326'))
        GROUP BY latitude, longitude
        ORDER BY latitude DESC, longitude
    """
    show_sql(sql.replace(wkt, f"<{name}'s outline>"))
    for _ in measured(f"SQL: {name}'s 24-hour high, ST_Within + bbox pruning"):
        country = xql.to_dataset(con.sql(sql), template=gfs).t_max
    for _ in measured("shapely + xarray reference"):
        ref = _inside(gfs.temperature_2m.max("lead_time"), wkt, bounds)
    assert_grid_close(f"{name}'s 24-hour high (°C)", country, ref)
    print(
        f"  {name}: {int(country.count()):,} grid cells, mean high "
        f"{float(country.mean()):.1f} °C"
    )

    # ── 2. The 24-hour high everywhere, as a table ───────────────────────
    # Every later query uses each grid cell's high, so compute it once into
    # a DuckDB table. It keeps xarray-sql's point geometry for the
    # point-in-polygon joins below.
    with timed("SQL: 24-hour high for every grid cell (reads the globe)"):
        con.execute("""
            CREATE TABLE high AS
            SELECT latitude, longitude, MAX(temperature_2m) AS t_max,
                   ANY_VALUE(geometry) AS geometry
            FROM gfs GROUP BY latitude, longitude
        """)
    with timed("xarray reference: gfs.max('lead_time')"):
        ref_high = gfs.temperature_2m.max("lead_time").load()
    high = xql.to_dataset(
        con.sql(
            "SELECT latitude, longitude, t_max FROM high"
            " ORDER BY latitude DESC, longitude"
        ),
        template=gfs,
    ).t_max
    assert_grid_close("24-hour high (°C)", high, ref_high)

    # ── 3. Exact geometry: point-in-polygon joins ────────────────────────
    # A grid cell's area shrinks with the cosine of its latitude, so weight
    # each cell by it: an unweighted mean would overweight a country's
    # high latitudes, where the grid's points crowd together.
    exact_means_sql = """
        SELECT k.name AS country,
               SUM(h.t_max * cos(radians(h.latitude)))
                 / SUM(cos(radians(h.latitude))) AS t_max
        FROM high AS h JOIN countries AS k ON ST_Within(h.geometry, k.geom)
        GROUP BY k.name
    """
    show_sql(exact_means_sql, label="SQL: country mean highs, exact")
    for _ in measured(
        f"SQL: country mean highs, ST_Within ({high.size:,} points × "
        f"{n_countries} countries)"
    ):
        exact_means = con.sql(exact_means_sql).df().set_index("country").t_max
    for name in _CHECKED:
        cells = _inside(ref_high, *_outline(con, name))
        weight = np.cos(np.radians(cells.latitude))
        want = float(cells.weighted(weight.fillna(0)).mean())
        np.testing.assert_allclose(exact_means[name], want, rtol=1e-6)
    print(f"  ✅ PASS  exact means match shapely for {', '.join(_CHECKED)}")

    # A city's forecast is its nearest grid cell's. On a regular grid that
    # needs no geometry: round the city's coordinates to the grid's 0.25°
    # step (and wrap 180° to -180°), then join on equality.
    exact_people_sql = f"""
        WITH snapped AS (
            SELECT name, population, geom,
                   ROUND(latitude * 4) / 4 AS lat,
                   ((ROUND(longitude * 4) / 4 + 180) % 360) - 180 AS lon
            FROM cities
        )
        SELECT k.name AS country, SUM(c.population) AS people
        FROM snapped AS c
        JOIN high AS h ON h.latitude = c.lat AND h.longitude = c.lon
        JOIN countries AS k ON ST_Within(c.geom, k.geom)
        WHERE h.t_max > {_HOT}
        GROUP BY k.name
    """
    show_sql(exact_people_sql, label="SQL: people above 35 °C, exact")
    for _ in measured("SQL: people above 35 °C, nearest cell + ST_Within"):
        exact_people = con.sql(exact_people_sql).df().set_index("country")

    # Check the snapping join: each city's high, against indexing the
    # reference grid at the same rounded coordinates.
    cities = con.sql("""
        SELECT c.latitude, c.longitude, h.t_max
        FROM cities AS c JOIN high AS h
          ON h.latitude = ROUND(c.latitude * 4) / 4
         AND h.longitude = ((ROUND(c.longitude * 4) / 4 + 180) % 360) - 180
    """).df()

    def round_like_sql(v):  # halves away from zero, as SQL's ROUND does
        return np.sign(v) * np.floor(np.abs(v) + 0.5)

    lat = round_like_sql(cities.latitude.to_numpy() * 4) / 4
    lon = (
        round_like_sql(cities.longitude.to_numpy() * 4) / 4 + 180
    ) % 360 - 180
    want = ref_high.sel(
        latitude=xr.DataArray(lat), longitude=xr.DataArray(lon)
    ).values
    np.testing.assert_allclose(cities.t_max, want, rtol=1e-6)
    print(f"  ✅ PASS  each of {len(cities):,} cities' high is its grid cell's")

    # ── 4. H3: every shape becomes hexagon IDs ───────────────────────────
    # A hexagon belongs to the country that contains its center. A coastal
    # hexagon whose center is at sea goes to a country it overlaps, so
    # coastal cities aren't lost; each hexagon still has one country.
    # ST_Dump splits a country of several islands into its polygons.
    with timed(f"SQL: cover the countries with H3 resolution {_RES} hexagons"):
        con.execute(f"""
            CREATE TABLE country_cells AS
            WITH parts AS (
                SELECT name, UNNEST(ST_Dump(geom)).geom AS geom FROM countries
            ),
            centered AS (
                SELECT name,
                       UNNEST(h3_polygon_wkb_to_cells(ST_AsWKB(geom), {_RES}))
                           AS cell
                FROM parts
            ),
            touching AS (
                SELECT name,
                       UNNEST(h3_polygon_wkt_to_cells_experimental(
                           ST_AsText(geom), {_RES}, 'overlap')) AS cell
                FROM parts
            )
            SELECT name, cell FROM centered
            UNION ALL
            SELECT MIN(name), cell FROM touching
            WHERE cell NOT IN (SELECT cell FROM centered)
            GROUP BY cell
        """)
    with timed("SQL: each grid cell's hexagon, and each hexagon's mean high"):
        con.execute(f"""
            CREATE TABLE hex_high AS
            SELECT h3_latlng_to_cell(latitude, longitude, {_RES}) AS cell,
                   AVG(t_max) AS t_max
            FROM high GROUP BY cell
        """)

    # Hexagons have nearly equal areas, so a plain average of a country's
    # hexagons is already weighted by area.
    h3_means_sql = """
        SELECT k.name AS country, AVG(h.t_max) AS t_max
        FROM country_cells AS k JOIN hex_high AS h ON h.cell = k.cell
        GROUP BY k.name
    """
    show_sql(h3_means_sql, label="SQL: country mean highs, H3")
    for _ in measured("SQL: country mean highs, H3 equality JOIN"):
        h3_means = con.sql(h3_means_sql).df().set_index("country").t_max

    h3_people_sql = f"""
        SELECT k.name AS country, SUM(c.population) AS people
        FROM cities AS c
        JOIN hex_high AS h
          ON h.cell = h3_latlng_to_cell(c.latitude, c.longitude, {_RES})
        JOIN country_cells AS k ON k.cell = h.cell
        WHERE h.t_max > {_HOT}
        GROUP BY k.name
    """
    show_sql(h3_people_sql, label="SQL: people above 35 °C, H3")
    for _ in measured("SQL: people above 35 °C, H3 equality JOINs"):
        h3_people = con.sql(h3_people_sql).df().set_index("country")

    # Check the hexagons' means against h3-py, H3's Python bindings,
    # applied to the reference grid.
    with timed("h3-py reference: index every grid cell"):
        grid = ref_high.to_dataframe().reset_index()
        grid["cell"] = [
            h3.str_to_int(h3.latlng_to_cell(y, x, _RES))
            for y, x in zip(grid.latitude, grid.longitude)
        ]
        want_hex = grid.groupby("cell").temperature_2m.mean()
    got_hex = con.sql("SELECT cell, t_max FROM hex_high").df()
    got_hex = got_hex.set_index(got_hex.cell.astype("uint64")).t_max
    assert set(got_hex.index) == set(want_hex.index), "different hexagons"
    np.testing.assert_allclose(got_hex[want_hex.index], want_hex, rtol=1e-6)
    print(f"  ✅ PASS  {len(want_hex):,} hexagons' mean highs match h3-py")

    # ── How close H3 comes to exact geometry ─────────────────────────────
    both = pd.concat({"exact": exact_means, "h3": h3_means}, axis=1).dropna()
    diff = (both.h3 - both.exact).abs().sort_values()
    print(
        f"\n  Country mean highs, {len(diff)} countries: H3 is within "
        f"{diff.median():.2f} °C of exact for half of them, "
        f"{diff.quantile(0.9):.2f} °C for 90%."
    )
    print("  Largest differences (small or ragged countries, few hexagons):")
    for country in diff.index[-3:][::-1]:
        print(
            f"    {country:<24} exact {both.exact[country]:6.2f} °C,"
            f" H3 {both.h3[country]:6.2f} °C"
        )

    people = pd.concat(
        {"exact": exact_people.people, "h3": h3_people.people}, axis=1
    ).fillna(0)
    people = people.sort_values("exact", ascending=False)
    print(f"\n  People in cities above {_HOT:.0f} °C, top 10 countries:")
    print(f"\n    {'Country':<24} {'exact':>12} {'H3':>12}")
    for country, row in people.head(10).iterrows():
        print(f"    {country:<24} {int(row.exact):>12,} {int(row.h3):>12,}")
    print(
        f"    {'Total':<24} {int(people.exact.sum()):>12,}"
        f" {int(people.h3.sum()):>12,}"
    )
    print(
        "\n  H3 counts a city as hot when its hexagon's mean high is, and"
        " exact geometry\n  when its nearest grid cell's is. A coastal"
        " city's hexagon averages land with\n  cooler sea, so H3 counts"
        " fewer hot coastal cities."
    )


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Rasters × points × polygons: exact geometry and H3")
    )
