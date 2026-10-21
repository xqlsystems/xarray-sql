# Learn xarray-sql in 15 minutes

A tour of xarray-sql in one runnable Python file, after the style of
[Learn X in Y Minutes](https://learnxinyminutes.com/), a site that teaches
a language in a single commented source file. It's for readers who know
Python, NumPy or Xarray but not SQL, and for readers who know SQL but not
Xarray. Read it top to bottom; every `# =>` comment shows what its line
evaluates to.

To run it, install the engines it tours, copy the code into a file, and run
it with `python`:

```shell
pip install "xarray-sql[duckdb,polars,adbc]" adbc-driver-sqlite adbc-driver-flightsql
```

DuckDB downloads its `spatial` and `h3` extensions the first time the tour
installs them.

To try SQL over a real forecast without installing anything, use the
[demo that runs in your browser](browser.md).

## If you know SQL

[Xarray](https://xarray.dev/) holds N-dimensional arrays whose dimensions
have names, like `time`, `lat` and `lon`, and whose positions along each
dimension have labels, called coordinates. A *Dataset* is several such
arrays, called variables, on shared dimensions. xarray-sql shows a Dataset
to a SQL engine as a table with a column per dimension, a column per
variable, and a row per cell of the grid. Section 0 below draws that table.
Nothing is copied up front: the engine reads the arrays chunk by chunk
while a query runs.

## If you know Xarray or NumPy

Most of what you do with Xarray has a SQL spelling:

| Xarray | SQL |
|---|---|
| `ds[["temp"]]` | `SELECT temp` |
| `ds.sel(lat=20.0)` | `WHERE lat = 20.0` |
| `ds.temp.mean(["time", "lon"])` | `SELECT lat, AVG(temp) ... GROUP BY lat` |
| `ds.groupby("time.month").mean()` | `GROUP BY date_part('month', time)` |
| `xr.where(ds.rain > 5, 1, 0)` | `CASE WHEN rain > 5 THEN 1 ELSE 0 END` |
| `ds.rain.cumsum("time")`, `.rolling(...)` | window functions: `SUM(rain) OVER (... ORDER BY time)` |
| `a - b` on aligned arrays | `JOIN` on the dimension columns |
| `ds.temp.sum()` skipping NaN | `SUM(temp)` skipping NULL |

The [Geospatial page](geospatial.md) works through climatologies,
anomalies, regridding and more in both.

## The tour

```python
# Single-line comments start with a number sign.

"""Multi-line strings use three quotes. This page writes its SQL in them.
"# =>" shows what a line evaluates to.
"""

import numpy as np
import pandas as pd
import xarray as xr
import xarray_sql as xql

####################################################
## 0. The one idea: a Dataset is a table
####################################################

# Xarray, a Python library for labeled N-dimensional arrays, stores data as
# arrays with named dimensions ("dims") and a coordinate (a label) for each
# position along each dim. Here is three days of temperature and rain on a
# 2 x 2 grid of latitudes and longitudes:
ds = xr.Dataset(
    {
        "temp": (("time", "lat", "lon"), np.arange(12.0).reshape(3, 2, 2) + 10),
        "rain": (
            ("time", "lat", "lon"),
            [[[0, 1], [2, np.nan]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
        ),
    },
    coords={
        "time": pd.date_range("2024-01-01", periods=3),
        "lat": [10.0, 20.0],
        "lon": [100.0, 110.0],
    },
)
ds.temp.sel(time="2024-01-02", lat=20.0, lon=110.0).item()  # => 17.0

# xarray-sql "pivots" the Dataset into a table: a column per dim, a column
# per variable, and one row per cell of the grid. 3 x 2 x 2 cells is 12 rows:
#
#   time        lat    lon    temp  rain
#   2024-01-01  10.0  100.0   10.0   0.0
#   2024-01-01  10.0  110.0   11.0   1.0
#   2024-01-01  20.0  100.0   12.0   2.0
#   2024-01-01  20.0  110.0   13.0   NULL   <- NaN becomes SQL's NULL
#   2024-01-02  10.0  100.0   14.0   4.0
#   ...
#
# That's most of what there is to learn: a query sees cells as rows.

####################################################
## 1. Register a Dataset, then query it
####################################################

# A Context is a SQL session on DataFusion, the query engine xarray-sql
# bundles. from_dataset registers a Dataset under a table name. For a lazy
# Dataset, one opened from files or the cloud, nothing is read yet: each
# query reads only the chunks it needs, while it runs.
ctx = xql.Context()
ctx.from_dataset("weather", ds)

# sql() returns a lazy DataFrame; to_pandas() runs it.
ctx.sql("SELECT COUNT(*) AS n FROM weather").to_pandas()["n"][0]  # => 12


# A helper for the rest of the page: run a query, get its rows as tuples.
def rows(sql):
    return list(ctx.sql(sql).to_pandas().itertuples(index=False, name=None))


rows(
    "SELECT lat, lon, temp FROM weather WHERE time = '2024-01-02'"
    " ORDER BY lat, lon"
)
# => [(10.0, 100.0, 14.0), (10.0, 110.0, 15.0),
#     (20.0, 100.0, 16.0), (20.0, 110.0, 17.0)]

####################################################
## 2. SQL for Xarray users
####################################################

# A SELECT names the columns it wants. A query reads only those variables
# from the Dataset, so on wide data never write SELECT *.

# WHERE is .sel(): it keeps the rows that match.  ds.temp.sel(lat=20.0).mean():
rows("SELECT AVG(temp) FROM weather WHERE lat = 20.0")  # => [(16.5,)]

# GROUP BY is a reduction. Group by the dims you keep; the rest are
# reduced away.  ds.temp.mean(["time", "lon"]):
rows("SELECT lat, AVG(temp) AS temp FROM weather GROUP BY lat ORDER BY lat")
# => [(10.0, 14.5), (20.0, 16.5)]

# An aggregate skips NULLs, as ds.rain.sum() skips NaNs.
rows("SELECT SUM(rain), COUNT(rain), COUNT(*) FROM weather")
# => [(63.0, 11, 12)]

# ds.groupby("time.dayofweek") is GROUP BY on an expression of a column.
# (SQL's day of week counts from Sunday = 0; Xarray's from Monday = 0.)
rows("""
    SELECT date_part('dow', time) AS dow, MAX(temp) AS hottest
    FROM weather GROUP BY dow ORDER BY dow
""")
# => [(1, 13.0), (2, 17.0), (3, 21.0)]

# CASE is xr.where(cond, a, b).
rows("""
    SELECT SUM(CASE WHEN rain > 5 THEN 1 ELSE 0 END) AS wet_cells
    FROM weather
""")
# => [(6,)]

# Window functions compute along a dim without reducing it: .cumsum(),
# .rolling(), .diff(). PARTITION BY says which cells form one series;
# ORDER BY says which way the series runs.  ds.rain.cumsum("time"):
rows("""
    SELECT time, SUM(rain) OVER (PARTITION BY lat, lon ORDER BY time)
    FROM weather WHERE lat = 10.0 AND lon = 100.0 ORDER BY time
""")
# => [(Timestamp('2024-01-01 00:00:00'), 0.0),
#     (Timestamp('2024-01-02 00:00:00'), 4.0),
#     (Timestamp('2024-01-03 00:00:00'), 12.0)]

# A WITH clause (a "CTE") names a subquery, like assigning a variable.
# JOIN lines up rows from two tables by their keys, as Xarray aligns arrays
# by their coordinates. An anomaly, ds.temp - ds.temp.mean("time"):
rows("""
    WITH clim AS (
        SELECT lat, lon, AVG(temp) AS mean FROM weather GROUP BY lat, lon
    )
    SELECT w.time, w.temp - clim.mean AS anomaly
    FROM weather AS w JOIN clim ON w.lat = clim.lat AND w.lon = clim.lon
    WHERE w.lat = 10.0 AND w.lon = 100.0
    ORDER BY w.time
""")
# => [(Timestamp('2024-01-01 00:00:00'), -4.0),
#     (Timestamp('2024-01-02 00:00:00'), 0.0),
#     (Timestamp('2024-01-03 00:00:00'), 4.0)]

# Two Datasets on the same grid join on their dims. Register a second one:
ctx.from_dataset("sensor", (ds.temp + 0.5).to_dataset(name="reading"))
rows("""
    SELECT AVG(s.reading - w.temp) AS bias
    FROM weather AS w JOIN sensor AS s
      ON w.time = s.time AND w.lat = s.lat AND w.lon = s.lon
""")
# => [(0.5,)]

####################################################
## 3. Back to Xarray
####################################################

# to_dataset() turns a result back into a Dataset. Dims the result kept
# become dims again, and attributes come back from the registered Dataset.
zonal = ctx.sql(
    "SELECT time, lat, AVG(temp) AS temp FROM weather"
    " GROUP BY time, lat ORDER BY time, lat"
).to_dataset()
zonal.temp.dims  # => ('time', 'lat')
zonal.temp.sel(time="2024-01-03").values  # => array([18.5, 20.5])

# A column you computed isn't a dim of any registered Dataset, so name it:
by_dow = ctx.sql(
    "SELECT date_part('dow', time) AS dow, AVG(temp) AS temp"
    " FROM weather GROUP BY dow ORDER BY dow"
).to_dataset(dims=["dow"])
by_dow.temp.values  # => array([11.5, 15.5, 19.5])

# Results that arrive sorted by their dims, outermost first, rebuild with
# one reshape. ORDER BY the dims before turning a large result back.

####################################################
## 4. Variables with different dims
####################################################

# A table needs one set of dims, so a Dataset whose variables have
# different dims becomes several tables, one per set of dims, in a SQL
# schema (a namespace) named after the Dataset.
terrain = ds.assign(elevation=(("lat", "lon"), [[5.0, 50.0], [500.0, 1500.0]]))
ctx.from_dataset(
    "terrain",
    terrain,
    table_names={("time", "lat", "lon"): "daily", ("lat", "lon"): "static"},
)
rows("""
    SELECT s.elevation, AVG(d.temp) AS temp
    FROM terrain.daily AS d JOIN terrain.static AS s
      ON d.lat = s.lat AND d.lon = s.lon
    GROUP BY s.elevation ORDER BY s.elevation
""")
# => [(5.0, 14.0), (50.0, 15.0), (500.0, 16.0), (1500.0, 17.0)]
# Without table_names, a table is named after its dims: terrain.lat_lon.

####################################################
## 5. The same table on other engines
####################################################

# xarray-sql translates data, not queries: you write each engine's own
# SQL. xql.register(connection, name, ds) registers a Dataset on any
# engine it supports, and xql.to_dataset(result, template=ds) turns any
# engine's result back into a Dataset. Data moves between them as Apache
# Arrow, a columnar in-memory format that every engine here reads.

# DuckDB, an embedded analytical database:
import duckdb

con = duckdb.connect()
xql.register(con, "weather", ds)
hot = con.sql("""
    SELECT lat, lon, MAX(temp) AS temp FROM weather
    GROUP BY lat, lon ORDER BY lat, lon
""")
xql.to_dataset(hot, template=ds).temp.sel(lat=20.0).values
# => array([20., 21.])
# Without ORDER BY, rows (and so the Dataset's coordinates) come back in
# whatever order the engine's threads finish: SQL promises no order.

# Polars, a DataFrame library, scans the same lazy table as a pyarrow
# dataset. Polars has its own SQL too.
import polars as pl

lazy = pl.scan_pyarrow_dataset(xql.arrow_dataset(ds))
lazy.filter(pl.col("lat") == 10.0).select(pl.col("temp").max()).collect().item()
# => 19.0

# ADBC, the Arrow Database Connectivity API, has drivers for PostgreSQL,
# SQLite, Snowflake, BigQuery and more. A database in another process
# can't read a lazy Dataset, so registering copies the Dataset into a new
# table. Select the region and variables you need first.
import adbc_driver_sqlite.dbapi

with adbc_driver_sqlite.dbapi.connect() as sqlite:  # in memory
    xql.register(sqlite, "weather", ds[["rain"]].sel(lat=10.0))
    with sqlite.cursor() as cur:
        cur.execute("SELECT COUNT(*), SUM(rain) FROM weather")
        cur.fetchone()  # => (6, 27.0)

####################################################
## 6. Geometry and H3, in DuckDB
####################################################

# DuckDB's spatial extension adds geometry types and ST_ functions. Its
# community h3 extension adds H3, Uber's grid of hexagons over the globe,
# at 16 resolutions from continents to square meters.
con.execute("INSTALL spatial; LOAD spatial; INSTALL h3 FROM community; LOAD h3")

# geometry=(x, y) adds a point column built from two dims.
xql.register(con, "cells", ds, geometry=("lon", "lat"))

# ST_Within tests each point against a polygon. A function can't skip
# chunks, but BETWEEN on a dim can, so pair the polygon with its bounding
# box; bbox_conjuncts writes that part.
bbox = xql.bbox_conjuncts((95, 5, 115, 15), x="lon", y="lat")
bbox  # => '"lon" BETWEEN 95.0 AND 115.0 AND "lat" BETWEEN 5.0 AND 15.0'
con.sql(f"""
    SELECT AVG(temp) FROM cells
    WHERE {bbox}
      AND ST_Within(geometry, ST_GeomFromText(
          'POLYGON ((95 5, 115 5, 115 15, 95 15, 95 5))'))
""").fetchall()
# => [(14.5,)]

# H3 joins data of different shapes. Give every row the ID of the hexagon
# it falls in, and "is near" becomes "=", which any engine joins fast.
# Here a table of stations (points) meets the grid. (The page's grid is
# 10 degrees apart, so the hexagons are coarse: resolution 1, ~600,000 km².)
con.execute("""
    CREATE TABLE stations AS
    SELECT * FROM (VALUES ('Yangon', 16.8, 96.2), ('Hanoi', 21.0, 105.8))
        AS t(name, lat, lon)
""")
con.sql("""
    SELECT s.name, AVG(c.temp) AS temp
    FROM stations AS s JOIN cells AS c
      ON h3_latlng_to_cell(s.lat, s.lon, 1) = h3_latlng_to_cell(c.lat, c.lon, 1)
    GROUP BY s.name ORDER BY s.name
""").fetchall()
# => [('Hanoi', 17.0), ('Yangon', 16.0)]

# Polygons from files, like ST_Read('countries.geojson'), are tagged
# EPSG:4326, and ST_Within won't compare them with xarray-sql's default
# OGC:CRS84 tag (both mean longitude and latitude): register with
# geometry_crs="EPSG:4326". And join the geometry column to a table of
# polygons through ST_Point(lon, lat): DuckDB can't yet join it directly.

####################################################
## 7. Serve Datasets to any Flight SQL client
####################################################

# xql.serve turns the process into an Arrow Flight SQL server, a protocol
# for SQL over the network. Clients in any language query your Datasets
# without a copy: the server reads chunks only while a query runs.
server = xql.serve({"weather": ds})  # on 127.0.0.1, a free port

# A client: here, ADBC's Flight SQL driver. JDBC and ODBC tools work too.
import adbc_driver_flightsql.dbapi as flight_sql

with (
    # autocommit=True: the server is read-only, so it has no transactions.
    flight_sql.connect(server.uri, autocommit=True) as client,
    client.cursor() as cur,
):
    cur.execute(
        "SELECT time, AVG(temp) AS temp FROM weather"
        " GROUP BY time ORDER BY time"
    )
    xql.to_dataset(cur, template=ds).temp.values  # => array([11.5, 15.5, 19.5])

server.shutdown()  # in a script, server.wait() serves until Ctrl+C

# The server has no authentication or encryption, so it accepts only local
# connections unless you pass host=. Read the Engines page before you do.

####################################################
## 8. Gotchas
####################################################

# Quote a name in double quotes when it starts with a digit, has spaces or
# capitals, or is a SQL keyword: SELECT "2m_temperature" FROM era5.
# Single quotes make a string: WHERE time > '2024-01-02'.

# DataFusion plans statements lazily, so a statement that changes
# something only runs when you collect it:
ctx.sql("CREATE TABLE warm AS SELECT * FROM weather WHERE temp > 15").collect()
rows("SELECT COUNT(*) FROM warm")  # => [(6,)]

# A WHERE on a dim's coordinate skips the chunks it rules out, before
# reading them. A WHERE on a variable's values has to read every chunk.

# Chunks set how a lazy Dataset is read: from_dataset(..., chunks={"time":
# 24}) reads 24 time steps at a time, in parallel across chunks.
```

## Further reading

- [Examples](examples.md) queries ERA5, a global weather reanalysis, straight
  from the cloud.
- [Engines](engines.md) covers DataFusion, DuckDB, Polars, ADBC and serving
  over Flight SQL in depth, with a support matrix.
- [Geospatial in SQL](geospatial.md) poses climate and remote-sensing
  analysis as queries, including geometry columns and their pruning.
- [Performance](performance.md) explains chunk pruning and how to tune
  reads.
- [Known issues](limitations.md) lists what doesn't work yet.
