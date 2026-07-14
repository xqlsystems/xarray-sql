# xarray-sql

_Query [Xarray](https://xarray.dev/) with SQL_

[![ci](https://github.com/alxmrs/xarray-sql/actions/workflows/ci.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/ci.yml)
[![lint](https://github.com/alxmrs/xarray-sql/actions/workflows/lint.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/lint.yml)
[![ci-build](https://github.com/alxmrs/xarray-sql/actions/workflows/ci-build.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/ci-build.yml)
[![ci-rust](https://github.com/alxmrs/xarray-sql/actions/workflows/ci-rust.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/ci-rust.yml)

```shell
pip install xarray-sql
```

## What is this?

This is an experiment to provide a SQL interface for array datasets.
Succinctly, we "pivot" Xarray Datasets to treat them like tables so we can run
SQL queries against them — on the query engine of your choice. xarray-sql
translates data, not queries: it registers a lazy Dataset as a table on
DataFusion (built in), DuckDB, or Polars, and turns any engine's Arrow result
back into a labeled Dataset. Dialects, geometry functions, and optimizers stay
with the engine.

## Quickstart

Open a Dataset, register it as a table with `from_dataset`, compute a
climatology in SQL, then write the result back to Xarray and plot it:

> **Note:** this example also needs `pooch` and a netCDF backend (for the
> tutorial download) and `matplotlib` (for the plot):
> `pip install pooch netCDF4 matplotlib`.

```python
import xarray as xr
import xarray_sql as xql

# 4x-daily surface air temperature on a lat/lon grid, 2013-2014.
ds = xr.tutorial.open_dataset('air_temperature')

ctx = xql.XarrayContext()
ctx.from_dataset('air', ds, chunks=dict(time=100))

# A climatology — the mean annual cycle — computed in SQL: average air
# temperature for each month of the year, over all grid cells and years.
clim = ctx.sql('''
  SELECT
    CAST(date_part('month', "time") AS INTEGER) AS month,
    AVG("air") AS air
  FROM "air"
  GROUP BY CAST(date_part('month', "time") AS INTEGER)
  ORDER BY month
''')

# Round-trip the result back to Xarray. `month` is a derived column, so name
# it as the dimension.
clim_ds = clim.to_dataset(dims=["month"])

# Plot the annual cycle as a time series.
clim_ds["air"].plot()  # in a script, call matplotlib.pyplot.show() to display
```

That's the round trip — Xarray in, SQL in the middle, Xarray (and a plot) back
out.

The same Dataset registers on other engines with one call — DuckDB gets a
native lazy table with predicate pushdown, Polars scans the same object:

```python
import duckdb

con = duckdb.connect()
xql.register(con, 'air', ds, chunks=dict(time=100))
rel = con.sql('SELECT time, AVG("air") AS air FROM air GROUP BY time ORDER BY time')
xql.to_dataset(rel, template=ds)   # any engine's Arrow result round-trips
```

See [Engines](docs/engines.md) for the support matrix, DuckDB/Polars details,
and the lazy chunked round-trip.

## A bigger example: ARCO-ERA5

The same interface scales to cloud-native datasets with hundreds of variables,
like [ARCO-ERA5](https://github.com/google-research/arco-era5).

> **Note:** reading from `gs://` requires `gcsfs` (`pip install gcsfs`).

```python
import xarray as xr
import xarray_sql as xql


# Open ARCO-ERA5 — a weather dataset with 273 variables since 1940. 
# Turning off dask means we don't have to wait to construct a task graph.
ds = xr.open_zarr(
  'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
  chunks=None,  # Turn dask off
  storage_options={'token': 'anon'}  # Anonymous read from the public GCS bucket — no auth required.
)

ctx = xql.XarrayContext()
# Make sure to pass `chunks`!
ctx.from_dataset('era5', ds, chunks=dict(time=6), table_names={
    ('time', 'latitude', 'longitude'): 'surface',
    ('time', 'level', 'latitude', 'longitude'): 'atmosphere',
})
# Registration takes ~10s on my machine.

# Heads up: ARCO-ERA5 has 262 surface + 11 atmospheric variables. The library
# pushes column projection down to Zarr, so SELECT only fetches what you ask
# for — but `SELECT * FROM era5.surface` would try to pull every variable
# across the year (terabytes from GCS). 
#  ---> Always SELECT specific columns. <---

# Average 2m-temperature over NYC on the morning of 2020-01-01. The library
# pushes WHERE clauses on dimension columns down to partition pruning.
ctx.sql('''
  SELECT AVG("2m_temperature") - 273.15 AS avg_c
  FROM era5.surface
  WHERE time BETWEEN TIMESTAMP '2020-01-01'
                 AND TIMESTAMP '2020-01-01 05:00:00'
    AND latitude  BETWEEN 39 AND 40
    AND longitude BETWEEN 286 AND 287  -- ERA5 uses 0-360 longitudes
''').to_pandas()
#       avg_c
# 0  8.640069

# Average temperature per pressure level, globally. 
result = ctx.sql('''
  SELECT level, AVG(temperature) - 273.15 AS avg_c
  FROM era5.atmosphere
  WHERE time BETWEEN TIMESTAMP '2020-01-01'
                 AND TIMESTAMP '2020-01-01 05:00:00'
  GROUP BY level
  ORDER BY level DESC
''')
# DataFrame()
# +-------+----------------------+
# | level | avg_c                |
# +-------+----------------------+
# | 1000  | 6.6210120796502565   |
# | 975   | 5.185637919348153    |
# | 950   | 4.028428657263021    |
# | 925   | 3.0828117974912743   |
# | 900   | 2.2109172992531967   |
# | 875   | 1.395017610194202    |
# | 850   | 0.6342670572626616   |
# | 825   | -0.21037158786759846 |
# | 800   | -1.1810754318269687  |
# | 775   | -2.3064649711534457  |
# +-------+----------------------+

ctx.sql('''
  SELECT latitude, longitude, AVG("2m_temperature") - 273.15 AS avg_c
  FROM era5.surface
  WHERE time BETWEEN TIMESTAMP '2020-01-01'
                 AND TIMESTAMP '2020-01-01 05:00:00'
  GROUP BY latitude, longitude
  ORDER BY latitude DESC, longitude
# `latitude`/`longitude` are inferred from the registered table's surviving
# dims; `template` is kept only to recover metadata (attrs, encoding).
''').to_dataset(template=ds)
# <xarray.Dataset> Size: 8MB
# Dimensions:    (latitude: 721, longitude: 1440)
# Coordinates:
#   * latitude   (latitude) float32 3kB 90.0 89.75 89.5 ... -89.5 -89.75 -90.0
#   * longitude  (longitude) float32 6kB 0.0 0.25 0.5 0.75 ... 359.2 359.5 359.8
# Data variables:
#     avg_c      (latitude, longitude) float64 8MB -26.84 -26.84 ... -27.38 -27.38
# Attributes:
#     last_updated:           2026-06-20 02:33:34.265980+00:00
#     valid_time_start:       1940-01-01
#     valid_time_stop:        2025-12-31
#     valid_time_stop_era5t:  2026-06-14
```

_(A runnable version of this example lives at
[`perf_tests/era5_temp_profile.py`](perf_tests/era5_temp_profile.py).)_

## Why build this?

A few reasons:

* Even though SQL is the lingua franca of data, scientific datasets are often
  inaccessible to non-scientists (SQL users).
* Joining tabular data with raster data is common yet difficult. It could be
  easy.
* There are many cloud-native, Xarray-openable datasets,
  from [Google Earth Engine](https://github.com/google/Xee)
  to the [Source Cooperative](https://source.coop/products?tags=zarr). Wouldn’t it be great if these
  were also SQL-accessible? How can the bridge be built with minimal effort?

This is a light-weight way to prove the value of the interface.

The larger goal is to explore the hypothesis that the [Pangeo
ecosystem is a scientific database](https://www.hytradboi.com/2025/c18b8cdc-fd17-4099-9c03-eb107217f627-pangeo-is-a-database). Here, xarray-sql can be thought of as a missing
DB front end.

## How does it work?

All chunks in a Xarray Dataset are transformed into a Dask DataFrame via
`from_map()` and `to_dataframe()`. For SQL support, we just use `dask-sql`.
That's it!

_2025 update_: This library now implements a Dask-like `from_map` interface in
pure DataFusion and PyArrow, but works with the same principle!

_2026 update_: Instead of `from_map()`, we create a way to translate Xarray chunks
into Arrow RecordBatches. We pass a Python callback into a DataFusion `TableProvider`
that lets the DB engine translate the underlying Dataset arrays into DataFusion partitions.
The same chunks-to-batches translation is also exposed as a
`pyarrow.dataset.Dataset` with predicate and projection pushdown, which is how
DuckDB and Polars consume registered Datasets with no engine-specific code.
Ultimately, the initial insight of the `pivot()` function -- that any ndarray can be
translated into a 2D table -- underlies this performant query mechanism. 

## Does it work?

Yes. The recurring worry is that the SQL interface is a toy — fine for `SELECT`s,
but not for the operations geoscience actually runs. So we wrote a suite that
takes the staples of geospatial and climate analysis — the ones we assume *need*
an array library — and expresses each one in SQL, then **checks the SQL answer
against an xarray/array reference** to floating-point tolerance:

* **Spectral indices** (NDVI) — column arithmetic over a real Sentinel-2 scene.
* **Climatology, anomalies, zonal means** — `GROUP BY` and self-`JOIN` against
  the 0.25° **ARCO-ERA5** archive registered as a lazy table. Each query is
  bounded to a small window (a few days over a region) and reads only that
  slice — the point is that you can aim a query at a multi-decade archive and
  pay only for the data it asks for, not that the query scans the whole record.
* **Forecast skill** — scoring the **Pangu-Weather** and **GraphCast** ML models
  against ERA5 (WeatherBench 2) as a `JOIN` on `valid_time = init + lead`; it
  reproduces the published result that GraphCast beats Pangu at every lead.
* **Raster × vector zonal stats** — a range `JOIN` of the ERA5 grid against a
  table of regions.
* **Reprojection and regridding** — a scalar PROJ UDF (validated against Earth
  Engine's own geodesy via [Xee](https://github.com/google/Xee)) and a
  sparse-weight-table `JOIN` (regridding real SRTM terrain).

Every case matches its array reference. The headline finding: these operations
are not really "array" operations at all — they are `GROUP BY`, `JOIN`, window
functions, and `CASE` in disguise, and a query engine runs them at scale. See
[`benchmarks/geospatial/`](benchmarks/geospatial/) and the write-up,
[Geospatial operations are relational operations](docs/geospatial.md).

## Why does this work?

Underneath Xarray, Dask, and Pandas, there are NumPy arrays. These are paged in
chunks and represented contiguously in memory. It is only a matter of metadata
that breaks them up into ndarrays. `pivot()`, which uses `to_dataframe()`,
just changes this metadata (via a `ravel()`/`reshape()`), back into a column
amenable to a DataFrame. We take advantage of this light weight metadata change to
make chunked information scannable by a DB engine (DataFusion, DuckDB, Polars —
anything that speaks Arrow).

## What are the current limitations?

The sharp edges we know about — per engine and fundamental — are cataloged in
[Known issues & limitations](docs/limitations.md). Currently, we're looking for
early users – "tire kickers", if you will. We'd love your input to shape the direction of this
project! Please, give this a try and [file issues](https://github.com/alxmrs/xarray-sql/issues) as
you see fit. Check out our [contributing guide](CONTRIBUTING.md), too 😉.

## What would a deeper integration look like?

I have a few ideas so far. One approach involves applying operations directly on
Xarray Datasets. This approach is being pursued
[here](https://github.com/google/weather-tools/tree/main/xql), as `xql`.

Deeper still: I was thinking we could make
a [virtual](https://fsspec.github.io/kerchunk/)
filesystem for parquet that would internally map to Zarr. Raster-backed virtual
parquet would open up integrations to numerous tools like dask, pyarrow, duckdb,
and BigQuery. More thoughts on this
in [#4](https://github.com/alxmrs/xarray-sql/issues/4).

_2025 update_: Something like this is being built across a few projects! The ones I know about are:

- [CartoDB's Raquet](https://github.com/CartoDB/raquet)
- The DataFusion community's [arrow-zarr](https://github.com/datafusion-contrib/arrow-zarr)

_2026 update_: A colleague and I are experimenting with native Zarr RDBMS engines. Check out:

- [Zarr-Datafusion](https://lib.rs/crates/zarr-datafusion)
- [DuckDB-Zarr](https://github.com/alxmrs/duckdb-zarr)

## Roadmap

- [x] ~Lazy evaluation via the pyarrow Dataset interface [#93](https://github.com/alxmrs/xarray-sql/issues/93).~ _Implemented in [#100](https://github.com/alxmrs/xarray-sql/pull/100)_
- [x] Support proper parallelism via proper partition handling on the rust/datafusion side. [#106](https://github.com/alxmrs/xarray-sql/issues/106)
- [x] Support core datafusion optimizations to scan less data, like [104](https://github.com/alxmrs/xarray-sql/issues/104), ...
- [x] Translate a single Zarr to a collection of tables [#85](https://github.com/alxmrs/xarray-sql/issues/85).
- [ ] Distributed beyond a single node through the DataFusion integration with Ray Datasets [#68](https://github.com/alxmrs/xarray-sql/issues/68) or Apache Ballista [#98](https://github.com/alxmrs/xarray-sql/issues/98).
- [ ] Demo: calculate Sea Surface Temperature from 1940 - Present in SQL [#36](https://github.com/alxmrs/xarray-sql/issues/36).
- [ ] Provide an option to integrate DataFusion directly to Zarr via Rust [#4](https://github.com/alxmrs/xarray-sql/issues/4).
- [ ] (To be formally announced eventually): The 100 Trillion Row Challenge [#34](https://github.com/alxmrs/xarray-sql/issues/34).

## Sponsors & Contributors

I want to give a special thanks to the following folks and institutions:

- Pramod Gupta and the Anthromet Team at Google Research for the problem
  formation and design inspiration.
- Jake Wall and AI2/Ecoscope for compute resources and key use cases.
- Charles Stern, Stephan Hoyer, Alexander Kmoch, Wei Ji, and Qiusheng Wu
  for the early review and discussion of this project.
- Tom Nichols, Kyle Barron, Tom White, and Maxime Dion for the [Array Working
  Group](https://discourse.pangeo.io/t/new-working-group-for-distributed-array-computing/2734)
  and DataFusion-specific collaboration.
- The gracious volunteer data science students at [UCSD's DS3](https://www.ds3atucsd.com/) org,
  who are working to make this library better.
- Andrew Huang for the sense of taste he brings to the project and consummate code
  changes.
- Aman Kumar for spending a considerable amount of his GSoC internship 
  contributing to this project. 


## License

```
Copyright 2024 Alexander Merose

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

All vendored code has proper license attribution.
