# Examples

A query result can be consumed two ways: as a flat pandas DataFrame
(`to_pandas`) or written back to an Xarray Dataset (`to_dataset`). This computes
a climatology — the mean annual cycle, one value per month of the year — and
shows both.

> **Note:** this example also needs `pooch` and a netCDF backend (for the
> tutorial download) and `matplotlib` (for the plot):
> `pip install pooch netCDF4 matplotlib`.

```python
import xarray as xr
import xarray_sql as xql

ds = xr.tutorial.open_dataset('air_temperature')

ctx = xql.XarrayContext()
ctx.from_dataset('air', ds, chunks=dict(time=100))

clim = ctx.sql('''
  SELECT
    CAST(date_part('month', "time") AS INTEGER) AS month,
    AVG("air") AS air
  FROM
    "air"
  GROUP BY
    CAST(date_part('month', "time") AS INTEGER)
  ORDER BY
    month
''')

# Option 1: a flat pandas DataFrame.
clim.to_pandas().head()

# Option 2: round-trip back to an Xarray Dataset and plot the annual cycle as
# a time series. `to_dataset()` infers dimensions from the registered table's
# surviving dims, so a GROUP BY on a real dimension needs no `dims=`. Here
# `month` is a derived column, not a registered dim, so name it explicitly;
# the variable's units are recovered from the registered table.
clim_ds = clim.to_dataset(dims=["month"])
clim_ds["air"].plot()
```

## Mixed-dimension datasets: ARCO-ERA5

When a Dataset has variables with differing dimensions (e.g. surface fields on
`(time, latitude, longitude)` and atmospheric fields on
`(time, level, latitude, longitude)`), `from_dataset` splits them into one
table per dimension group, registered together under a SQL schema named after
the first argument. [ARCO-ERA5][arco-era5] is a good example: 262 of its
variables are surface fields and 11 are atmospheric.

Open a year of ARCO-ERA5 and let SQL `WHERE` clauses do the filtering — the
library prunes time partitions and pushes dimension-column filters down. Use
the `table_names` kwarg to give each dimension group a friendly name:

> **Note:** reading from `gs://` requires `gcsfs` (`pip install gcsfs`).

```python
import xarray as xr
import xarray_sql as xql

# Open ARCO-ERA5 directly from GCS (anonymous read).
url = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full = xr.open_zarr(url, chunks=None, storage_options={'token': 'anon'})

# A full year of hourly ERA5 — all 273 variables. No spatial slicing on the
# xarray side; SQL WHERE clauses below express the filters. `chunks={'time': 1}`
# aligns Dask chunks to native Zarr chunks of shape (1, 37, 721, 1440) so
# chunk reads from GCS happen concurrently.
#
# Heads up: 262 of those variables are surface and 11 are atmospheric. The
# library pushes column projection down, so SELECT only fetches what you ask
# for — but `SELECT * FROM era5.surface` would try to pull every variable
# across the year (terabytes from GCS). Always SELECT specific columns.
ds = full.sel(time='2020').chunk({'time': 1})

ctx = xql.XarrayContext()
ctx.from_dataset('era5', ds, table_names={
    ('time', 'latitude', 'longitude'): 'surface',
    ('time', 'level', 'latitude', 'longitude'): 'atmosphere',
})
# Registers two tables under a SQL schema named 'era5': 'surface' and 'atmosphere'.

# Average 2m-temperature over the NYC area on the morning of 2020-01-01.
ctx.sql('''
  SELECT AVG("2m_temperature") - 273.15 AS avg_c
  FROM era5.surface
  WHERE time BETWEEN TIMESTAMP '2020-01-01'
                 AND TIMESTAMP '2020-01-01 05:00:00'
    AND latitude  BETWEEN 39 AND 40
    AND longitude BETWEEN 286 AND 287
''').to_pandas()

# Average temperature per pressure level, globally — the standard
# atmospheric temperature profile. Scans ~230M rows.
ctx.sql('''
  SELECT level, AVG(temperature) - 273.15 AS avg_c
  FROM era5.atmosphere
  WHERE time BETWEEN TIMESTAMP '2020-01-01'
                 AND TIMESTAMP '2020-01-01 05:00:00'
  GROUP BY level
  ORDER BY level DESC  -- surface (1000 hPa) first
''').to_pandas()
```

If you omit `table_names`, each table is named by joining its dimension names
with underscores, e.g. `era5.time_latitude_longitude` and
`era5.time_level_latitude_longitude`.

## GOES satellite imagery (scalar variables)

Real-world stores often mix gridded data with scalar (0-dimensional) metadata.
GOES satellite imagery, for example, pairs `(y, x)` image bands with dozens of
scalar variables such as `goes_imager_projection`. `from_dataset` groups all the
scalars into a single one-row table named `scalar`:

```python
import fsspec
import xarray as xr
from xarray_sql import XarrayContext

# A real GOES-16 ABI cloud-and-moisture file from NOAA's public bucket:
# (y, x) image bands alongside dozens of scalar metadata variables.
url = (
    'https://noaa-goes16.s3.amazonaws.com/ABI-L2-MCMIPM/2024/001/00/'
    'OR_ABI-L2-MCMIPM1-M6_G16_s20240010000281_e20240010000350_c20240010000426.nc'
)
ds = xr.open_dataset(fsspec.open_local(f'simplecache::{url}')).chunk(
    {'y': 250, 'x': 250}
)

ctx = XarrayContext()
ctx.from_dataset('goes', ds)

# The gridded bands and the scalar metadata are separate tables.
ctx.sql('SELECT COUNT(*) AS n FROM goes.y_x').to_pandas()['n'][0]  # -> 250000
ctx.sql('SELECT * FROM goes.scalar').to_pandas().shape            # -> (1, 89)
```

Override the default name like any other group with `table_names={(): 'metadata'}`.

A runnable version of the ERA5 example lives at
[`perf_tests/era5_temp_profile.py`](../perf_tests/era5_temp_profile.py).

[arco-era5]: https://github.com/google-research/arco-era5
