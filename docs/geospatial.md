# Geospatial operations are relational operations

A working hypothesis, and a slightly radical one: **the core operations of
geospatial and climate analysis — the ones we reach for an array library to
perform — are, underneath, relational operations.** Climatologies, anomalies,
zonal means, spectral indices, forecast skill, even regridding: each maps onto
ordinary SQL — `GROUP BY`, `JOIN`, window functions, `CASE`, and the occasional
scalar UDF.

The array paradigm (NumPy, Xarray, Dask) is a wonderful *interface* for these
operations. But it is not the only one, and for a large and growing audience —
the people fluent in SQL rather than in `apply_ufunc` and rechunking — it is not
the most accessible one. [`xarray-sql`](../README.md) lets you pose these
questions in SQL and answers them with a real query engine (DataFusion). The
datasets are opened *lazily*, so a query against the whole archive reads only the
variable and the slice it actually needs. And because a gridded result is still
gridded data, every query here round-trips its answer straight back to an
`xarray.Dataset` — SQL in, an array out, ready to plot or save.

This page makes the argument case by case. Every claim below is backed by a
runnable script in [`benchmarks/geospatial/`](../benchmarks/geospatial/) that
poses the operation in SQL and **asserts the answer matches an xarray/array
reference** to floating-point tolerance. The point is not that "SQL is faster";
the point is that the SQL reads like the *definition* of the operation and
computes the same numbers — at ERA5's real 0.25° global resolution.

## Where this list comes from

The operations here aren't a set we hand-picked to suit SQL. They're taken from
[**Large Scale Geospatial Benchmarks**](https://github.com/coiled/benchmarks/discussions/1545)
(coiled/benchmarks #1545), a discussion [James Bourbeau](https://github.com/jrbourbeau)
opened in 2024 asking the
geospatial and climate community a pointed question: what are the *end-to-end
workflows* the Xarray/Dask ecosystem needs to handle smoothly at the
100-terabyte scale? The replies are a representative survey of what geoscience
actually runs — and this suite works through nearly all of it:

| #1545 workflow | Covered by |
|----------------|------------|
| Remote-sensing indices (NDVI/NDWI/NDSI over Sentinel-2 or Landsat) | case 01 |
| Vectorized functions (`apply_ufunc`-style per-cell math) | case 01 |
| Climatology (average weather for a time of year/day, per location) | case 02 |
| Transformed Eulerian Mean (circulation diagnostics — zonal means and anomalies) | cases 03, 04 |
| Forecast evaluation (scoring forecasts against ground truth) | case 05 |
| Regridding and reprojection (resolution and CRS changes) | cases 07, 08, 09 |
| Spatial joins (large polygon-to-polygon joins) | *not covered* — a vector-data problem; the closest analogue here is the raster × vector join in case 06 |

So the claim isn't that a few cherry-picked operations happen to be relational.
It's that an independent survey of the operations geoscience runs at scale, run
through SQL one by one, turns out to be — almost entirely — queries.

## The mapping

| Operation | The "array" framing | The relational reality | Script |
|-----------|---------------------|------------------------|--------|
| Spectral index (NDVI) | `apply_ufunc` over a raster | column arithmetic | [`01_ndvi.py`](../benchmarks/geospatial/01_ndvi.py) |
| Climatology | rechunk → grouped reduction | `GROUP BY lat, lon, hour-of-day` | [`02_climatology.py`](../benchmarks/geospatial/02_climatology.py) |
| Zonal mean | reduce over lon/time axes | `GROUP BY lat` | [`03_zonal_mean.py`](../benchmarks/geospatial/03_zonal_mean.py) |
| Anomaly | grouped broadcast-subtract | climatology CTE self-`JOIN` | [`04_anomaly.py`](../benchmarks/geospatial/04_anomaly.py) |
| Forecast skill (RMSE) | align valid/init/lead, reduce | forecast↔truth `JOIN` on `valid_time` | [`05_forecast_skill.py`](../benchmarks/geospatial/05_forecast_skill.py) |
| Zonal stats over regions | rasterize polygons + mask | raster × vector range `JOIN` | [`06_zonal_vector.py`](../benchmarks/geospatial/06_zonal_vector.py) |
| Reprojection | per-pixel CRS transform | scalar **UDF** (`ST_Transform`-style) | [`07_reproject_udf.py`](../benchmarks/geospatial/07_reproject_udf.py) |
| Regridding | interpolation to a new grid | sparse-weight table `JOIN` | [`08_regrid_weights.py`](../benchmarks/geospatial/08_regrid_weights.py) |
| Warp (reproject + resample) | CRS transform *and* interpolation | reproject **UDF** → weight-table `JOIN` | [`09_warp.py`](../benchmarks/geospatial/09_warp.py) |

## 1. A pixel-wise formula is a column expression

NDVI is `(NIR − Red) / (NIR + Red)`, per pixel. The array idiom broadcasts a
ufunc over the raster. But "one output per pixel, computed from that pixel's
bands" is the definition of a SQL projection:

```sql
SELECT x, y, (nir - red) / (nir + red) AS ndvi
FROM scene
ORDER BY y, x
```

Invalid pixels need no special handling: xarray decodes the band's `_FillValue`
to `NaN` on open, and `NaN` propagates through the arithmetic on both sides, so
the masking is free.

[`01_ndvi.py`](../benchmarks/geospatial/01_ndvi.py) runs this against a **real
Sentinel-2 L2A scene in Zarr** — discovered with `pystac-client` and opened the
canonical way with `xr.open_datatree` (ESA's EOPF sample service) — and matches
xarray's `apply_ufunc`-style result over a million pixels.

## 2. A climatology is a `GROUP BY` over the cycle

A climatology is the average value for each time-of-cycle at each location. In
the array world this is the canonical painful workload — load native chunks,
*rechunk* so all of time lands in one chunk, reduce, rechunk back. The
rechunking serves the array layout, not the question. The question is:

```sql
SELECT latitude, longitude, date_part('hour', time) AS hour,
       AVG("2m_temperature")
FROM era5 GROUP BY latitude, longitude, date_part('hour', time)
```

The grouping keys are the dimensions you keep; everything else is reduced. No
layout to reason about. [`02_climatology.py`](../benchmarks/geospatial/02_climatology.py)
computes the **diurnal cycle** of ERA5 2m-temperature over a region — averaging
each cell by hour of day — and matches `da.groupby("time.hour").mean()` across
~500k cells.

A **zonal mean** ([`03_zonal_mean.py`](../benchmarks/geospatial/03_zonal_mean.py))
is the same idea with fewer keys: the axes you "reduce over" are simply the
columns you don't `GROUP BY`.

## 3. Broadcasting a normal back onto observations is a `JOIN`

An anomaly subtracts each cell's climatological normal from every matching
observation. Xarray expresses the realignment with grouped broadcasting
(`ds.groupby("time.hour") - climatology`). That realignment — *attach each
cell's normal to every timestep that shares its key* — is a JOIN on the
grouping key:

```sql
WITH clim AS (
  SELECT latitude, longitude, date_part('hour', time) AS hour,
         AVG("2m_temperature") AS clim_t
  FROM era5 GROUP BY latitude, longitude, date_part('hour', time)
)
SELECT a.time, a.latitude, a.longitude,
       a."2m_temperature" - c.clim_t AS anomaly
FROM era5 a JOIN clim c
  ON a.latitude = c.latitude AND a.longitude = c.longitude
 AND date_part('hour', a.time) = c.hour
```

[`04_anomaly.py`](../benchmarks/geospatial/04_anomaly.py) computes the
climatology once (the CTE) and joins it back to every observation.

## 4. Forecast evaluation is a `JOIN` on valid time + aggregate

This is the real workload of [WeatherBench 2](https://weatherbench2.readthedocs.io/):
scoring machine-learning weather models — **Pangu-Weather** and **GraphCast** —
against ERA5 ground truth. A forecast is indexed by *initialization time* and
*lead time* (`prediction_timedelta`); the truth is indexed by *valid time*.
Evaluation aligns them by `valid_time = init + lead` and reduces the error to
RMSE as a function of lead.

That alignment is a relational JOIN, and `valid_time = init + lead` is just
timestamp + duration arithmetic the engine does natively:

```sql
SELECT f.model, f.prediction_timedelta AS lead,
       SQRT(AVG(POWER(f."2m_temperature" - e."2m_temperature", 2))) AS rmse
FROM forecasts f
JOIN era5 e
  ON  e.time = f.time + f.prediction_timedelta   -- valid_time = init + lead
  AND e.latitude  = f.latitude
  AND e.longitude = f.longitude
GROUP BY f.model, f.prediction_timedelta
```

Both models are stacked along a `model` dimension into one forecast table, so a
single query scores them together, grouped by the `model` column. The entire
evaluation — temporal alignment across three time axes, spatial matching, and the
score — is one JOIN and one aggregate.
[`05_forecast_skill.py`](../benchmarks/geospatial/05_forecast_skill.py) runs it
for both models, matches an xarray reference, and reproduces the published result
that GraphCast edges out Pangu at every lead — the classic "error grows with
horizon" curve (≈0.3 K at 6 h rising to ≈2.5 K at 9 days):

The result round-trips to a `pandas` table directly (`got.to_pandas()`), RMSE in
kelvin by lead time:

```
model        graphcast  pangu
lead (days)
0.25             0.296  0.336
1.25             0.464  0.554
2.25             0.608  0.734
3.25             0.780  0.936
4.25             0.988  1.191
5.25             1.228  1.469
6.25             1.470  1.747
7.25             1.763  2.096
8.25             2.092  2.489
9.25             2.380  2.814
```

## 5. Raster × vector zonal statistics is a range `JOIN`

"Average the raster inside each region" is the canonical raster-meets-vector
task. The array idiom rasterizes each polygon to a mask and reduces under it. But
a region is a row in a table of bounds, and "pixel inside region" is a range
predicate — so zonal statistics is a JOIN:

```sql
SELECT r.region, AVG(a."2m_temperature") - 273.15 AS avg_c
FROM era5.surface a JOIN regions r
  ON  a.latitude  BETWEEN r.lat_min AND r.lat_max
  AND a.longitude BETWEEN r.lon_min AND r.lon_max
WHERE a.time BETWEEN TIMESTAMP '2020-06-01' AND TIMESTAMP '2020-06-01 23:00:00'
GROUP BY r.region
```

This is the README's promise — *joining tabular data with raster data* — made
literal: the raster is the full ERA5 archive (the `WHERE` prunes it to a day),
the regions are a second SQL table, and the spatial relationship is an ordinary
`BETWEEN`. See [`06_zonal_vector.py`](../benchmarks/geospatial/06_zonal_vector.py)
— it reports e.g. Sahara 33 °C vs Greenland −8 °C for a June day. (Rectangular
regions keep this simple; arbitrary polygons would follow the same shape, with a
point-in-polygon test in the join.)

## 6. The hard cases: where a UDF fits, and where it doesn't

Reprojection and regridding are the operations most wedded to the array
paradigm. They split cleanly along one line: **is the operation row-independent?**

**Reprojection is.** Moving a coordinate from one CRS to another depends only on
that coordinate, so it is a *scalar function* — exactly what PostGIS and
DuckDB-spatial already ship as `ST_Transform`. We register a PROJ-backed scalar
UDF (mirroring the `cftime()` UDF already in `xarray_sql/cftime.py`) and
reproject in SQL:

```sql
SELECT x, y, reproject(x, y)['lon'] AS lon, reproject(x, y)['lat'] AS lat
FROM grid
```

[`07_reproject_udf.py`](../benchmarks/geospatial/07_reproject_udf.py) validates
this against **Earth Engine itself**: it opens a UTM grid through
[Xee](https://github.com/google/Xee) carrying `ee.Image.pixelLonLat()`, so EE's
own geodesy engine reports the true lon/lat of every pixel — an *independent*
reprojection reference, not PROJ-vs-PROJ. The SQL UDF and EE agree to sub-metre
precision. The script flags one practical gotcha (PROJ is not thread-safe, so the
UDF runs serially), but the caveat that matters here is conceptual: reprojection
moves the coordinates without resampling the data onto a new grid — and *that* is
the next operation.

**Regridding is not** row-independent: each output cell is a weighted blend of
several input cells. That is a *many-to-many* relationship — and a many-to-many
weighted blend is a sparse matrix–vector product, which is a `JOIN` against a
weight table plus a weighted `GROUP BY`:

```sql
SELECT w.dst_id, SUM(s.value * w.weight) AS regridded
FROM weights w JOIN src s ON s.cell_id = w.src_id
GROUP BY w.dst_id
```

[`08_regrid_weights.py`](../benchmarks/geospatial/08_regrid_weights.py) regrids
real **SRTM elevation** (Sierra Nevada terrain, opened from the Earth Engine
catalog through [Xee](https://github.com/google/Xee)) coarse → fine and matches
xarray's bilinear `.interp()` exactly. So regridding does not weaken the thesis —
it is the most relational operation of all.

**A warp is just the two composed.** The full operation a GIS calls *warp* (GDAL
and rasterio's `reproject`) does both at once: change the CRS *and* resample onto
the new grid. [`09_warp.py`](../benchmarks/geospatial/09_warp.py) writes it as the
two cases above run back to back — the 07 reproject UDF carries the target
lon/lat grid back into the source UTM space, arrays turn those reprojected points
into bilinear weights, and the 08 `JOIN` applies them:

```sql
-- 1. reproject the target grid into source coordinates (the 07 UDF)
SELECT dst_lat, dst_lon, reproject(dst_lon, dst_lat)['x'] AS sx,
                         reproject(dst_lon, dst_lat)['y'] AS sy
FROM target
-- 2. apply the bilinear weights built from those points (the 08 JOIN)
SELECT w.dst_lat AS lat, w.dst_lon AS lon, SUM(s.value * w.weight) AS warped
FROM weights w JOIN src s ON s.x = w.src_x AND s.y = w.src_y
GROUP BY w.dst_lat, w.dst_lon
```

It warps SRTM from a UTM grid onto a lon/lat grid and matches xarray's `.interp()`
at the reprojected points exactly, with Earth Engine's own lon/lat SRTM as a
second, cross-CRS sanity check (a loose match — EE resamples its native 30 m data,
we resample the 2 km source — so it is a corroboration, not the assertion). The
warp lands exactly where the split predicts: the row-independent half is a UDF,
the many-to-many half is a `JOIN`, and the only genuinely geometric step — turning
the reprojected points into weights — is the array work the next section is about.

## GeoArrow point-geometry columns

`register(..., geometry=("x", "y"))` derives a `geometry` point column
from two coordinate dims. With the default `"wkb"` encoding DuckDB
(spatial loaded) ingests it as a native `GEOMETRY` with the CRS
attached, so geometry predicates need no `ST_Point(x, y)` construction:

```sql
SELECT avg(risk) FROM eri
WHERE y BETWEEN -29 AND -28 AND x BETWEEN -58 AND -57   -- prunes chunks
  AND ST_Within(geometry, ST_GeomFromText('POLYGON (...)'))  -- refines
```

**Always pair geometry predicates with bbox conjuncts on the coordinate
columns.** Engines do not push functions like `ST_Within` into the
scan, so a geometry-only predicate scans (and encodes) every chunk —
measured ~29x slower than the paired form on a 10M-row grid, where the
bbox prunes first and the exact polygon test is nearly free.
`xql.bbox_conjuncts(geom, x=..., y=...)` renders the conjuncts from any
geometry's envelope (shapely objects or plain
`(xmin, ymin, xmax, ymax)` tuples), with `pad=` for
`ST_DWithin`-style margins — so the idiom is one f-string. The full
reasoning lives in [Behaviors & limitations](limitations.md).

`geometry_encoding="point"` emits GeoArrow-native separated coordinates
instead (the struct children *are* the coordinate arrays): zero-parse
for GeoPandas 1.x (`GeoDataFrame.from_arrow`), lonboard, geoarrow-rs
and SedonaDB. DuckDB does not consume this encoding — pick per
destination. The CRS tag defaults to `OGC:CRS84`; pass
`geometry_crs=...` for anything else.
## Where the array paradigm still earns its keep

The boundary is **weight generation**. Applying a regridding is a join;
*computing* the weights — cell overlaps for conservative remapping, stencils and
spherical geometry for bilinear, the whole machinery of xESMF/ESMF — is genuinely
geometric work that arrays (and specialized libraries) do well. The relational
view does not replace that; it consumes its output. The division of labor is
clean and, we think, the right one:

> **Arrays compute the geometry (the weights). SQL applies it (the join).**

Likewise, the array libraries remain the right tool for building the inputs in
the first place — opening Zarr, decoding CF metadata, the numerics of generating
a weight matrix. `xarray-sql` sits downstream of all that as a query front-end:
once the data is openable as an `xarray.Dataset`, these everyday operations are
expressible — and accessible — as SQL.

That is the qualitative boundary; the rest of this page puts numbers to it. The
**Results** below report what each operation costs in SQL versus the array
reference, **Analysis** explains *why* the relational form is slower and where the
time goes, and the **Conclusion** turns the whole thing into a when-to-use-which.

## Running the suite

```shell
python benchmarks/geospatial/02_climatology.py   # inside the repo
uv run benchmarks/geospatial/02_climatology.py   # standalone (PEP 723 deps)
```

Each script prints its SQL, runs the array reference, and asserts the two agree.
See [`benchmarks/geospatial/README.md`](../benchmarks/geospatial/README.md) for
the full list and dataset notes.

## Results

Correctness is the headline, but every case is also profiled. The numbers below
come from [`run_perf.sh`](../benchmarks/geospatial/run_perf.sh) on a single Google
Compute Engine `e2-standard-8` (8 vCPU, 32 GB) in `us-central1` — in-region with the
ARCO-ERA5 and WeatherBench 2 buckets, so the cloud read is fast — with Earth Engine
reached from the same VM, so all nine cases share one machine and one release build.
Each case runs **once per fresh process**, with no warmup, repeated five times: the
SQL operation *and* the xarray reference each pay a **cold** read on every
measurement.

Fairness here took some care, because the obvious trap is caching. A reference
that calls `.load()` caches its data *in place* on the very object the SQL table
also reads from, so a later read — even just running the reference after the SQL
query in the same process — could be served warm. We close that two ways. The one
case that loads shared objects (05, forecast skill) uses `.compute()` instead,
which returns a fresh array and leaves the inputs lazy, caching nothing; the other
references either reopen their data or recompute their reduction eagerly on every
read (`chunks=None` is NumPy, not Dask, so there is no graph to keep warm). And
`run_perf.sh` runs each case in a fresh process per repetition, ruling out any
carryover between reps. We verified the result directly: reading a window
repeatedly in one process stays flat, and running either side after the other
speeds up neither — the SQL query and the reference do not warm each other.

| Case | Step | median (s) | stdev (s) | min (s) | max (s) | peak (MB) |
|---|---|--:|--:|--:|--:|--:|
| 01 · NDVI (per-pixel arithmetic) | SQL | 3.528 | 0.803 | 2.861 | 5.024 | 114.0 |
|  | xarray reference | 0.304 | 0.104 | 0.282 | 0.496 | 42.0 |
| 02 · Climatology (`GROUP BY` lat, lon, hour) | SQL | 4.443 | 0.383 | 4.216 | 5.198 | 490.2 |
|  | xarray reference | 1.867 | 0.106 | 1.844 | 2.053 | 43.7 |
| 03 · Zonal mean (`GROUP BY` latitude) | SQL | 2.406 | 0.122 | 2.333 | 2.631 | 236.9 |
|  | xarray reference | 0.385 | 0.006 | 0.381 | 0.395 | 249.5 |
| 04 · Anomaly (climatology self-`JOIN`) | SQL | 7.027 | 0.123 | 6.950 | 7.239 | 511.5 |
|  | xarray reference | 2.549 | 0.219 | 2.126 | 2.657 | 72.1 |
| 05 · Forecast skill (forecast↔truth `JOIN`) | SQL | 10.714 | 0.093 | 10.663 | 10.891 | 6.6 |
|  | xarray reference | 0.248 | 0.013 | 0.220 | 0.254 | 2.2 |
| 06 · Zonal stats (raster × vector `JOIN`) | SQL | 4.308 | 0.053 | 4.299 | 4.401 | 509.1 |
|  | xarray reference | 1.557 | 0.029 | 1.499 | 1.567 | 1262.1 |
| 07 · Reprojection (PROJ scalar UDF) | SQL | 0.029 | 0.003 | 0.024 | 0.031 | 0.3 |
| 08 · Regridding (weight-table `JOIN`) | SQL | 0.875 | 0.037 | 0.845 | 0.933 | 11.9 |
|  | xarray reference | 0.850 | 0.658 | 0.809 | 2.310 | 13.3 |
| 09 · Warp (reproject UDF → regrid `JOIN`) | SQL | 0.281 | 0.038 | 0.250 | 0.353 | 0.8 |
|  | xarray reference | 0.817 | 0.030 | 0.764 | 0.828 | 11.2 |

Two patterns are visible before any analysis. SQL is slower on wall-clock wherever
a cloud read or a large relational expansion dominates — by ~2.5–6× on the
`GROUP BY` and `JOIN` cases against ARCO-ERA5, and ~43× on case 05, the smallest
grid but the biggest blow-up into rows — and its peak memory is highest on those
join/group-by cases (≈0.5 GB on 02, 04, 06). But the pattern is **not** universal.
On cases 08 and 09, where the interpolation *weights* are precomputed and SQL just
applies them, SQL is at parity with the array reference (08: 0.875 vs 0.850 s) or
**faster** (09: 0.281 vs 0.817 s — the reference pays for `pyproj` + `.interp`,
while SQL streams the prebuilt weight `JOIN`). The slow and the fast cases follow
from the same cause, which the next section pins down. (Case 01 reads Sentinel-2
from Europe, the only non-US source, so its SQL time includes a cross-region read.
Cases 07–09 run against Earth Engine from the same VM: 07 times only the SQL
reproject transform, checked against Earth Engine's own `pixelLonLat`; 08 and 09
read SRTM lazily on **both** the SQL and reference sides, so that comparison is
symmetric.)

Case 05 is the suite's most hardware-sensitive number: its SQL time is CPU-bound on
the join and the (GIL-held) row production that feeds it, so it swings with the
machine — across three `e2-standard-8` runs it has measured ≈10.7 s, ≈12 s, and
≈23 s, while the read-bound *reference* stays near 0.25 s. So read the 05 ratio as
"the relational form costs real CPU here," not as a fixed multiplier.

## Analysis: how a relational operation spends its time

Why is SQL slower, and where does the time actually go? Profiling case 05 — the
forecast-skill `JOIN`, the widest gap — with `cProfile`, run cold then warm so that
`cold − warm` isolates the cloud read and the warm floor is ≈pure compute,
decomposes it cleanly. (These are single-process numbers from a laptop with a slow
cross-region read — a *different* machine from the in-region table above, on
purpose: it puts both sides' reads on equal, slow footing so the compute gap shows
through. The absolute seconds therefore differ from the table; the decomposition,
not the totals, is the point.)

| | read (I/O) | compute | total (cold) |
|---|--:|--:|--:|
| SQL | ~0.95 s | **~0.71 s** | ~1.66 s |
| xarray reference | ~0.79 s | **~0.024 s** | ~0.81 s |

The read is comparable on both sides — both open the same Zarr store cold. **The
gap is compute, and it is about 30×.** The SQL path explodes the 64×32×20×2 grid
into Arrow rows, runs a hash `JOIN` to align each forecast row with its truth row
on `(valid_time, latitude, longitude)`, aggregates, and streams the result batches
back. The array reference does the identical math as a handful of vectorized NumPy
reductions over contiguous buffers. Row materialization + hashing + the join probe
is simply heavier than dense arithmetic on a regular grid — and it is the same
work that inflates SQL's peak memory in the Results table: the join and group-by
cases hold the grid as rows.

`cProfile` is unambiguous about *where* the SQL time sits. Essentially all of it is
in pulling record batches from the DataFusion execution stream; the SQL→xarray
round-trip that turns the query result back into a gridded `Dataset`
(`to_dataset`) is **sub-millisecond — under 1% of the query.** So the cost is the
relational engine doing row-oriented work, not the array reconstruction. The
paradigm itself is the price, paid where the relational algebra runs.

This explains the shape of the whole table. Case 05 stands alone at ~43× not
because its join is exotic but because its *reference* is nearly free — a 64×32×20×2
grid reduces in-memory in a quarter-second — while SQL still has to explode that
grid into rows and hash-join them; a huge ratio over a tiny denominator. The
ARCO-ERA5 cases (02, 03, 04, 06) instead cluster at ~2.5–6×, because there a large
cloud read is a cost *both* sides pay, compressing the ratio. And cases 08 and 09
invert it entirely: once the geometry — the interpolation weights — is precomputed,
applying it is a `JOIN` that streams about as fast as (or faster than) the array
reference's `pyproj`/`.interp`. The relational *overhead* is constant; the *ratio*
you observe depends on how much non-relational work (the cloud read, the weight
generation) sits on the other side of the comparison. And it shifts with hardware
too: SQL is CPU-bound on the join while the array reference is read-bound, so the
two are gated by different resources. On a fast laptop with a slow cross-region
read the gap nearly closes; on an in-region VM with modest cores it widens. The
underlying cause is
constant — materialize rows, hash-join, aggregate — but which resource you are
waiting on is not.

## Conclusion

None of this is an argument that SQL is *faster*. On a single node, for the
reduction-shaped operations, it is not — it pays a real per-operation overhead to
express an array reduction as relational algebra. (The exceptions, cases 08 and 09,
prove the rule: once the array work — generating the weights — is already done, the
relational half that remains is competitive, because there is no dense reduction
left for arrays to win.) The honest tradeoff is about which property you are
optimizing for.

**Reach for the array paradigm when the work is dense and grid-aligned.** Per-pixel
formulas, stencils, convolutions, FFTs, linear algebra — anything that stays in
contiguous typed buffers and treats the chunk grid as its unit of parallelism. The
array model has the lowest overhead here, and the lead is structural, not
incidental: there are no rows to materialize and nothing to shuffle. NDVI (case 01)
is the tell — column arithmetic expresses cleanly in SQL, but the array side is
~10× faster (part of which is case 01's cross-region read; the rest is that
per-pixel math is exactly what arrays are for).

**Reach for SQL when the work is relationally shaped, or the audience is.** Joins,
group-bys, alignment across data with different indexes (case 05's three time
axes), raster-meets-vector predicates (case 06) — these are awkward to express and
to reason about as array operations, and they are the native vocabulary of a query
engine. The overhead buys you an operation that reads like its own definition, that
prunes its own reads (a query against the whole ERA5 archive touches only the
variable and window it asks for), and that is accessible to the large audience
fluent in SQL rather than in `apply_ufunc` and rechunking.

There is also a payoff this single-node benchmark cannot show. The same overhead —
row materialization and a hash join — is what makes the operation a *first-class
citizen of a distributed query engine.* Cost-based query optimization (join
reordering, choosing broadcast vs. shuffle joins, predicate pushdown), mature
partitioned shuffle and spill-to-disk, partitioning driven by the query rather than
locked to a physical chunk grid — these are exactly the capabilities the
array/Dask ecosystem struggles to provide for join- and group-by-heavy workloads
at scale, and exactly what the relational framing puts within reach. Whether the
constant-factor overhead is worth paying flips as the data grows and the bottleneck
moves from per-element compute to data movement. `xarray-sql` is single-node today,
so that is a direction rather than a result — but it is the latent reason the
thesis matters beyond expressibility.

So the division of labor from the section above generalizes past regridding. Arrays
own the dense numerics and the geometry; SQL owns the relational shape — the joins,
the alignment, the aggregation — and, increasingly, the path to running them at
scale. The point of this suite is not to crown a winner but to show that the line
between the two is exactly where the operation is dense versus where it is
relational, and that for a surprising share of geoscience, the operation is
relational.

