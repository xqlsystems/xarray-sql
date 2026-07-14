# Engines

xarray-sql translates **data, not queries**. It does not own a SQL
dialect, a query IR, or a transpiler: you pick a query engine and write
that engine's native SQL, using that engine's extension ecosystem
(spatial, H3, …) directly. xarray-sql implements the two seams no engine
builds for itself:

1. **register** — a lazy `xarray.Dataset` becomes a table on the
   engine's own connection, streamed as Arrow record batches only while
   a query executes.
2. **round-trip** — the engine's Arrow result plus the source Dataset as
   a *template* becomes a labeled `xr.Dataset` again: attrs, non-dim
   coordinates, and dtypes recovered. *SQL in, array out.*

Everything between the seams — geometry functions, dialects,
optimizers — belongs to the engine.

## DataFusion (default)

DataFusion is the built-in engine, wrapped in a session:

```python
import xarray_sql as xql

ctx = xql.XarrayContext()
ctx.from_dataset("era5", ds, chunks={"time": 24})
result = ctx.sql("SELECT ... FROM era5").to_dataset()
```

This is the deepest integration: the Rust `TableProvider` gives
partition pruning on dimension predicates, projection pushdown to the
storage layer, exact per-partition statistics for the optimizer, and a
lazy chunked round-trip (`to_dataset(chunks=...)`).

The generic entry point dispatches here too: `xql.register(ctx, "era5", ds)`
works on any `datafusion.SessionContext`.


## DuckDB (adapter)

```sh
pip install xarray-sql[duckdb]
```

```python
import duckdb
import xarray_sql as xql

con = duckdb.connect()
xql.register(con, "era5", ds)                      # seam 1

con.sql("INSTALL spatial; LOAD spatial;")           # DuckDB's own shelf
rel = con.sql("""
    SELECT time, lat, lon, AVG(t2m) AS t2m
    FROM era5
    WHERE lat BETWEEN 40 AND 41
    GROUP BY time, lat, lon
""")

out = xql.to_dataset(rel, template=ds)              # seam 2
```

The adapter registers an `XarrayPushdownDataset` — a
`pyarrow.dataset.Dataset` subclass (the same pattern Lance uses), so
DuckDB hands each query's column list and pushed predicate to the
source. The scan then loads only the data variables the query mentions,
prunes chunks whose coordinate ranges cannot satisfy the predicate
(via Arrow's own guarantee simplification — sound for every predicate
shape), and prefetches surviving chunks on a thread pool. The table is
lazy, re-queryable, and a bounding-box query over a billions-of-pixels
raster answers in about a second because only the intersecting chunks
are ever read.

Pushed comparison filters are a correctness contract in DuckDB (it
deletes them from its own plan), so the scanner always applies the
exact expression via pyarrow — pruning is only an optimization on top.
`XarrayArrowStream`, the dependency-light re-scannable C-stream wrapper
without pushdown, remains available as a fallback.

Details that matter in production:

- **Finely partitioned axes** (e.g. hourly-chunked reanalysis time with
  hundreds of thousands of chunks) prune through a two-level shadow:
  a coarse pass over at most 1024 buckets, refined per surviving
  bucket — so pruning cost is bounded regardless of chunk count, and
  refinement is skipped when a predicate matches most of the axis.
- **Tuning** via `xql.register(con, name, ds, batch_size=...,
  prefetch=..., prefetch_bytes=..., coalesce_rows=...)`: `prefetch`
  bounds concurrent chunk loads, `prefetch_bytes` caps estimated bytes
  in flight, `coalesce_rows` merges runs of consecutive surviving
  chunks into single reads, `batch_size` caps rows per Arrow batch.
  See the [performance guide](performance.md#the-memory-contract).
- **Source parallelism matters as much as the adapter's**: rioxarray
  serializes GDAL tile reads behind a lock by default, which caps any
  scan at single-stream speed regardless of `prefetch`. Open rasters
  with `rioxarray.open_rasterio(..., lock=False)` — measured 6× on
  full scans of a 9-billion-pixel cloud GeoTIFF, making remote reads
  as fast as a local copy.


### Relation to duckdb-zarr

[duckdb-zarr](https://github.com/xqlsystems/duckdb-zarr) reads Zarr
stores natively inside DuckDB, with projection pushdown — for
plain-Zarr sources it is the engine-native path and will beat this
adapter. The adapter's role is complementary: anything xarray can open
(NetCDF, GRIB, Earth Engine via Xee, CF-decoded/virtual datasets,
in-memory arrays), and the round-trip from a DuckDB result back to a
labeled Dataset, which no engine extension provides.

## Polars (via the pyarrow dataset protocol)

`xql.arrow_dataset(ds)` returns a real `pyarrow.dataset.Dataset`, so
any engine that consumes that protocol gets the same lazy scan with
projection pushdown and coordinate-range chunk pruning — no adapter
code at all. Polars works today:

```python
import polars as pl
import xarray_sql as xql

lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(ds))
out = (
    lf.filter(pl.col("lat") > 0)
    .group_by("time")
    .agg(pl.col("t2m").mean())
    .collect()
)
xql.to_dataset(out, template=ds)   # polars frames speak Arrow PyCapsule
```

Polars pushes its predicate and column selection into the dataset scan
(verified: a filtered group-by read 1 of 20 chunks and 3 of 5 columns),
and its results round-trip through `xql.to_dataset` unchanged. The
chunked round-trip is fully supported: windows re-execute on Polars'
streaming engine.


## Behaviors and limitations by engine

| | DataFusion | DuckDB | Polars |
|---|---|---|---|
| Register | `XarrayContext` / any `SessionContext` | `xql.register(con, name, ds)` | `pl.scan_pyarrow_dataset(xql.arrow_dataset(ds))` |
| Projection pushdown | yes | yes | yes |
| Chunk pruning on dim predicates | yes | yes | yes |
| Eager round-trip (`xql.to_dataset`) | yes | yes | yes |
| Chunked round-trip (`chunks=`) | re-execution | `spill=True` only [^duckdb-spill] | re-execution (streaming engine) |
| `geometry` column | passes through as annotated WKB | native `GEOMETRY` (`"wkb"` encoding only) [^geometry] | plain binary/struct; no geo types |
| Float `is_in` value sets | exact | exact | upstream precision bug — use `is_between` [^polars-isin] |
| Concurrency | re-executable across threads | one dedicated engine thread per handle [^duckdb-serial] | single-threaded pull; parallelism from the scan's prefetch pool |
| Mixed-dimension datasets | one schema, `name.group` tables | split into `<name>_<dims>` tables | filter `data_vars` before `arrow_dataset` |
| Version floor | bundled (core dependency) | `duckdb >= 1.4` (tested on 1.5) | tested on `polars 1.42` |

[^duckdb-spill]: Re-executing a relation from worker threads deadlocks
    intermittently inside duckdb-python (reproduced on duckdb 1.4–1.5 /
    CPython 3.12); without `spill=` the library raises instead of
    hanging. Details under
    [Round-trip behaviors](limitations.md#round-trip-behaviors).

[^geometry]: DuckDB does not consume GeoArrow-native points; Polars has
    no geo support at all. Pick the encoding per destination — see
    [Geospatial in SQL](geospatial.md#geoarrow-point-geometry-columns).

[^polars-isin]: Polars' translation of `is_in` **float** literals into
    pyarrow expressions can silently drop matching rows (reproducible
    without xarray-sql); integer and timestamp value sets are
    unaffected. The lazy round-trip's own window queries render float
    value lists as ranges internally for this reason.

[^duckdb-serial]: Derived relations share execution state upstream, so
    the round-trip handle serializes all engine calls on one dedicated
    thread; concurrent materialization of derived relations raises
    otherwise.

The general scan behaviors every engine shares — pruning soundness,
`count(*)` cost, NaN and non-numeric dimensions, memory bounds — are
cataloged in [Behaviors & limitations](limitations.md).

## The lazy round-trip across engines

(The decision tree, and every edge of it, is diagrammed in
[Behaviors & limitations](limitations.md#round-trip-behaviors).)

`xql.to_dataset(result, chunks=...)` reconstructs a query result as a
*chunked, lazy* `xr.Dataset`: each output chunk re-executes the engine's
query narrowed to that chunk's coordinate window on first access. Over a
table registered through xarray-sql, the window's range predicate flows
back into chunk pruning at the source — accessing one output chunk reads
only the source chunks it maps onto.

One-shot results (`pyarrow` tables/readers, C-stream objects) are
eager-only unless spilled; engine support for `chunks=` is in the
matrix above.

Two knobs matter at scale:

- `coords="template"` trusts the template's coordinate arrays instead of
  running one `DISTINCT` query per dimension — construction then reads
  nothing at all. Only valid when the result spans the template's full
  extent (an unfiltered scan). On ARCO-ERA5 (1.32M hourly chunks) this
  builds a lazy view over a 1.37-trillion-row table in ~0.3 s with zero
  source reads; a one-day window then computes in ~2 s reading only the
  source chunks under the window.
- Contiguous windows become two-literal range predicates the engine can
  push and the source can prune on; stepped or fancy selections fall
  back to explicit value lists (exact, just less prunable).

With `spill=True`, the result is streamed **once** (bounded memory)
into a temporary Parquet file and windows re-execute against that
file — the right shape when most of the result will be touched, the
only chunked option for one-shot Arrow streams, and the required path
for DuckDB relations (see the DuckDB section above). Polars/DataFusion
re-execution remains the default for window-at-a-time access over huge
results.

## Adding an engine

An adapter implements one small contract
(`xarray_sql.backends.base.EngineAdapter`): `matches(con)` recognizes
the engine's connection object without importing the engine, and
`register(con, name, ds, chunks=...)` attaches the Dataset as a table.
Arrow C streams are the common wire; pushdown quality is where adapters
differ. The round-trip needs no per-engine work as long as the engine
can hand back Arrow.

