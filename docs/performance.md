# Performance guide

How to get engine-limited speed out of registered xarray tables. Every
number below was measured on real cloud rasters (billions of pixels);
your mileage scales with network and core count, but the *ratios* are
structural.

## How a scan decides what to read

Every engine query over a registered table flows through one pipeline;
each tuning knob on this page acts on one of its stages:

```mermaid
flowchart TB
    Q["engine calls scanner(columns, filter)"] --> P["prune chunks<br/>per-dim coordinate ranges +<br/>Arrow guarantee simplification"]
    Q --> J["project<br/>only referenced variables are read"]
    P --> C["coalesce (opt-in)<br/>merge consecutive surviving chunks<br/>into single reads"]
    C --> F["prefetch pool<br/>bounded by prefetch (threads)<br/>and prefetch_bytes (memory)"]
    J --> F
    F --> X["exact filter<br/>the pushed expression is applied<br/>row-exactly — pruning is only<br/>ever an optimization"]
    X --> B["Arrow batches → engine"]
```

Two invariants hold everywhere: pruning never decides correctness (the
exact expression is always applied — engines delete pushed conjuncts
from their own plans), and only what reaches `scanner()` can prune
(engines push plain comparisons, never function calls).

## Make the source read in parallel

The single biggest lever is usually the reader, not the engine.

**GeoTIFF / rioxarray**: `rioxarray.open_rasterio` serializes GDAL tile
reads behind a lock by default, capping every scan at single-stream
speed no matter how many threads the adapter runs. On GDAL ≥ 3.11 use
the natively thread-safe LIBERTIFF driver; on older GDAL pass
`lock=False`:

```python
da = rioxarray.open_rasterio(
    url, chunks={"x": 2048, "y": 2048},
    driver="LIBERTIFF",   # GDAL >= 3.11; else keep lock=False only
    lock=False,
)
```

Measured on a 9-billion-pixel public cloud GeoTIFF, full-table
aggregation: default open **277 s** → `lock=False` **43 s** →
LIBERTIFF + `GDAL_NUM_THREADS=ALL_CPUS` **24 s**. With parallel reads,
remote (`/vsicurl/`) matched a local copy of the same file — the
network was never the bottleneck, the lock was.

Remote-read environment preset worth exporting for `/vsicurl/` sources:

```python
os.environ.update(
    GDAL_NUM_THREADS="ALL_CPUS",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    VSI_CACHE="TRUE",
)
```

**Zarr**: zarr-python 3's async store defaults to only 10 concurrent
requests; raise it before opening remote stores:

```python
zarr.config.set({"async.concurrency": 128})
```

On a moderately sized windowed query (~40 chunks of 4096² uint8 per
variable, GCS) this was a modest gain (4.2 s → 3.7 s); it matters more
as chunk counts grow and chunks shrink. The obstore-backed
`zarr.storage.ObjectStore` is worth benchmarking for high-concurrency
workloads, but was not faster at this scale in our tests — measure
before switching.

## Choose chunk sizes for the scan, not just the store

Every chunk costs one prefetch task, one pivot call, and one shadow
fragment. Aim for **1–8 M rows per chunk** (e.g. 2048²–4096² pixels for
2-D grids). The same 10 M-row scan ran 1.7× faster in 4 chunks than in
20. Axes with hundreds of thousands of chunks still prune in
milliseconds (the shadow index is bucketed), but scanning them pays
per-chunk overhead.

## Tune the adapter knobs

```python
xql.register(con, "t", ds, prefetch=12, batch_size=262_144)
```

- `prefetch`: chunk loads kept in flight ahead of the engine. The
  default (4) saturates local CPU work; raise to 8–12 for remote
  sources where latency dominates. Memory scales with
  `prefetch × pivoted chunk size`.
- `batch_size`: rows per Arrow batch. The default (64 Ki) is fine;
  values between 64 Ki and 1 Mi measured within a few percent of each
  other.

## The memory contract

Peak scan memory is bounded by `prefetch × pivoted-block-size` plus the
engine's own aggregation state — it does not grow with the amount of
data scanned. Measured on ARCO-ERA5 over anonymous GCS: a one-month
full-globe aggregation (772M rows) peaks at the same RSS as the
one-week scan (174M rows), ~0.75 GB with the defaults.

`prefetch_bytes` caps *estimated bytes* in flight instead of block
count — set it when `coalesce_rows` makes blocks large or ragged.
The block size is the source chunk size unless `coalesce_rows` is set,
in which case in-flight units are merged blocks: raising
`coalesce_rows` buys fewer round-trips at proportionally higher peak
memory (`prefetch=16, coalesce_rows=8_000_000` peaked at ~1.2 GB on the
same scan while cutting wall time ~1.5-2x). Size the two together.

`count(*)`-shaped queries never pay scan memory at all: unfiltered
counts are pure chunk arithmetic, and filtered counts scan only the
boundary chunks the filter cannot prove — at any filter breadth; see
[What counting costs](#what-counting-costs).
## Let pushdown do its job

Selective queries are fast *because of their predicates*: bounding-box
`WHERE` clauses on dimension columns prune to intersecting chunks, and
only the variables a query references are read. Corollaries:

- Prefer explicit column lists over `SELECT *` on wide datasets.
- Spatial functions (`ST_Within`, ...) are not pushed down — pair them
  with a bounding-box predicate that is: the box prunes, the geometry
  refines.
- A query with no `WHERE` on dimension columns is a full scan on any
  engine; that's physics, not a missing optimization.

## Threads and DuckDB connections

Registered Python objects are connection-local in DuckDB: `con.cursor()`
does not inherit them, and one connection's result slot is not
thread-safe. For multithreaded querying, give each thread its own
cursor and register the *same* dataset object on it:

```python
dataset = xql.arrow_dataset(ds)
def worker():
    cur = con.cursor()
    cur.register("t", dataset)   # cheap; shares the pruning index
    ...
```

The dataset object itself is safe to share across threads (verified
under concurrent query load).

## What counting costs

`count(*)` never pays scan memory, and usually no I/O either:

```mermaid
flowchart TB
    C["count_rows(filter)"] --> U{"filter?"}
    U -- none --> A["pure arithmetic<br/>0 reads"]
    U -- "coordinate ranges" --> H["hierarchical strictness:<br/>bucket-products proven or pruned<br/>whole; only mixed cells recurse"]
    H --> E["boundary chunks scanned exactly<br/>(usually 0-2 per range edge,<br/>at any axis size)"]
    U -- "data variables" --> S["every surviving chunk scanned<br/>(values carry no coordinate<br/>guarantee — see Known issues)"]
```

Coordinate-range counts stay arithmetic at any breadth (a
near-universal filter over a million single-row chunks counts with
zero reads), and the strictness pass applies cross-dimension
information, so paired-range predicates count without reading the
cross combinations.

## Stop re-scanning: cache or pyramid

Registered tables are virtual — every query re-streams the source.
Statistics you ask repeatedly should pay the scan once. Two patterns,
for two question shapes:

### Cache one derived table (plain SQL)

There is no helper for this because none is needed: create a native
table from your query, sorted by the coordinate columns so the engine's
storage compresses the repetitive coordinates (DuckDB picks ALP/RLE on
sorted runs) and zone maps prune range predicates.

```sql
CREATE OR REPLACE TABLE grid_cube AS
SELECT FLOOR(y) AS lat, FLOOR(x) AS lon, klass, COUNT(*) AS n
FROM grid GROUP BY 1, 2, 3
ORDER BY lat, lon;

SELECT * FROM grid_cube WHERE lat = -32;  -- native speed
```

One engine quirk to know: on DataFusion, DDL is a lazy plan — collect
it or nothing happens:

```python
ctx.sql("CREATE OR REPLACE TABLE grid_cube AS ...").collect()
```

### `pyramid`: one cube, every zoom level

Use it when the *resolution* of the question varies — dashboards,
maps, "country then province then plot" drill-downs. Think raster
overviews / map-tile pyramids, in SQL.

Smallest possible example — a 4x4 grid of pixels valued 1..16 on a
2x2-degree extent, binned into 1-degree cells (`base_cell=1.0`), three
levels:

```python
xql.pyramid(con, "pyr", "grid",
    aggs={"n": ("count", "*"), "total": ("sum", "v")},
    base_cell=1.0, levels=3)
```

The entire resulting table:

```text
 level  x_idx  y_idx  x_bin  y_bin   n  total
     0      0      0    0.0    0.0   4   14.0   ┐ four 1-degree cells,
     0      1      0    1.0    0.0   4   22.0   │ 4 pixels each — the
     0      0      1    0.0    1.0   4   46.0   │ only scan of the
     0      1      1    1.0    1.0   4   54.0   ┘ source
     1      0      0    0.0    0.0  16  136.0   ← the 4 cells, added up
     2      0      0    0.0    0.0  16  136.0   ← same again (extent < cell)
```

Each level doubles the cell size and is computed by *adding up* the
level below — never rescanning the source. That is why aggregates must
be decomposable (`sum`/`count`/`min`/`max`): sums and counts add.
An average is a sum plus a count, divided at query time —
`SELECT total / n FROM pyr WHERE level = 2` gives 8.5, exactly the
mean of pixels 1..16.

The use case, on an eri-shaped grid (16.7M uint8 pixels, ~500 m cells):
the cube builds in one scan into ~365k rows, and then

```sql
-- Country-wide overview map: share of high-risk pixels per ~1.6° cell.
SELECT x_bin, y_bin, high / n AS share
FROM eri_pyr WHERE level = 5;                    -- 286 rows, ~1 ms

-- User zooms into one province: same cube, finer level, range filter.
SELECT x_bin, y_bin, high / n AS share
FROM eri_pyr
WHERE level = 1
  AND x_bin BETWEEN -60 AND -58 AND y_bin BETWEEN -34 AND -32;
```

The overview reads a few hundred pre-aggregated rows instead of
grouping every pixel — ~100x faster than the raw `GROUP BY FLOOR(...)`
even with this source in local memory, and the gap grows with the
source: against a remote 9-billion-pixel raster the raw side is a
minutes-long scan while the pyramid stays at milliseconds.

Practical notes: query the level whose cell size matches what you are
rendering; filter `x_bin`/`y_bin` with ranges (float cell origins), or
join on the exact integer `x_idx`/`y_idx` when combining levels or
cubes. `pyramid` runs identically on DuckDB and DataFusion.

## The round-trip is optimized for grids

`xql.to_dataset` locates rows by arithmetic when an axis is uniformly
spaced (any regular raster or time step, ascending or descending) and
reshapes without any scatter when the result arrives grid-ordered.
Sparse or irregular results fall back to a positional scatter
automatically. If you want the raw sub-array of a registered Dataset
rather than a relational answer, plain `ds.sel(...)` is the direct
path — SQL adds value when the question is relational.

