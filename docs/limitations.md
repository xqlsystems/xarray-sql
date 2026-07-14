# Known issues & limitations

What does not work, why, and what to do instead. How the machinery
*works* — scan pipeline, tuning, cost model — lives in
[Engines](engines.md) and the [performance guide](performance.md); this
page is only the sharp edges. Everything here is pinned by tests.

## Upstream issues

### DuckDB: re-executing relations from worker threads deadlocks

**Symptom.** A chunked round-trip (`chunks=`) of a DuckDB relation would
hang intermittently (~50% of runs) when dask workers re-execute the
relation, whenever the query scans a Python-backed table.

**Scope.** duckdb-python 1.4–1.5 with CPython 3.12 (observed on macOS);
unaffected by `SET threads=1`, connection-level serialization, or
thread-pool pre-warming. The identical topology through Polars never
hangs. The deadlock is in the interpreter/engine thread-state
interaction, not in xarray-sql.

**What the library does.** `chunks=` on a DuckDB relation raises
`NotImplementedError` immediately rather than hanging. `spill=True`
provides the chunked path without ever re-executing: the result is
streamed once (bounded memory, on the handle's dedicated engine thread)
into a temporary Parquet file that windows re-execute against. The
eager round-trip is unaffected.

### DuckDB: derived relations break under concurrent materialization

Relations derived from the same base share pending-query state
upstream; materializing two concurrently raises
`InvalidInputException`. The round-trip handle therefore serializes
every engine call on one dedicated thread — nothing to do on your side,
documented so the serialization is not mistaken for a missing
optimization.

### Polars: float `is_in` literals lose precision

Polars' translation of `is_in` **float** literals into pyarrow
expressions can silently match nothing (reproducible without
xarray-sql; integer and timestamp value sets are unaffected). Prefer
`is_between` for float coordinates in your own queries. The lazy
round-trip's window queries render float value lists as degenerate
ranges internally, so reconstruction is immune.

## Fundamental constraints

### Geometry predicates alone cannot prune

Engines never push function calls (`ST_Within`, casts, arithmetic) into
a scan — only plain column-vs-constant comparisons, `IN`, `IS NULL`,
and boolean combinations. A geometry-only `WHERE` therefore scans and
encodes every chunk (measured ~29x slower than the paired form on a
10M-row grid). Pair every geometry predicate with range conjuncts on
the coordinate columns; [`bbox_conjuncts`][xarray_sql.bbox_conjuncts]
renders them from the geometry's envelope. See
[Geospatial in SQL](geospatial.md#geoarrow-point-geometry-columns).

### Filters on data variables always scan

Chunk pruning and arithmetic counting rest on per-chunk *coordinate*
ranges. A predicate on a data variable (`t2m > 300`) carries no such
guarantee: every surviving chunk is scanned, and the filter is applied
row-exactly. No configuration changes this; it is what the data model
can prove.

### NaN coordinates disable pruning for their chunks

A NaN/NaT anywhere in a chunk's coordinate span poisons its min/max
guarantee, so that chunk is kept for every predicate. This is the
correct trade: a range that pretended to cover NaN would let engines
whose NaN ordering differs (DuckDB sorts NaN greatest) silently lose
rows. Chunks without NaN prune normally.

### String, object, and cftime dimensions never prune

Chunk guarantees are built for numeric and datetime coordinates only;
predicates on other dimension types conservatively scan every chunk
(row-exactly, as always).

### Scan-path pruning is per-dimension

On the scan path, each dimension's surviving chunks are computed
independently and combined as a product, so a predicate pairing
*specific* ranges across dims — `(t < a AND lat < b) OR (t > c AND
lat > d)` — also reads the cross combinations (sound, conservative;
per-dim indexes are what keep million-chunk axes cheap). `count_rows`
refines the crosses away with cross-dimension bucket analysis; ordinary
scans accept the extra reads.

### Sparse results can explode the dense grid

The eager round-trip reconstructs the coordinate-product grid: a
diagonal of n rows becomes an n×n array that can dwarf its Arrow
payload. `max_result_bytes=` raises cleanly at both danger points
(stream collection and dense allocation); it is opt-in and unlimited
by default.

### One-shot Arrow streams cannot re-execute

A materialized table or bare C-stream has no query behind it, so the
re-execution form of `chunks=` cannot serve it — `spill=True` (one
pass to a temporary Parquet file) is the chunked path for these.

### Mixed-dimension datasets split into one DuckDB table per dim group

DuckDB registration has no schema namespace, so variables with
different dims land in suffixed tables (`<name>_<dims>`), sharing one
set of coordinate reads. DataFusion registers the same layout as
`name.group` tables inside one schema.

### Pointwise indexers on lazy round-trip arrays are slower

Vectorized (pointwise) selection goes through xarray's
outer-then-gather fallback — correct, but slower than slice windows.
