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

The adapter registers a re-scannable Arrow C-stream view over the
Dataset: DuckDB pulls fresh record batches on every scan, so the table
is lazy and can be queried any number of times. This first version does
**no** projection or filter pushdown — every scan streams all columns of
all partitions and DuckDB filters afterwards. Prefer selecting variable
subsets (`xql.register(con, "t", ds[["t2m"]])`) for wide datasets.

`xql.to_dataset` is engine-agnostic: it accepts DuckDB relations,
`pyarrow.Table`/`RecordBatchReader`, or any object implementing the
Arrow PyCapsule stream protocol, and is eager (the result is
materialized once, the right shape for aggregations and filtered
selections).

### Relation to duckdb-zarr

[duckdb-zarr](https://github.com/xqlsystems/duckdb-zarr) reads Zarr
stores natively inside DuckDB, with projection pushdown — for
plain-Zarr sources it is the engine-native path and will beat this
adapter. The adapter's role is complementary: anything xarray can open
(NetCDF, GRIB, Earth Engine via Xee, CF-decoded/virtual datasets,
in-memory arrays), and the round-trip from a DuckDB result back to a
labeled Dataset, which no engine extension provides.

## Adding an engine

An adapter implements one small contract
(`xarray_sql.backends.base.EngineAdapter`): `matches(con)` recognizes
the engine's connection object without importing the engine, and
`register(con, name, ds, chunks=...)` attaches the Dataset as a table.
Arrow C streams are the common wire; pushdown quality is where adapters
differ. The round-trip needs no per-engine work as long as the engine
can hand back Arrow.
