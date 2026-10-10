# xarray-sql in the browser

A static page that runs SQL over NOAA GFS forecasts from
[dynamical.org](https://dynamical.org)'s public Zarr store. xarray-sql and
DuckDB run inside [Pyodide](https://pyodide.org) in a Web Worker, so there is
no server. Each query downloads only the chunks that its filters on
`init_time`, `latitude`, and `longitude` reach.

| File | Role |
|------|------|
| `index.html` | The page: example queries, SQL editor, results table, and a map for `latitude, longitude, value` results. |
| `worker.js` | Loads Pyodide, installs xarray-sql, DuckDB, and zarr, and runs queries off the main thread. |
| `app.py` | Opens the dataset, registers it on DuckDB as `gfs`, and runs queries. |
| `chunk_cache.py` | Lazy reads that work in the browser (see below). |
| `fetch_store.py` | A read-only Zarr store that uses `fetch` with `Range` requests. |
| `smoke.mjs` | Runs every example query headlessly in Node. |

## Run it

Serve this directory over HTTP and open it in Chrome or Edge 137 or newer.
The demo needs WebAssembly JavaScript Promise Integration (JSPI), which
Safari and iOS browsers do not support yet.

```shell
python -m http.server 8000 --directory benchmarks/browser
```

By default the worker installs xarray-sql from PyPI. To test a local build,
copy the wheel from `pyodide build` (see "Building for WebAssembly" in
`CONTRIBUTING.md`) into this directory and pass it in the URL:

```
http://localhost:8000/?wheel=./xarray_sql-0.5.0-cp310-abi3-pyemscripten_2026_0_wasm32.whl
```

To run the examples without a browser:

```shell
cd benchmarks/browser
npm install --no-save pyodide@314.0.7
node smoke.mjs path/to/xarray_sql-*-pyemscripten_*.whl
```

## How lazy reads work in the browser

Browsers can only read over the network asynchronously. zarr's synchronous
API waits on those reads with Pyodide's JSPI support, but JSPI cannot
suspend while DuckDB or pyarrow is calling back into Python: their C++ frames
sit between the suspension point and Python. So a query cannot read chunks in
the middle of a scan.

`ChunkCache` runs each query against a cache of decoded chunks instead:

1. Run the query. A chunk that is not cached is recorded as missing and
   scanned as fill values, so one run records every chunk that survives
   xarray-sql's chunk pruning.
2. Fetch the missing chunks concurrently with `Dataset.load_async`, then run
   the query again.
3. A run with no missing chunks used only real data, so its result is
   exact.

Most queries need two runs. The stats line under the editor shows the runs,
chunks, requests, and bytes for each query.

## Known limits

- **`lead_time` filters do not prune chunks.** DuckDB does not push `INTERVAL`
  predicates down to Arrow scans. The demo registers only the first lead-time
  chunk (forecast hours 0–104), so every query reads at most that one.
- **Queries that touch the whole grid are slow.** For example,
  `SELECT max(init_time) FROM gfs` builds every coordinate row (about 440
  million) even though it downloads nothing. Use the literal timestamps that
  the example queries use.
- **The cache holds 768 MiB of decoded chunks.** A query that needs more
  fails with a message to narrow its filters.
