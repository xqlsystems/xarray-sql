# Run in the browser

xarray-sql runs in [Pyodide](https://pyodide.org), Python compiled to
WebAssembly. The demo below runs SQL over NOAA GFS forecasts read straight
from [dynamical.org](https://dynamical.org)'s public Zarr store. xarray-sql
and DuckDB run in a Web Worker in your browser, so there is no server. Each
query downloads only the chunks that its filters on `init_time`, `latitude`,
and `longitude` reach.

The demo needs Chrome or Edge 137 or newer. It takes about 15 seconds to
start, most of it downloading and loading Pyodide, pyarrow, DuckDB, and
pandas.

<iframe id="demo" src="../demo/" title="xarray-sql browser demo" loading="lazy"
  style="width: 100%; height: 1150px; border: 1px solid var(--md-default-fg-color--lightest); border-radius: 8px;"></iframe>

[Open the demo in its own page](demo/){ #demo-link target="_blank" }

<script>
  // Pass ?wheel=<url> through to the demo, to try a locally built wheel.
  (() => {
    const wheel = new URLSearchParams(location.search).get("wheel");
    if (!wheel) return;
    for (const [id, attr] of [["demo", "src"], ["demo-link", "href"]]) {
      const el = document.getElementById(id);
      const url = new URL(el.getAttribute(attr), location.href);
      url.searchParams.set("wheel", new URL(wheel, location.href).href);
      el.setAttribute(attr, url.href);
    }
  })();
</script>

## Install in Pyodide

Releases from 0.6.0 on publish a Pyodide wheel to PyPI. The DuckDB engine
adapter works in Pyodide, including the round-trip back to a Dataset with
[to_dataset][xarray_sql.to_dataset]. DataFusion does not yet, because its
Python bindings have no Pyodide build, so [Context][xarray_sql.Context] is
unavailable there.

Copy [`fetch_store.py`](https://github.com/xqlsystems/xarray-sql/tree/main/docs/demo/fetch_store.py)
from the demo to open Zarr stores over HTTP in Pyodide, since fsspec's HTTP
filesystem needs sockets. Then open a window of the dataset, read it into
memory, and query it:

```python
import micropip
await micropip.install(["xarray-sql", "duckdb", "zarr"])

import duckdb
import xarray as xr
import xarray_sql as xql
from fetch_store import FetchStore

store = FetchStore("https://data.dynamical.org/noaa/gfs/forecast/latest.zarr")
ds = (
    xr.open_zarr(store, chunks=None)[["temperature_2m"]]
    .isel(init_time=-1, lead_time=slice(0, 24))
    .sel(latitude=slice(50, 25), longitude=slice(-100, -75))
)
ds = await ds.load_async()

con = duckdb.connect()
xql.register(con, "gfs", ds)
hottest = con.sql("""
    SELECT latitude, longitude, max(temperature_2m) AS t2m_max
    FROM gfs
    GROUP BY latitude, longitude
""")
out = xql.to_dataset(hottest, template=ds)  # dims latitude, longitude
```

The window is read into memory first because a DuckDB scan cannot wait on
the network in Pyodide (see the next section). Pyodide also cannot start
threads, so xarray-sql loads chunks one at a time there. To query a larger
store, read it through a cache like the demo's.

## How lazy reads work in the browser

Browsers can only read over the network asynchronously. zarr's synchronous
API waits on those reads with Pyodide's JavaScript Promise Integration
(JSPI), but JSPI cannot suspend while DuckDB or pyarrow is calling back into
Python: their C++ frames sit between the suspension point and Python. So a
query cannot read chunks in the middle of a scan.

The demo's `ChunkCache` runs each query against a cache of decoded chunks
instead:

1. Run the query. A chunk that is not cached is recorded as missing and
   scanned as fill values, so one run records every chunk that survives
   xarray-sql's chunk pruning.
2. Fetch the missing chunks concurrently with `Dataset.load_async`, then run
   the query again.
3. A run with no missing chunks used only real data, so its result is
   exact.

Most queries need two runs. The stats line under the editor shows the runs,
chunks, requests, and bytes for each query.

The demo's files are in
[`docs/demo/`](https://github.com/xqlsystems/xarray-sql/tree/main/docs/demo):

| File | Role |
|------|------|
| `index.html` | The page: example queries, SQL editor, results table, and an interactive map ([deck.gl](https://deck.gl) over a [MapLibre](https://maplibre.org) basemap) for `latitude, longitude, value` results. |
| `worker.js` | Loads Pyodide, installs xarray-sql, DuckDB, and zarr, and runs queries off the main thread. |
| `app.py` | Opens the dataset, registers it on DuckDB as `gfs`, and runs queries. |
| `chunk_cache.py` | Lazy reads that work in the browser, as described above. |
| `fetch_store.py` | A read-only Zarr store that uses `fetch` with `Range` requests. |
| `smoke.mjs` | Runs every example query headlessly in Node. |

## Run it locally

Serve `docs/demo` over HTTP and open it in Chrome or Edge:

```shell
python -m http.server 8000 --directory docs/demo
```

By default the worker installs xarray-sql from PyPI. To test a local build,
copy the wheel from `pyodide build` (see "Building for WebAssembly" in the
[contributing guide](contributing.md)) into `docs/demo` and pass it in the
URL:

```
http://localhost:8000/?wheel=./xarray_sql-0.6.0-cp310-abi3-pyemscripten_2026_0_wasm32.whl
```

On this docs page, `?wheel=` is passed through to the embedded demo, so a
locally built docs site can use a local wheel too (for example,
`/browser/?wheel=../demo/xarray_sql-...whl` after copying the wheel into the
built `demo/` directory).

To run the examples without a browser:

```shell
cd docs/demo
npm install --no-save pyodide@314.0.7
node smoke.mjs path/to/xarray_sql-*-pyemscripten_*.whl
```

## Known limits

- **`lead_time` filters do not prune chunks.** DuckDB does not push
  `INTERVAL` predicates down to Arrow scans. The demo registers only the
  first lead-time chunk (forecast hours 0–104), so every query reads at most
  that one.
- **Queries that touch the whole grid are slow.** For example,
  `SELECT max(init_time) FROM gfs` builds every coordinate row (about 440
  million) even though it downloads nothing. Use the literal timestamps that
  the example queries use.
- **The cache holds 768 MiB of decoded chunks.** A query that needs more
  fails with a message to narrow its filters.
