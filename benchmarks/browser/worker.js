// Runs Pyodide, xarray-sql, and DuckDB off the page's main thread.
//
// Messages in:  {id, sql}
// Messages out: {type: "status", message}
//               {type: "ready", info}  |  {type: "failed", error}
//               {type: "result", id, result}  |  {type: "error", id, error}
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

// xarray-sql comes from PyPI unless the page passes a wheel URL, e.g. a
// locally built one: index.html?wheel=./xarray_sql-...-pyemscripten_2026_0_wasm32.whl
const wheel = new URL(import.meta.url).searchParams.get("wheel");
const status = (message) => postMessage({ type: "status", message });

async function start() {
  // zarr's synchronous API, which xarray uses to open the store, blocks on
  // fetch() through JavaScript Promise Integration.
  if (typeof WebAssembly.Suspending !== "function") {
    throw new Error(
      "This demo needs WebAssembly JavaScript Promise Integration (JSPI), " +
        "available in Chrome and Edge 137 or newer.",
    );
  }
  status("Loading Pyodide…");
  const py = await loadPyodide();
  await py.loadPackage("micropip");
  status("Installing xarray-sql, DuckDB, and zarr…");
  for (const module of ["app.py", "chunk_cache.py", "fetch_store.py"]) {
    const response = await fetch(new URL(module, import.meta.url));
    py.FS.writeFile(module, await response.text());
  }
  const micropip = py.pyimport("micropip");
  await micropip.install([wheel ?? "xarray-sql", "duckdb", "zarr"]);
  status("Opening the dataset…");
  const app = py.pyimport("app");
  postMessage({ type: "ready", info: JSON.parse(await app.setup()) });
  return app;
}

const app = start().catch((error) => {
  postMessage({ type: "failed", error: String(error.message ?? error) });
});

onmessage = async ({ data: { id, sql } }) => {
  const loaded = await app;
  if (!loaded) return;
  try {
    postMessage({ type: "result", id, result: JSON.parse(await loaded.run(sql)) });
  } catch (error) {
    // Python errors carry the full traceback; the last line is the message.
    const message = String(error.message ?? error).trim().split("\n");
    postMessage({ type: "error", id, error: message.at(-1), detail: message.join("\n") });
  }
};
