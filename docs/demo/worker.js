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
  status("Loading Pyodide and packages…");
  // Everything Pyodide ships is fetched in parallel while the runtime
  // boots; micropip then only resolves xarray-sql from PyPI.
  const modules = Promise.all(
    ["app.py", "chunk_cache.py", "fetch_store.py"].map(async (name) => {
      const response = await fetch(new URL(name, import.meta.url));
      return [name, await response.text()];
    }),
  );
  const py = await loadPyodide({
    packages: ["micropip", "duckdb", "pyarrow", "xarray", "zarr"],
  });
  for (const [name, source] of await modules) py.FS.writeFile(name, source);
  status("Installing xarray-sql…");
  const micropip = py.pyimport("micropip");
  await micropip.install(wheel ?? "xarray-sql");
  status("Opening the dataset…");
  const app = py.pyimport("app");
  postMessage({ type: "ready", info: JSON.parse(await app.setup()) });
  return app;
}

// Python errors carry the full traceback. Show the exception line (some
// messages add hints after it) and keep the rest for the console.
const describe = (error) => {
  const detail = String(error.message ?? error).trim();
  const lines = detail.split("\n");
  const exception = lines.findLast((line) => /^[\w.]*(Error|Exception)\b/.test(line));
  return { error: exception ?? lines.at(-1), detail };
};

const app = start().catch((error) => {
  postMessage({ type: "failed", ...describe(error) });
});

onmessage = async ({ data: { id, sql } }) => {
  const loaded = await app;
  if (!loaded) return;
  try {
    postMessage({ type: "result", id, result: JSON.parse(await loaded.run(sql)) });
  } catch (error) {
    postMessage({ type: "error", id, ...describe(error) });
  }
};
