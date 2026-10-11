// Run the demo's example queries headlessly in Node, against the live dataset.
//
//   npm install pyodide@314.0.7
//   node smoke.mjs path/to/xarray_sql-*-pyemscripten_*.whl
//
// Without a wheel path, xarray-sql is installed from PyPI.
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const wheel = process.argv[2];

const py = await loadPyodide();
await py.loadPackage("micropip");
for (const module of ["app.py", "chunk_cache.py", "fetch_store.py"]) {
  py.FS.writeFile(module, readFileSync(join(here, module)));
}
let requirement = "xarray-sql";
if (wheel) {
  requirement = `emfs:/tmp/${basename(wheel)}`;
  py.FS.writeFile(`/tmp/${basename(wheel)}`, readFileSync(wheel));
}
const micropip = py.pyimport("micropip");
await micropip.install([requirement, "duckdb", "zarr"]);

const app = py.pyimport("app");
let start = performance.now();
const info = JSON.parse(await app.setup());
console.log(
  `setup: ${((performance.now() - start) / 1000).toFixed(1)}s,`,
  `runs ${info.init_times.join(", ")}`,
);

let failed = false;
for (const { title, sql } of info.examples) {
  start = performance.now();
  try {
    const result = JSON.parse(await app.run(sql));
    const seconds = ((performance.now() - start) / 1000).toFixed(1);
    console.log(`\n${title}\n  ${seconds}s, ${result.num_rows} rows`, result.stats);
    console.log("  first row:", result.columns, result.rows[0]);
    if (result.num_rows === 0) failed = true;
  } catch (error) {
    failed = true;
    // Python tracebacks end with the exception line.
    const lines = String(error.message ?? error).trim().split("\n");
    console.error(`\n${title}\n  FAILED: ${lines.at(-1)}\n${lines.slice(-12).join("\n")}`);
  }
}
process.exit(failed ? 1 : 0);
