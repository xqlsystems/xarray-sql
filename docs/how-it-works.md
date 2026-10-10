# How it works

xarray-sql lets SQL engines query Xarray Datasets as if they were tables,
and turns query results back into Datasets. The explainer below walks
through the three steps with a small air-temperature Dataset. It runs
entirely in this page; nothing is downloaded.

<iframe id="explainer" src="../explainer/" title="How xarray-sql works" loading="lazy"
  style="width: 100%; height: 1500px; border: 0;"></iframe>

<script>
  // Fit the iframe to the explainer's height.
  addEventListener("message", (event) => {
    const height = event.data && event.data.xqlExplainerHeight;
    const frame = document.getElementById("explainer");
    if (frame && height && event.source === frame.contentWindow) {
      frame.style.height = `${height}px`;
    }
  });
</script>

## What happens in each step

1. **Registering** a Dataset with [register][xarray_sql.register] (or
   [Context.from_dataset][xarray_sql.Context.from_dataset]) creates a table
   whose rows are the Dataset's grid cells. The table reads from the
   Dataset as the query runs, one chunk at a time, so registering copies
   nothing.
2. **Querying** runs in the engine: DuckDB, DataFusion, or any engine
   that reads Arrow. Filters on dimension columns are pushed down to the
   table, which skips every partition whose coordinate range cannot match,
   so a selective query reads few chunks. See [Engines](engines.md).
3. **Round-tripping** with [to_dataset][xarray_sql.to_dataset] pivots the
   result rows back onto a grid and restores metadata from the registered
   Dataset. By default the result is lazy: each access reruns the query
   for just the part you select.

To run real queries over a public Zarr store, try
[xarray-sql in the browser](browser.md).
