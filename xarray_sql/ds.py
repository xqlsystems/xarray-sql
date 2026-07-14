"""Reconstruct xarray Datasets from SQL query results.

The inverse of the forward Dataset-to-table pivot done by
:func:`xarray_sql.df.pivot`. Internally defines an :class:`XarrayDataFrame`
wrapper around the DataFusion ``DataFrame`` returned by
:meth:`XarrayContext.sql`, with a :meth:`XarrayDataFrame.to_dataset`
method that round-trips a query result back to ``xr.Dataset``.

Reconstruction is controlled by the ``chunks`` argument to
:meth:`XarrayDataFrame.to_dataset` -- the xarray idiom for tuning how a
result is partitioned -- rather than by reflecting on the query plan:

* **Eager** (``chunks=None``, or the default ``"inherit"`` when the
  result keeps no multi-chunk source dimension): the plan executes
  exactly once via ``execute_stream`` and the result is scattered into a
  dense in-memory Dataset. This is the right default for reductions
  (aggregations), whose results are small, and it never re-executes.
* **Lazy / chunked** (``chunks`` is a mapping, ``"auto"``, or
  ``"inherit"`` over a multi-chunk source dimension): data variables are
  backed by :class:`SQLBackendArray` wrapped in
  ``xarray.core.indexing.LazilyIndexedArray`` and chunked via xarray's
  configured chunk manager (dask, cubed, ...). Each chunk maps onto the
  source partitions and reads its coordinate range on access by
  translating the indexer into a DataFusion ``filter`` expression, so only
  the requested partitions are materialized as Arrow ``RecordBatch`` es
  and scattered into numpy.

``.compute()`` materializes the whole Dataset in memory.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from .lazyscan import DataFusionHandle, DimSpec, LazyResultHandle

Sparsity = Literal["result", "template"]
"""Output coordinate extent for a filtered round-trip.

* ``"result"`` keeps only the dim values present in the query result, so
  the output is sparse and equal to whatever rows came back.
* ``"template"`` reindexes to the registered Dataset's full coord ranges
  and fills absent cells with ``fill_value``.
"""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ds_var_dims(ds: xr.Dataset) -> list[str]:
    """Return a Dataset's data-variable dim order.

    The forward path validates that all data variables share the same dims
    tuple, so the first var's dim order is canonical. Falls back to
    ``ds.dims`` keys for empty Datasets. Always use this rather than
    ``list(ds.dims)`` when round-tripping, since the latter is in
    canonical name order and may not match the variable's axis order.
    """
    if ds.data_vars:
        return list(next(iter(ds.data_vars.values())).dims)
    return list(ds.dims)


def _apply_template(ds: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
    """Recover metadata that the forward SQL pivot strips.

    Adds back, where unambiguous:

    * Data-variable ``attrs`` and ``encoding`` for vars present in
      ``template`` (aggregation aliases like ``air_avg`` get nothing).
      Dtype-bound encoding keys (``dtype``, ``_FillValue``,
      ``missing_value``) are intentionally dropped: SQL may have
      changed the column's dtype (e.g. ``int16`` -> ``float64`` after
      ``AVG`` or a null-introducing filter), and reattaching the
      source's packing would make a later ``ds.to_netcdf()`` write
      corrupt values.
    * Dim-coordinate dtype, where SQL upcasted (datetime is the
      canonical case).
    * Non-dim coordinates whose dims are all present in ``ds`` (scalar
      coords attach as-is; vector coords use ``.sel``).
    * Dataset-level ``attrs``.

    Skipped coords are warned about once per call.
    """
    out = ds.copy()

    # 1. Data-var attrs / encoding for vars present in the template.
    #    Aggregation aliases absent from template intentionally inherit nothing.
    for name in list(out.data_vars):
        if name in template.data_vars:
            out[name].attrs = dict(template[name].attrs)
            # Drop dtype-bound encoding keys; SQL may have changed dtype.
            enc = {
                k: v
                for k, v in template[name].encoding.items()
                if k not in {"dtype", "_FillValue", "missing_value"}
            }
            out[name].encoding = enc

    # 2. Restore dim-coordinate dtype when SQL changed it (e.g. datetime
    #    upcast through pyarrow / pandas) and copy the source's dim-coord
    #    attrs (``standard_name``, ``long_name``, ``units``, etc.).
    for d in list(out.dims):
        if d in template.coords:
            tdt = template.coords[d].dtype
            if out.coords[d].dtype != tdt:
                try:
                    out = out.assign_coords({d: out.coords[d].astype(tdt)})
                except (ValueError, TypeError):
                    pass  # incompatible cast; leave as-is
            out[d].attrs = dict(template.coords[d].attrs)

    # 3. Non-dim coordinates whose dims are all present in the result.
    out_dims = set(out.dims)
    skipped: list[str] = []
    for cname, coord in template.coords.items():
        if cname in template.dims:
            continue  # dim coord; already in out
        if not set(coord.dims) <= out_dims:
            continue  # spans dims the result lacks
        try:
            if not coord.dims:
                # Scalar coord (e.g. weather_dataset.reference_time).
                out = out.assign_coords({cname: coord})
            else:
                sel = {d: out.coords[d] for d in coord.dims}
                out = out.assign_coords({cname: coord.sel(sel)})
        except (KeyError, ValueError, TypeError):
            skipped.append(cname)

    # 4. Dataset-level attrs.
    out.attrs = dict(template.attrs)

    if skipped:
        warnings.warn(
            f"Could not re-attach non-dim coordinates from template: {skipped}",
            stacklevel=3,
        )
    return out


def _axis_numeric(values: np.ndarray) -> np.ndarray:
    """View an axis as float64 for affine position arithmetic."""
    if values.dtype.kind == "M":
        return values.astype("datetime64[ns]").view("int64").astype("float64")
    return values.astype("float64", copy=False)


def _affine_axis(requested: np.ndarray) -> tuple[float, float] | None:
    """``(origin, step)`` when *requested* is uniformly spaced, else None.

    Uniform spacing must hold exactly enough that ``rint((v - origin) /
    step)`` recovers every index: the deviation of each element from its
    affine prediction is checked against a quarter step. Non-numeric
    axes (strings, cftime objects) never qualify.
    """
    if requested.dtype.kind not in ("i", "u", "f", "M") or len(requested) < 2:
        return None
    numeric = _axis_numeric(requested)
    step = (numeric[-1] - numeric[0]) / (len(numeric) - 1)
    if step == 0 or not np.isfinite(step):
        return None
    predicted = numeric[0] + step * np.arange(len(numeric))
    # Written as a <= comparison so a NaN anywhere in the axis (e.g. a
    # NULL dim value in the result) fails the check and falls back to
    # the searchsorted path, which handles it positionally.
    if not (np.abs(numeric - predicted) <= 0.25 * abs(step)).all():
        return None
    return float(numeric[0]), float(step)


def _scatter_batches_to_ndarray(
    batches: list[pa.RecordBatch],
    dimension_columns: list[str],
    requested: dict[str, np.ndarray],
    var_name: str,
    out_shape: tuple[int, ...],
    dtype: np.dtype,
    drop_axes: list[int],
) -> np.ndarray:
    """Convert filtered Arrow ``RecordBatch`` rows into a dense N-D numpy array.

    SQL query results arrive as flat rows; xarray expects N-D arrays.
    This bridges the two: each row carries the dim-coord values that
    identify its cell in the output cube plus the value to write there.
    We look up the row's N-D position by binary-searching its coord
    values within the caller's requested coord arrays
    (``np.searchsorted``), then scatter-write the value at that index.

    Missing combinations (sparse results from filtered queries) stay as
    ``NaN`` for floating-point outputs by pre-filling the buffer; integer
    outputs leave them as ``np.empty``-style undefined values.
    """
    # NaN fill for float outputs; default for int/datetime falls through
    # to ``np.empty``-style undefined values (but every output cell is
    # written below for non-sparse cases).
    out = (
        np.full(out_shape, np.nan, dtype=dtype)
        if np.issubdtype(dtype, np.floating)
        else np.empty(out_shape, dtype=dtype)
    )

    # ``requested[d]`` may be in any order (callers can iselect arbitrary
    # positions, and template coords like air_temperature.lat are descending).
    # ``np.searchsorted`` requires ascending input, so we sort each requested
    # array once, search there, and remap back to the original positions.
    # Uniformly spaced axes (the norm for rasters and regular time steps,
    # ascending or descending) skip the search entirely: the position is
    # ``rint((value - origin) / step)``, a fused vector op several times
    # faster than a per-row binary search.
    affine = {d: _affine_axis(requested[d]) for d in dimension_columns}
    sorted_idx = {
        d: np.argsort(requested[d])
        for d in dimension_columns
        if affine[d] is None
    }
    sorted_req = {d: requested[d][sorted_idx[d]] for d in sorted_idx}

    for batch in batches:
        if batch.num_rows == 0:
            continue
        schema_names = batch.schema.names
        # Build per-dim position arrays for this batch (positions within
        # the caller's requested coord order).
        positions = []
        for d in dimension_columns:
            col_arr = batch.column(schema_names.index(d))
            vals = col_arr.to_numpy(zero_copy_only=False)
            if affine[d] is not None:
                origin, step = affine[d]
                pos = np.rint((_axis_numeric(vals) - origin) / step).astype(
                    np.intp
                )
                positions.append(pos)
            else:
                pos_in_sorted = np.searchsorted(sorted_req[d], vals)
                positions.append(sorted_idx[d][pos_in_sorted])
        value_arr = batch.column(schema_names.index(var_name)).to_numpy(
            zero_copy_only=False
        )
        out[tuple(positions)] = value_arr.astype(dtype, copy=False)

    if drop_axes:
        out = np.squeeze(out, axis=tuple(drop_axes))
    return cast(np.ndarray, out)


class SQLBackendArray(xr.backends.BackendArray):
    """Read-only lazy N-D array view over a re-executable SQL result.

    Bridges xarray's lazy-indexing interface
    (:class:`xarray.backends.BackendArray`) to an engine query result,
    so an xarray ``Dataset`` can present a SQL query as if it were a
    materialized N-D array without actually loading any data until the
    caller asks for it. This is the workhorse that lets
    :meth:`XarrayDataFrame.to_dataset` (and the engine-agnostic
    ``xql.to_dataset(chunks=...)``) return a Dataset cheaply.

    On each ``__getitem__`` call, the requested xarray indexer is
    translated into per-dimension coordinate windows and a column
    projection, executed through a
    :class:`~xarray_sql.lazyscan.LazyResultHandle` (DataFusion, DuckDB,
    or Polars — each renders the windows with its own typed expression
    API). The resulting Arrow ``RecordBatch`` es are scattered into a
    preallocated numpy buffer, so only the requested data is
    materialized.

    Constraints and caveats:

    - Read-only: there is no write path; the backend exists to surface
      query results, not to round-trip writes into a SQL store.
    - The underlying engine object may hold non-picklable references
      (DataFusion's ``SessionContext``, a DuckDB connection). The class
      therefore overrides ``__copy__`` and ``__deepcopy__`` to return
      ``self`` -- this is safe because the backend is read-only.
    - ``IndexingSupport.OUTER``: ``BasicIndexer`` and ``OuterIndexer``
      are translated to filter predicates directly; ``VectorizedIndexer``
      paths through xarray's adapter to outer-then-gather and so still
      works, just less efficiently.

    Raises:
        ValueError, engine exceptions: propagated from the underlying
            filter/project/execute chain if a predicate refers to a
            missing column, the dtype of a literal is incompatible, or
            the execution itself fails.
        AssertionError: from ``np.searchsorted`` mis-alignment, which
            indicates the result contains coordinate values not present
            in the wrapper's pre-computed coord arrays -- usually a
            symptom of a filtered query whose coord discovery missed a
            value.

    Constructed by :func:`_build_lazy_scan`; users should not instantiate
    this class directly.
    """

    def __init__(
        self,
        handle: LazyResultHandle,
        var_name: str,
        dimension_columns: list[str],
        coord_arrays: dict[str, np.ndarray],
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        self._handle = handle
        self._var_name = var_name
        self._dimension_columns = list(dimension_columns)
        self._coord_arrays = coord_arrays
        # Computed once per dim: whether the whole coordinate array is
        # strictly monotonic, the precondition for translating contiguous
        # positional windows into value ranges (see _dim_spec).
        self._monotonic = {
            d: _strictly_monotonic(coord_arrays[d]) for d in dimension_columns
        }
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def __getitem__(self, key: Any) -> np.ndarray:
        return cast(
            np.ndarray,
            xr.core.indexing.explicit_indexing_adapter(
                key,
                self.shape,
                xr.core.indexing.IndexingSupport.OUTER,
                self._raw_getitem,
            ),
        )

    def __copy__(self) -> "SQLBackendArray":
        # The backend is read-only; the underlying DataFusion DataFrame
        # holds a non-picklable SessionContext reference, so sharing the
        # same backend across a copy is both safe and necessary.
        return self

    def __deepcopy__(self, memo: dict) -> "SQLBackendArray":
        return self

    # ------------------------------------------------------------------

    def _raw_getitem(self, key: tuple) -> np.ndarray:
        """Materialize the indexed region described by *key* via the engine.

        ``key`` is a tuple of ``int``/``slice``/1-D integer-array, one per
        dim, in :attr:`_dimension_columns` order.
        """
        requested: dict[str, np.ndarray] = {}
        # Per-dim windows for the engine. Dims whose indexer covers the
        # full extent are omitted entirely so the engine doesn't have to
        # evaluate a tautology.
        specs: dict[str, DimSpec] = {}
        drop_axes: list[int] = []
        for axis, (dim, k) in enumerate(
            zip(self._dimension_columns, key, strict=True)
        ):
            coord = self._coord_arrays[dim]
            contiguous = False
            if isinstance(k, slice):
                start = 0 if k.start is None else k.start
                stop = len(coord) if k.stop is None else k.stop
                step = 1 if k.step is None else k.step
                requested[dim] = np.asarray(coord[start:stop:step])
                contiguous = step == 1
                if start == 0 and stop >= len(coord) and step == 1:
                    continue
            elif isinstance(k, (int, np.integer)):
                requested[dim] = np.asarray([coord[int(k)]])
                drop_axes.append(axis)
            else:
                arr = np.asarray(k)
                requested[dim] = np.asarray(coord[arr])
                if (
                    len(arr) == len(coord)
                    and (arr == np.arange(len(coord))).all()
                ):
                    continue
                contiguous = len(arr) > 1 and bool((np.diff(arr) == 1).all())
            specs[dim] = _dim_spec(
                requested[dim], contiguous, self._monotonic[dim]
            )

        out_shape = tuple(len(requested[d]) for d in self._dimension_columns)
        if any(n == 0 for n in out_shape):
            empty = np.empty(out_shape, dtype=self.dtype)
            squeezed = (
                np.squeeze(empty, axis=tuple(drop_axes)) if drop_axes else empty
            )
            return cast(np.ndarray, squeezed)

        batches = self._handle.fetch(
            specs, self._dimension_columns + [self._var_name]
        )
        return _scatter_batches_to_ndarray(
            batches=batches,
            dimension_columns=self._dimension_columns,
            requested=requested,
            var_name=self._var_name,
            out_shape=out_shape,
            dtype=self.dtype,
            drop_axes=drop_axes,
        )


def _strictly_monotonic(coord: np.ndarray) -> bool:
    """Whether ``coord`` is strictly increasing or strictly decreasing.

    Strict monotonicity of the whole coordinate array is the
    precondition for translating a contiguous positional window into a
    value range: with duplicated or unsorted values, ``[min, max]`` of a
    window admits coordinate values at positions outside the window.
    NaN/NaT (whose comparisons are all false) and non-comparable object
    arrays report ``False``, which safely falls back to value lists.
    """
    if len(coord) < 2:
        return True
    head, tail = coord[:-1], coord[1:]
    try:
        return bool((tail > head).all() or (tail < head).all())
    except TypeError:
        return False


def _dim_spec(
    vals: np.ndarray, contiguous: bool, coord_monotonic: bool
) -> DimSpec:
    """The engine window for one dim's requested coordinate values.

    A contiguous run of positions over a strictly monotonic coordinate
    array is exactly the value range ``[min, max]`` — a two-literal
    predicate engines can push into range pruning. Monotonicity must
    hold for the *entire* coordinate array (``coord_monotonic``), not
    just the requested window: template coords are used verbatim, and
    over a non-monotonic array a window's ``[min, max]`` admits values
    at positions outside the window, which the scatter would then write
    to wrong cells. Anything else (stepped slices, fancy indexers,
    non-monotonic or duplicated coords) must be an explicit value list:
    a range would admit rows the scatter did not request.
    """
    if contiguous and coord_monotonic and len(vals) > 1:
        return ("range", vals.min(), vals.max())
    return ("values", vals, None)


def _c_order_grid(
    dim_cols: dict[str, np.ndarray],
    coord_arrays: dict[str, np.ndarray],
    dimension_columns: list[str],
    total_rows: int,
) -> bool:
    """Whether the result rows form the complete grid in C order.

    True iff the row count is exactly the coordinate product and every
    dimension column is its coordinates repeated/tiled in C order — the
    shape any unfiltered or bbox-windowed scan produces. When it holds,
    data variables are dense row-major arrays already and can be
    reshaped instead of scatter-written (one memcpy versus a
    ``searchsorted`` per dimension per row).
    """
    shape = tuple(len(coord_arrays[d]) for d in dimension_columns)
    if total_rows != int(np.prod(shape)) or total_rows == 0:
        return False
    for k, d in enumerate(dimension_columns):
        inner = int(np.prod(shape[k + 1 :]))
        outer = int(np.prod(shape[:k]))
        view = dim_cols[d].reshape(outer, shape[k], inner)
        if not (view == coord_arrays[d][None, :, None]).all():
            return False
    return True


def _dataset_from_batches(
    batches: list[pa.RecordBatch],
    dimension_columns: list[str],
    field_names: list[str],
    field_types: dict[str, Any],
) -> xr.Dataset:
    """Build a dense in-memory Dataset from Arrow ``RecordBatch`` es.

    The engine-agnostic core of the eager round-trip: derives the
    coordinates and every data variable from a single already-executed
    result, whichever engine produced it. ``field_types`` values only
    need a ``to_pandas_dtype()`` method (both ``pyarrow.DataType`` and
    DataFusion's Arrow type wrappers qualify).

    Complete grid-ordered results (unfiltered scans, bbox windows) are
    reshaped directly; anything else — sparse results from filtered
    queries, engine-reordered rows — falls back to the positional
    scatter, which handles arbitrary row order.
    """
    dim_cols: dict[str, np.ndarray] = {}
    coord_arrays: dict[str, np.ndarray] = {}
    for d in dimension_columns:
        if not batches:
            dim_cols[d] = np.asarray([])
            coord_arrays[d] = np.asarray([])
            continue
        vals = np.concatenate(
            [
                b.column(b.schema.names.index(d)).to_numpy(zero_copy_only=False)
                for b in batches
            ]
        )
        dim_cols[d] = vals
        # Preserve the order coordinate values first appear in the result so an
        # ORDER BY direction (e.g. ``ORDER BY level DESC``) carries through to
        # the Dataset dimension instead of being force-sorted ascending.
        # pd.unique keeps first-appearance order; the scatter below argsorts
        # internally, so arbitrarily-ordered coordinates are placed correctly.
        coord_arrays[d] = np.asarray(pd.unique(vals))
    shape = tuple(len(coord_arrays[d]) for d in dimension_columns)
    total_rows = sum(b.num_rows for b in batches)

    grid_ordered = _c_order_grid(
        dim_cols, coord_arrays, dimension_columns, total_rows
    )

    data_vars: dict[str, xr.Variable] = {}
    for name in field_names:
        if name in dimension_columns:
            continue
        np_dtype = np.dtype(field_types[name].to_pandas_dtype())
        if grid_ordered:
            flat = np.concatenate(
                [
                    b.column(b.schema.names.index(name)).to_numpy(
                        zero_copy_only=False
                    )
                    for b in batches
                ]
            )
            dense = flat.astype(np_dtype, copy=False).reshape(shape)
        else:
            dense = _scatter_batches_to_ndarray(
                batches=batches,
                dimension_columns=dimension_columns,
                requested=coord_arrays,
                var_name=name,
                out_shape=shape,
                dtype=np_dtype,
                drop_axes=[],
            )
        data_vars[name] = xr.Variable(dimension_columns, dense)

    coords_arg = {d: coord_arrays[d] for d in dimension_columns}
    return xr.Dataset(data_vars=data_vars, coords=coords_arg)


def _materialize(
    inner_df: Any,
    dimension_columns: list[str],
    field_names: list[str],
    field_types: dict[str, Any],
) -> xr.Dataset:
    """Execute the query once and build a dense in-memory Dataset.

    Runs the plan exactly once via ``execute_stream()`` -- streaming the result
    as Arrow ``RecordBatch`` es (``datafusion.RecordBatch.to_pyarrow()``) -- then
    derives both the coordinates and every data variable from that single pass.
    This is the eager path, used when no output chunking is requested. It never
    re-executes, so an aggregation over a remote Zarr scan costs exactly one
    scan, regardless of how many dimensions or variables the result has.
    """
    batches = [b.to_pyarrow() for b in inner_df.execute_stream()]
    return _dataset_from_batches(
        batches, dimension_columns, field_names, field_types
    )


_PURE_SCAN_NODES = {"Projection", "Sort", "TableScan", "SubqueryAlias"}


def _unfiltered_scan_table(inner_df: Any) -> str | None:
    """Return the scanned table name iff the query is a pure unfiltered scan.

    A pure scan only contains ``Projection``, ``Sort``, ``TableScan``,
    ``SubqueryAlias`` nodes and exactly one ``TableScan``. Anything else
    (``Filter``, ``Aggregate``, ``Join``, ``Union``, ``Limit``, multi-table
    joins, ...) returns ``None`` so the caller falls back to per-dim
    discovery. The returned name is the registered table the caller can
    look up to source coord arrays from.
    """
    try:
        lp = inner_df.logical_plan()
    except Exception:
        return None
    table_name: str | None = None
    stack = [lp]
    while stack:
        node = stack.pop()
        try:
            variant = node.to_variant()
        except Exception:
            return None
        cls = type(variant).__name__
        if cls not in _PURE_SCAN_NODES:
            return None
        if cls == "TableScan":
            try:
                this = variant.table_name()
            except (AttributeError, TypeError):
                return None
            if not isinstance(this, str):
                return None
            if table_name is not None and table_name != this:
                return None  # multi-table scan; not a single source
            table_name = this
        stack.extend(node.inputs())
    return table_name


def _maybe_template_coords(
    templates: dict[str, xr.Dataset] | None,
    dimension_columns: list[str],
    inner_df: Any,
) -> dict[str, np.ndarray] | None:
    """Use the scanned table's registered coord arrays directly when safe.

    Returns coord arrays sourced from the registered Dataset for the
    scanned table iff the query is an unfiltered scan over that single
    table and the registered Dataset carries all requested dims. Returns
    ``None`` otherwise so the caller falls back to per-dim discovery.
    Skipping discovery avoids one full plan execution per dim and
    preserves the source's coordinate order.

    Coord values come from the **scanned** registered Dataset, not from
    any user-supplied ``template=`` (which is for metadata recovery
    only). That keeps the fast path correct when a user with multiple
    registered Datasets passes a metadata template that differs from
    the query's source.
    """
    if not templates:
        return None
    table = _unfiltered_scan_table(inner_df)
    if table is None or table not in templates:
        return None
    source = templates[table]
    if not all(d in source.coords for d in dimension_columns):
        return None
    return {d: np.asarray(source.coords[d].values) for d in dimension_columns}


def _build_lazy_scan(
    handle: LazyResultHandle,
    dimension_columns: list[str],
    field_names: list[str],
    field_types: dict[str, Any],
    coord_arrays: dict[str, np.ndarray] | None = None,
) -> xr.Dataset:
    """Build a lazy Dataset whose data vars are :class:`SQLBackendArray`.

    Used when output chunking is requested: each data variable stays lazy and,
    once wrapped by ``Dataset.chunk``, every chunk reads its coordinate range
    via a pushdown filter on first access. Coordinates come either from the
    caller (the scanned table's registered Dataset for unfiltered DataFusion
    scans -- see :func:`_maybe_template_coords` -- or an explicitly trusted
    template) or from per-dim distinct queries through the handle; over a
    registered pushdown table the engine projects to that single coordinate
    column, so discovery reads coordinate values only (no data-variable I/O).
    """
    if coord_arrays is None:
        coord_arrays = {}
        for d in dimension_columns:
            # ``distinct`` returns engine order; sort ascending so
            # positional slices map onto contiguous value ranges.
            coord_arrays[d] = np.sort(handle.distinct(d))
    shape = tuple(len(coord_arrays[d]) for d in dimension_columns)

    data_vars: dict[str, xr.Variable] = {}
    for name in field_names:
        if name in dimension_columns:
            continue
        np_dtype = field_types[name].to_pandas_dtype()
        backend = SQLBackendArray(
            handle=handle,
            var_name=name,
            dimension_columns=dimension_columns,
            coord_arrays=coord_arrays,
            shape=shape,
            dtype=np_dtype,
        )
        lazy = xr.core.indexing.LazilyIndexedArray(backend)
        data_vars[name] = xr.Variable(dimension_columns, lazy)

    coords_arg = {d: coord_arrays[d] for d in dimension_columns}
    return xr.Dataset(data_vars=data_vars, coords=coords_arg)


def _auto_chunk_target_bytes() -> int:
    """Byte target for ``chunks="auto"`` (the chunk manager's, else 128 MiB)."""
    try:
        import dask
        from dask.utils import parse_bytes

        return int(parse_bytes(dask.config.get("array.chunk-size")))
    except Exception:
        return 128 * 1024 * 1024


def _auto_chunks(
    template: xr.Dataset | None,
    dimension_columns: list[str],
    field_types: dict[str, Any],
) -> dict[str, int] | None:
    """Resolve ``chunks="auto"`` to a source-partition-aligned chunk spec.

    Sizes chunks to roughly the chunk manager's byte target (dask's
    ``array.chunk-size``, default 128 MiB) but snaps boundaries to whole source
    partitions, so every chunk is a union of source partitions -- no chunk splits
    a partition (which would make adjacent chunks re-read it). This is what makes
    ``"auto"`` useful for finely partitioned sources (e.g. ERA5
    ``chunks={"time": 1}``): it coarsens many tiny partitions into memory-sized,
    aligned chunks. Returns ``None`` when there is no resolvable source grid to
    align to, so the caller falls back to the chunk manager's own ``"auto"``.
    """
    if template is None:
        return None
    part = template.chunksizes  # dim -> tuple of source chunk lengths
    chunked_dims = [
        d for d in dimension_columns if d in part and len(part[d]) > 1
    ]
    if not chunked_dims:
        return None

    itemsizes = [
        np.dtype(t.to_pandas_dtype()).itemsize
        for name, t in field_types.items()
        if name not in dimension_columns
    ]
    itemsize = max(itemsizes) if itemsizes else 8

    # Bytes in one source-partition block: the nominal source chunk length per
    # dimension (``part[d][0]``) multiplied across all dims, times itemsize.
    block_bytes = itemsize
    for d in dimension_columns:
        if d in part:
            block_bytes *= int(part[d][0])
    # Number of source partitions to merge per chunk to approach the target.
    merge = max(1, _auto_chunk_target_bytes() // max(block_bytes, 1))

    # Absorb the coarsening into the most finely partitioned dimension; the rest
    # keep their source chunk length. xarray caps an oversize chunk at the dim
    # length, so an over-large merge simply yields a single chunk on that dim.
    primary = max(chunked_dims, key=lambda d: len(part[d]))
    return {
        d: int(part[d][0]) * (merge if d == primary else 1)
        for d in chunked_dims
    }


def _result_to_xarray(
    inner_df: Any,
    dimension_columns: list[str],
    template: xr.Dataset | None,
    sparsity: Sparsity,
    fill_value: Any,
    chunks: Mapping[str, int] | str | None,
    templates: dict[str, xr.Dataset] | None = None,
) -> xr.Dataset:
    """Reconstruct an ``xr.Dataset`` from a SQL result.

    ``chunks`` (already resolved by :meth:`XarrayDataFrame._resolve_chunks`)
    selects the execution strategy:

    * ``None`` -> eager: execute once and materialize a dense Dataset
      (:func:`_materialize`). Correct for any query and the right default for
      reductions, whose results are small.
    * a mapping (or ``"auto"``) -> lazy/chunked: build :class:`SQLBackendArray`
      data variables (:func:`_build_lazy_scan`) and wrap them with
      ``Dataset.chunk`` so each chunk reads its coordinate range via filter
      pushdown. The chunk grid maps onto the source partitions. Chunking goes
      through xarray's configured chunk manager (dask, cubed, ...), so no
      chunked-array backend is imported directly here.
    """
    if sparsity not in ("result", "template"):
        raise ValueError(
            f"sparsity must be 'result' or 'template', got {sparsity!r}"
        )
    if sparsity == "template" and template is None:
        raise ValueError(
            "sparsity='template' requires template= to be supplied"
        )

    schema = inner_df.schema()
    field_names = [f.name for f in schema]
    field_types = {f.name: f.type for f in schema}

    if chunks is None:
        ds = _materialize(inner_df, dimension_columns, field_names, field_types)
    else:
        ds = _build_lazy_scan(
            DataFusionHandle(inner_df),
            dimension_columns,
            field_names,
            field_types,
            coord_arrays=_maybe_template_coords(
                templates, dimension_columns, inner_df
            ),
        )
    return _finish_dataset(
        ds,
        dimension_columns,
        template,
        sparsity,
        fill_value,
        chunks,
        field_types,
    )


def _finish_dataset(
    ds: xr.Dataset,
    dimension_columns: list[str],
    template: xr.Dataset | None,
    sparsity: Sparsity,
    fill_value: Any,
    chunks: Mapping[str, int] | str | None,
    field_types: dict[str, Any],
) -> xr.Dataset:
    """Shared reconstruction tail: sparsity, template metadata, chunking."""
    if sparsity == "template":
        assert template is not None
        indexers = {
            d: template.coords[d].values
            for d in dimension_columns
            if d in template.coords and d in template.dims
        }
        if indexers:
            ds = ds.reindex(indexers, fill_value=fill_value)

    if template is not None:
        ds = _apply_template(ds, template)

    if chunks is not None:
        if chunks == "auto":
            # Snap the byte-budgeted "auto" sizing to source partition
            # boundaries; fall back to the chunk manager's own "auto" when there
            # is no source grid to align to.
            chunks = (
                _auto_chunks(template, dimension_columns, field_types) or "auto"
            )
        # Wrap the lazy data variables in the configured chunk manager (dask by
        # default). Each chunk reads its coordinate range via pushdown on access.
        ds = ds.chunk(chunks)
    return ds


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


class XarrayDataFrame:
    """Wrapper around a DataFusion ``DataFrame`` with xarray-aware helpers.

    Returned by :meth:`xarray_sql.XarrayContext.sql`. Forwards every
    attribute it does not define itself to the wrapped DataFrame, so
    ``.collect()``, ``.schema()``, ``.show()``, ``.count()`` all work
    unchanged.

    Carries a private snapshot of the context's registered Datasets so
    :meth:`to_dataset` can default ``dims`` and recover metadata
    dropped by the forward pivot.

    Users should not construct this class directly; let
    :meth:`XarrayContext.sql` produce it.
    """

    def __init__(
        self,
        inner: Any,
        templates: dict[str, xr.Dataset] | None = None,
    ) -> None:
        """Construct a wrapper.

        Args:
            inner: The underlying ``datafusion.DataFrame`` returned by
                :meth:`XarrayContext.sql`.
            templates: Snapshot of the registered Datasets on the producing
                context, keyed by the SQL identifier each was registered
                under. Used by :meth:`to_dataset` to recover metadata that
                the forward pivot strips. ``None`` means no metadata
                recovery is possible from registrations alone; callers may
                still pass ``template=`` to :meth:`to_dataset` explicitly.
        """
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_templates", dict(templates or {}))

    def to_pandas(self) -> pd.DataFrame:
        """Materialize the result as a ``pd.DataFrame`` (DataFusion API)."""
        return self._inner.to_pandas()

    def to_dataset(
        self,
        dims: list[str] | None = None,
        template: xr.Dataset | str | None = None,
        sparsity: Sparsity = "result",
        fill_value: Any = np.nan,
        chunks: Mapping[str, int] | str | None = "inherit",
    ) -> xr.Dataset:
        """Convert the result to an ``xr.Dataset``.

        Args:
            dims: Result columns to use as Dataset dimensions. When
                ``None``, defaults to a registered Dataset's dimensions that
                survive into the result columns, so an aggregation that drops
                dims (e.g. ``GROUP BY time`` over a ``(time, lat, lon)`` grid)
                round-trips on the remaining dim. Raises when no dimension
                survives, or when several registered Datasets imply different
                dims (pass ``dims`` explicitly then).
            template: Source to recover metadata (attrs, encoding, non-dim
                coordinates, dim-coord dtype) from. Either an ``xr.Dataset``
                used directly, or the name of a registered table (e.g.
                ``"era5.surface"``) whose Dataset is looked up. When ``None``
                and exactly one Dataset is registered, that one is used.
            sparsity: ``"result"`` (default) keeps only dim values
                present in the result. ``"template"`` reindexes to the
                template's full coord ranges, filling absent cells with
                ``fill_value``; requires a template.
            fill_value: Used when ``sparsity="template"`` reindexes
                to a wider extent. Defaults to ``np.nan``.
            chunks: Output chunking, controlling laziness (an xarray idiom).

                * ``"inherit"`` (default): reuse the source Dataset's chunk
                  sizes, but only for dimensions that were genuinely split into
                  multiple chunks in the input -- so the output chunk grid maps
                  onto the source partitions. A reduction that drops the chunked
                  dimension (e.g. a global aggregation) inherits nothing and so
                  is materialized eagerly. Falls back to eager when no source
                  Dataset is resolvable.
                * ``None``: eager. Execute the query once and return a dense
                  in-memory Dataset. Best for reductions (small results).
                * a mapping (e.g. ``{"time": 100}``): chunk explicitly. Each
                  chunk reads its coordinate range lazily via filter pushdown on
                  access, through xarray's configured chunk manager (dask,
                  cubed, ...).
                * ``"auto"``: size chunks to the chunk manager's byte target but
                  snap boundaries to whole source partitions, so each chunk is a
                  union of source partitions. Useful for finely partitioned
                  sources (e.g. ERA5 ``chunks={"time": 1}``), coarsening many
                  tiny partitions into memory-sized, aligned chunks.

        Returns:
            An ``xr.Dataset`` with ``dims`` as dimensions and the
            remaining result columns as data variables.

        Raises:
            ValueError: ``dims`` cannot be inferred, names a missing
                column, or the result has duplicate dim tuples;
                ``template`` names an unknown registered table; or
                ``sparsity="template"`` is requested without a
                resolvable template.
        """
        if not isinstance(template, xr.Dataset):
            # ``template`` is a registered-table name or None; look it up.
            template = self._resolve_template(template)
        if dims is None:
            dims = self._infer_dimension_columns(preferred_template=template)
        resolved_chunks = self._resolve_chunks(chunks, template, dims)
        return _result_to_xarray(
            inner_df=self._inner,
            dimension_columns=dims,
            template=template,
            sparsity=sparsity,
            fill_value=fill_value,
            chunks=resolved_chunks,
            templates=self._templates,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_chunks(
        chunks: Mapping[str, int] | str | None,
        template: xr.Dataset | None,
        dimension_columns: list[str],
    ) -> Mapping[str, int] | str | None:
        """Resolve the ``chunks`` argument to a concrete spec or ``None``.

        ``None`` selects the eager path; anything else selects the lazy/chunked
        path. ``"inherit"`` reuses the source Dataset's chunk sizes -- but only
        for dimensions actually split into more than one chunk in the input
        (a single full chunk is not "chunked"), so reductions that drop the
        chunked dimension resolve to ``None`` (eager) automatically. Mappings
        pass through unchanged; ``"auto"`` passes through here and is snapped to
        source partition boundaries later (see :func:`_auto_chunks`).
        """
        if chunks is None:
            return None
        if chunks == "inherit":
            if template is None:
                return None
            sizes = template.chunksizes  # dim -> tuple of chunk lengths
            inherited = {
                d: sizes[d][0]
                for d in dimension_columns
                if d in sizes and len(sizes[d]) > 1
            }
            return inherited or None
        return chunks

    def _resolve_template(self, name: str | None) -> xr.Dataset | None:
        """Pick a template Dataset for metadata recovery by registered name.

        Priority:
          1. The named registered table (``name``).
          2. If exactly one Dataset is registered on the context, use it.
          3. None.
        """
        templates = self._templates
        if name is not None:
            if name not in templates:
                raise ValueError(
                    f"template={name!r} is not a registered table on this "
                    f"context. Registered: {list(templates)}"
                )
            return templates[name]
        if len(templates) == 1:
            return next(iter(templates.values()))
        return None

    def _infer_dimension_columns(
        self, preferred_template: xr.Dataset | None = None
    ) -> list[str]:
        """Pick a default ``dimension_columns`` from the registry, or raise.

        A registered Dataset's dims that survive into the result columns
        become the dimensions, so aggregations that drop dims (e.g.
        ``GROUP BY time`` over a ``(time, lat, lon)`` grid) round-trip on the
        surviving dim(s). Uses the data variable's dim order (via
        :func:`_ds_var_dims`) so the original axis order is preserved.
        """
        result_cols = set(self._result_columns())

        def surviving(template: xr.Dataset) -> list[str]:
            # Template dims still present in the result, in var axis order.
            return [d for d in _ds_var_dims(template) if d in result_cols]

        if preferred_template is not None:
            preferred = surviving(preferred_template)
            if preferred:
                return preferred
        if not self._templates:
            raise ValueError(
                "dims cannot be inferred (no registered "
                "Dataset on this result); pass dims=[...] "
                "explicitly."
            )
        candidates = {tuple(surviving(t)) for t in self._templates.values()}
        candidates.discard(())  # templates with no surviving dim
        if len(candidates) == 1:
            return list(next(iter(candidates)))
        if not candidates:
            raise ValueError(
                "dims cannot be inferred: no registered Dataset "
                "dimension survives in the result columns. Pass "
                "dims=[...] explicitly."
            )
        raise ValueError(
            "dims cannot be inferred unambiguously: multiple "
            "registered Datasets are compatible with the result. Pass "
            "dims=[...] explicitly."
        )

    def _result_columns(self) -> list[str]:
        """Return the result's column names without materializing rows."""
        return [field.name for field in self._inner.schema()]

    def __getattr__(self, name: str) -> Any:
        # Runs only when ``name`` is not found via normal lookup, so this
        # safely forwards anything we have not overridden.
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return repr(self._inner)
