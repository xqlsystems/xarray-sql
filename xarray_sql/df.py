import itertools
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from . import cftime as cft


Block = dict[Hashable, slice]
Chunks = dict[str, int] | None


# Borrowed from Xarray
def _get_chunk_slicer(
    dim: Hashable, chunk_index: Mapping, chunk_bounds: Mapping
):
    if dim in chunk_index:
        which_chunk = chunk_index[dim]
        return slice(
            chunk_bounds[dim][which_chunk], chunk_bounds[dim][which_chunk + 1]
        )
    return slice(None)


def compute_chunks(
    ds: xr.Dataset, chunks: dict[str, int]
) -> dict[Hashable, tuple[int, ...]]:
    """Per-dim chunk-size tuples matching ``ds.chunk(chunks).chunks``.

    Pure arithmetic replacement for the dask rechunk round-trip; dask's
    ``.chunk()`` eagerly builds a task graph, which dominates
    ``block_slices()`` cost on large datasets.
    """
    existing = dict(ds.chunks) if ds.chunks else {}
    result: dict[Hashable, tuple[int, ...]] = {}
    for dim in ds.dims:
        size = ds.sizes[dim]
        if dim in chunks:
            cs = chunks[dim]
            if cs <= 0 or cs >= size:
                result[dim] = (size,)
            else:
                n_full, rem = divmod(size, cs)
                result[dim] = (cs,) * n_full + ((rem,) if rem else ())
        elif dim in existing:
            result[dim] = tuple(existing[dim])
        else:
            result[dim] = (size,)
    return result


def resolve_chunks(
    ds: xr.Dataset, chunks: Chunks
) -> Mapping[Hashable, tuple[int, ...]]:
    """Normalise the user's ``chunks`` argument to per-dim size tuples.

    Filters out keys for dims this dataset doesn't have (sub-datasets in a
    heterogeneous group need not contain every dimension named in the
    spec), then either rechunks arithmetically via ``compute_chunks`` or
    falls back to the dataset's existing dask chunks.

    Returns an empty mapping for scalar datasets; callers should treat that
    as "one block covering everything".
    """
    if chunks is not None:
        chunks = {dim: size for dim, size in chunks.items() if dim in ds.sizes}
    if chunks:
        return compute_chunks(ds, chunks)
    return {d: tuple(c) for d, c in ds.chunks.items()}


def _ensure_default_indexes(ds: xr.Dataset) -> xr.Dataset:
    """Attach a default integer index coordinate to every dimension lacking one.

    xarray allows "dimensions without coordinates"; these are absent from
    ``ds.coords``, so they are dropped from the SQL schema and, once a block is
    sliced out with ``isel``, their position is synthesized *relative to the
    block* (restarting at 0 in every partition). Materialising an explicit
    ``arange`` index up front turns them into ordinary dimension coordinates, so
    they appear as columns and carry their absolute position through chunked
    reads. Datasets whose dimensions already have coordinates are returned
    unchanged.
    """
    missing = {
        dim: np.arange(ds.sizes[dim]) for dim in ds.dims if dim not in ds.coords
    }
    return ds.assign_coords(missing) if missing else ds


def _block_slices_from_resolved(
    ds: xr.Dataset, resolved: Mapping[Hashable, tuple[int, ...]]
) -> Iterator[Block]:
    """Emit blocks given pre-resolved per-dim chunk tuples."""
    if not resolved:
        # No chunkable dimensions. A dimensionless dataset (e.g. scalar
        # metadata variables) is a single block; a dataset that has
        # dimensions but no chunking is a user error.
        assert not ds.sizes, (
            "Dataset `ds` must be chunked or `chunks` must be provided."
        )
        yield {}
        return

    chunk_bounds = {
        dim: np.cumsum((0,) + tuple(c)) for dim, c in resolved.items()
    }
    ichunk = {dim: range(len(tuple(c))) for dim, c in resolved.items()}
    ick, icv = zip(*ichunk.items())  # Makes same order of keys and val.
    chunk_idxs = (dict(zip(ick, i)) for i in itertools.product(*icv))
    yield from (
        {
            dim: _get_chunk_slicer(dim, chunk_index, chunk_bounds)
            for dim in ds.dims
        }
        for chunk_index in chunk_idxs
    )


# Adapted from Xarray `map_blocks` implementation.
def block_slices(ds: xr.Dataset, chunks: Chunks = None) -> Iterator[Block]:
    """Compute block slices for a chunked Dataset."""
    yield from _block_slices_from_resolved(ds, resolve_chunks(ds, chunks))


def explode(ds: xr.Dataset, chunks: Chunks = None) -> Iterator[xr.Dataset]:
    """Explodes a dataset into its chunks."""
    yield from (ds.isel(b) for b in block_slices(ds, chunks=chunks))


def _block_len(block: Block) -> int:
    return int(np.prod([v.stop - v.start for v in block.values()]))


def from_map_batched(
    func: Callable[..., pd.DataFrame],
    *iterables: tuple[Any, ...],
    args: tuple | None = None,
    schema: pa.Schema = None,
    **kwargs: dict[str, Any],
) -> pa.RecordBatchReader:
    """Create a PyArrow RecordBatchReader by mapping a function over iterables.

    This is equivalent to dask's from_map but returns a PyArrow
    RecordBatchReader that can be used with DataFusion. It iterates over
    RecordBatches which are created via the `func` one-at-a-time.

    Args:
      func: Function to apply to each element of the iterables. Currently, the
        function must return a Pandas DataFrame.
      *iterables: Iterable objects to map the function over.
      schema: Optional schema needed for the RecordBatchReader.
      args: Additional positional arguments to pass to func.
      **kwargs: Additional keyword arguments to pass to func.

    Returns:
      A PyArrow RecordBatchReader containing the stream of RecordBatches.
    """
    if args is None:
        args = ()

    def map_batches():
        for items in zip(*iterables):
            df = func(*items, *args, **kwargs)
            yield pa.RecordBatch.from_pandas(df, schema=schema)

    return pa.RecordBatchReader.from_batches(schema, map_batches())


def from_map(
    func: Callable,
    *iterables: tuple[Any, ...],
    args: tuple | None = None,
    **kwargs: dict[str, Any],
) -> pa.Table:
    """Create a PyArrow Table by mapping a function over iterables.

    This is equivalent to dask's from_map but returns a PyArrow Table
    that can be used with DataFusion instead of a Dask DataFrame.

    Args:
      func: Function to apply to each element of the iterables.
      *iterables: Iterable objects to map the function over.
      args: Additional positional arguments to pass to func.
      **kwargs: Additional keyword arguments to pass to func.

    Returns:
      A PyArrow Table containing the concatenated results.
    """
    if args is None:
        args = ()

    # Apply the function to each combination of iterable elements
    results = []
    for items in zip(*iterables) if len(iterables) > 1 else iterables[0]:
        if isinstance(items, tuple):
            result = func(*items, *args, **kwargs)
        else:
            result = func(items, *args, **kwargs)

        # Convert result to PyArrow Table
        if isinstance(result, pd.DataFrame):
            pa_table = pa.Table.from_pandas(result)
        elif isinstance(result, pa.Table):
            pa_table = result
        else:
            # Try to convert to pandas first, then to PyArrow
            try:
                df = pd.DataFrame(result)
                pa_table = pa.Table.from_pandas(df)
            except Exception as e:
                raise ValueError(
                    f"Cannot convert function result to PyArrow Table: {e}"
                )

        results.append(pa_table)

    # Concatenate all results
    if not results:
        raise ValueError("No results to concatenate")

    return pa.concat_tables(results)


def pivot(ds: xr.Dataset) -> pd.DataFrame:
    """Converts an xarray Dataset to a pandas DataFrame."""
    return ds.to_dataframe().reset_index()  # type: ignore[no-any-return]


def dataset_to_record_batch(
    ds: xr.Dataset, schema: pa.Schema
) -> pa.RecordBatch:
    """Convert an xarray Dataset partition to an Arrow RecordBatch.

    Builds the RecordBatch directly from numpy arrays, bypassing the pandas
    round-trip (to_dataframe → reset_index → from_pandas) used by pivot().
    For large partitions this reduces peak memory from ~5× to ~2× the
    partition size.

    Dimension coordinates are broadcast to the full partition shape and
    ravelled. np.broadcast_to() is zero-copy; the ravel() forces one copy
    per coordinate (unavoidable, since broadcast arrays are non-contiguous).
    Data variable arrays are ravelled in-place — a zero-copy view when the
    underlying array is already C-contiguous (the common case for numpy-backed
    xarray datasets).

    Args:
        ds: A partition-sized xarray Dataset (already sliced via isel).
        schema: The Arrow schema for the output, as produced by _parse_schema.
            Column order in the output matches schema field order.

    Returns:
        A RecordBatch with one column per dimension coordinate and data
        variable, in schema order.
    """
    # Use the data variable's dimension order as canonical so coordinate
    # broadcasts and data variable ravels use the same layout. All data
    # variables are validated to share the same dims tuple.
    if ds.data_vars:
        first_var = next(iter(ds.data_vars.values()))
        dim_names = list(first_var.dims)
        shape = first_var.shape
    else:
        dim_names = list(ds.sizes.keys())
        shape = tuple(ds.sizes[d] for d in dim_names)

    arrays = []
    for field in schema:
        name = field.name
        if name in ds.coords and name in ds.dims:
            axis = dim_names.index(name)
            coord = ds.coords[name].values
            if cft.is_cftime(coord):
                coord = cft.convert_for_field(coord, field)
            reshape = [1] * len(shape)
            reshape[axis] = coord.shape[0]
            if pa.types.is_dictionary(field.type):
                # Dictionary-encode: the distinct coordinate values are the
                # dictionary, and the per-row indices are the coordinate's own
                # axis index broadcast across the partition — no dense array of
                # repeated values is ever built.
                idx = np.broadcast_to(
                    np.arange(coord.shape[0]).reshape(reshape), shape
                ).ravel()
                dictionary = pa.array(coord, type=field.type.value_type)
                indices = pa.array(idx, type=field.type.index_type)
                arrays.append(
                    pa.DictionaryArray.from_arrays(indices, dictionary)
                )
            else:
                # Broadcast 1-D coordinate to the full partition shape, ravel.
                arr = np.broadcast_to(coord.reshape(reshape), shape).ravel()
                arrays.append(pa.array(arr, type=field.type))
        else:
            # Data variable: ravel to 1-D (zero-copy for C-contiguous arrays).
            raw = ds[name].values.ravel()
            if cft.is_cftime(ds[name].values):
                raw = cft.convert_for_field(ds[name].values, field)

            # from_pandas=True maps NaN → Arrow null inside the C++ copy kernel,
            # so SQL aggregates (MAX, MIN, AVG) skip missing values correctly.
            arrays.append(pa.array(raw, type=field.type, from_pandas=True))

    return pa.RecordBatch.from_arrays(arrays, schema=schema)


#: Default number of rows per emitted Arrow RecordBatch.
#: 64 K rows balances DataFusion pipeline depth against per-batch overhead.
DEFAULT_BATCH_SIZE: int = 65_536


def iter_record_batches(
    ds: xr.Dataset,
    schema: pa.Schema,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
    """Yield RecordBatches of at most *batch_size* rows from a partition Dataset.

    Unlike `dataset_to_record_batch`, which materialises the entire
    partition as one batch, this generator emits smaller batches so that
    DataFusion can begin filtering and aggregating before the full partition
    is loaded.  Peak memory per batch is O(batch_size) for coordinate columns
    and O(partition_size) for data-variable columns (which must be loaded in
    full from storage).

    Coordinate values are computed per batch via strided index arithmetic —
    no broadcast array spanning the whole partition is ever allocated.  Data
    variable flat arrays are loaded once (triggering any remote I/O) and then
    sliced as zero-copy views for each batch.

    Args:
        ds: A partition-sized xarray Dataset (already sliced via isel).
        schema: The Arrow schema for the output, as produced by _parse_schema.
        batch_size: Maximum number of rows per yielded RecordBatch.

    Yields:
        RecordBatches in schema column order, covering all rows of the
        partition exactly once.
    """
    if ds.data_vars:
        first_var = next(iter(ds.data_vars.values()))
        dim_names = list(first_var.dims)
        shape = first_var.shape
    else:
        dim_names = list(ds.sizes.keys())
        shape = tuple(ds.sizes[d] for d in dim_names)

    total_rows = int(np.prod(shape))

    # Preload small 1-D coordinate arrays (negligible memory).
    # Convert cftime objects to numeric values matching the schema type.
    coord_values = {}
    for name in dim_names:
        vals = ds.coords[name].values
        if cft.is_cftime(vals):
            coord_values[name] = cft.convert_for_field(vals, schema.field(name))
        else:
            coord_values[name] = vals

    # C-order stride for each dimension: stride[k] = prod(shape[k+1:]).
    # Flat row index i → coordinate index for dim k: (i // stride[k]) % shape[k].
    strides = [int(np.prod(shape[k + 1 :])) for k in range(len(shape))]

    # Load data-variable arrays fully (triggers Dask/Zarr compute once).
    # ravel() is a zero-copy view for C-contiguous arrays.
    data_arrays = {}
    for field in schema:
        if field.name not in ds.dims:
            raw = ds[field.name].values
            if cft.is_cftime(raw):
                data_arrays[field.name] = cft.convert_for_field(raw, field)
            else:
                data_arrays[field.name] = raw.ravel()

    for row_start in range(0, total_rows, batch_size):
        row_end = min(row_start + batch_size, total_rows)
        row_idx = np.arange(row_start, row_end)

        arrays = []
        for field in schema:
            name = field.name
            if name in ds.coords and name in ds.dims:
                k = dim_names.index(name)
                coord_idx = (row_idx // strides[k]) % shape[k]
                if pa.types.is_dictionary(field.type):
                    # Emit the coordinate as a dictionary array: the distinct
                    # coordinate values are the dictionary, and the strided
                    # per-row indices we just computed are exactly the
                    # dictionary indices — no broadcast of repeated values.
                    dictionary = pa.array(
                        coord_values[name], type=field.type.value_type
                    )
                    indices = pa.array(coord_idx, type=field.type.index_type)
                    arrays.append(
                        pa.DictionaryArray.from_arrays(indices, dictionary)
                    )
                else:
                    arrays.append(
                        pa.array(coord_values[name][coord_idx], type=field.type)
                    )
            else:
                arrays.append(
                    pa.array(
                        data_arrays[name][row_start:row_end],
                        type=field.type,
                        from_pandas=True,
                    )
                )

        yield pa.RecordBatch.from_arrays(arrays, schema=schema)


def _coord_index_type(n_values: int) -> pa.DataType:
    """Signed Arrow int type for a coordinate dictionary of ``n_values`` keys.

    The index must survive DataFusion *concatenating* per-batch dictionaries
    across a scan. Those concatenations are not always unified (arrow merges
    dictionary values on a size heuristic, not a guarantee), so the *combined*
    dictionary an operator sees can be many times a single partition's
    cardinality — an unchunked coordinate repeated across N partitions can reach
    ``card × N``. A narrow key (int8/int16) therefore overflows under streaming
    aggregation ("Dictionary key bigger than the key type"), so we do not use
    one. ``int32`` covers ~2.1B combined entries (any realistic grid), and
    ``int64`` backstops the astronomically-large case. Indices run
    ``0 .. n_values - 1``, so a signed max ``M`` holds a cardinality of ``M + 1``.
    """
    if n_values <= np.iinfo(np.int32).max + 1:
        return pa.int32()
    return pa.int64()


def _as_dictionary_field(field: pa.Field, n_values: int) -> pa.Field:
    """Dictionary-encode a coordinate field when it is worthwhile and safe.

    Coordinate columns repeat each value across the whole grid — a chunk of
    shape ``(time, lat, lon)`` carries each latitude ``time × lon`` times. A
    dictionary stores only the distinct values plus integer indices, shrinking
    the bytes the engine moves and letting ``GROUP BY`` / equality ``JOIN``
    compare integer keys instead of rehashing repeated floats.

    We only encode when the index type (see :func:`_coord_index_type`) is
    *strictly narrower* than the value type. That keeps 8-byte coordinates
    (``float64``, ``int64``, timestamps) — a 2× win with an overflow-safe int32
    key — while leaving 4-byte ``float32`` / ``int32`` coordinates dense, where a
    dictionary would be pure overhead and a narrower (overflow-prone) key would
    be the only way to win. Variable-width value types (e.g. strings) are always
    encoded — the classic dictionary win. The field's name, nullability, and
    metadata (e.g. cftime units/calendar) are preserved.
    """
    index_type = _coord_index_type(n_values)
    try:
        value_bit_width = field.type.bit_width
    except (ValueError, NotImplementedError):
        value_bit_width = None  # variable-width (e.g. string): always encode
    if value_bit_width is not None and index_type.bit_width >= value_bit_width:
        return field
    return field.with_type(pa.dictionary(index_type, field.type))


def _parse_schema(ds: xr.Dataset) -> pa.Schema:
    """Extracts a `pa.Schema` from the Dataset, treating dims and data_vars as columns.

    Only *dimension coordinates* become dimension columns, so a dimension
    without a coordinate would be dropped. Callers must run the Dataset through
    :func:`_ensure_default_indexes` first (the readers do) so every dimension
    has a coordinate and appears as a column.

    Uses the xarray index type to detect cftime coordinates without
    materializing their data — important for Dask/Zarr-backed datasets
    where .values would trigger eager computation.

    cftime coordinates are mapped to one of two Arrow types:

    * **Gregorian-like calendars** (standard, noleap, all_leap, etc.):
      ``pa.timestamp('us')`` so string-based SQL filters work naturally.
    * **Non-Gregorian calendars** (360_day, julian):
      ``pa.int64()`` with ``xarray:units`` / ``xarray:calendar`` metadata
      on the field, preserving lossless CF-convention encoding.
    """
    columns = []

    for coord_name, coord_var in ds.coords.items():
        # Only include dimension coordinates
        if coord_name in ds.dims:
            if cft.is_cftime_index(ds, coord_name):
                units, calendar = cft.encoding(ds, coord_name)
                field = cft.arrow_field(coord_name, units, calendar)
            else:
                pa_type = pa.from_numpy_dtype(coord_var.dtype)
                field = pa.field(coord_name, pa_type)
            # Dictionary-encode the coordinate column (see _as_dictionary_field),
            # sizing the index type to the dimension length.
            columns.append(_as_dictionary_field(field, ds.sizes[coord_name]))

    for var_name, var in ds.data_vars.items():
        # Data variables are virtually never cftime, but check dtype as a
        # cheap guard.  Only fall back to _is_cftime (which materializes
        # element 0) when dtype is object.
        if var.dtype == np.dtype("O") and cft.is_cftime(var.values):
            # Rare: a data variable holding cftime objects.  Use same encoding
            # as the first cftime dimension coordinate, or default.
            cal = var.values.ravel()[0].calendar
            columns.append(cft.arrow_field(var_name, cft.DEFAULT_UNITS, cal))
        else:
            pa_type = pa.from_numpy_dtype(var.dtype)
            columns.append(pa.field(var_name, pa_type))

    return pa.schema(columns)


# Type alias for partition metadata: maps dimension name to (min, max, dtype_str) values
PartitionBounds = dict[str, tuple[Any, Any, str]]


def _block_metadata(
    coord_arrays: dict,
    block: Block,
    dims: Iterable[Hashable] | None = None,
) -> PartitionBounds:
    """Compute min/max coordinate values for a single partition block.

    Args:
        coord_arrays: Pre-materialised coordinate arrays keyed by dimension name
            string.  Hoist this outside any loop to avoid repeated remote I/O
            for Zarr-backed datasets.
        block: A single block slice dict from block_slices().
        dims: Optional restriction to a subset of dims to compute.  Used by
            ``read_xarray_table`` to skip unchunked dims whose bounds are
            constant across all partitions and have been precomputed once.
            Defaults to all dims present in ``block``.

    Returns:
        Dict mapping dimension name to (min_value, max_value, dtype_str).
        Dimensions with an empty slice are omitted; the Rust pruning logic
        treats missing dimensions conservatively (never prunes on them).
    """
    items = ((d, block[d]) for d in dims) if dims is not None else block.items()
    ranges: PartitionBounds = {}
    for dim, slc in items:
        coord_values = coord_arrays[str(dim)][slc]
        if len(coord_values) == 0:
            continue
        # String/object dtypes are not representable as ScalarBound
        # (Int64/Float64/TimestampNanos) and numpy min/max ufuncs do not
        # support them.  Skip so pruning treats the dimension conservatively.
        if coord_values.dtype.kind in ("U", "S", "O"):
            continue

        if cft.is_cftime(coord_values):
            ranges[str(dim)] = cft.partition_bounds(coord_values)
            continue

        # Use actual min/max rather than first/last so that non-monotonic
        # coordinate axes (e.g. descending latitude 90→-90) are handled
        # correctly.  np.min/max work for both numeric and datetime64 arrays.
        min_val = coord_values.min()
        max_val = coord_values.max()

        if isinstance(min_val, (np.datetime64, pd.Timestamp)):
            min_val = int(pd.Timestamp(min_val).value)
            max_val = int(pd.Timestamp(max_val).value)
            ranges[str(dim)] = (min_val, max_val, "timestamp_ns")
        elif hasattr(min_val, "item"):
            min_val = min_val.item()
            max_val = max_val.item()
            dtype = "float64" if isinstance(min_val, float) else "int64"
            ranges[str(dim)] = (min_val, max_val, dtype)
        else:
            dtype = "float64" if isinstance(min_val, float) else "int64"
            ranges[str(dim)] = (min_val, max_val, dtype)
    return ranges


def partition_metadata(
    ds: xr.Dataset, blocks: list[Block]
) -> list[PartitionBounds]:
    """Compute min/max coordinate values for each partition.

    This metadata enables filter pushdown: SQL queries with WHERE clauses
    on dimension columns can prune partitions that can't contain matching rows.

    Args:
        ds: The xarray Dataset containing coordinate values.
        blocks: List of block slices from block_slices().

    Returns:
        List of dicts mapping dimension name to
        (min_value, max_value, dtype_str) tuples.

            - For datetime64, values are nanoseconds since Unix epoch
              (int64), dtype_str is "timestamp_ns"
            - For numeric types, values are Python int or float,
              dtype_str is "int64" or "float64"

    Note:
        If a partition has an empty slice for a dimension, that dimension is
        omitted from the partition's metadata. The Rust pruning logic treats
        missing dimensions conservatively (never prunes on them).
    """
    # Hoist coordinate array reads outside the partition loop.
    # ds.coords[dim].values materializes the full array on every call; doing it
    # N_partitions × N_dims times is wasteful and, for remote Zarr-backed datasets
    # (e.g. ARCO-ERA5 on GCS), may trigger repeated network I/O.
    coord_arrays = {str(dim): ds.coords[dim].values for dim in ds.dims}
    return [_block_metadata(coord_arrays, block) for block in blocks]
