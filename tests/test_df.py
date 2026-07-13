import tracemalloc

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

from xarray_sql.df import (
    DEFAULT_BATCH_SIZE,
    _parse_schema,
    block_slices,
    compute_chunks,
    dataset_to_record_batch,
    explode,
    from_map,
    from_map_batched,
    iter_record_batches,
    pivot,
)
from xarray_sql.reader import read_xarray, read_xarray_table


def test_explode_cardinality(air):
    dss = explode(air)
    assert len(list(dss)) == np.prod([len(c) for c in air.chunks.values()])


def test_explode_dim_sizes_one(air):
    chunks = {"time": 240}
    ds = next(iter(explode(air)))
    for k, v in chunks.items():
        assert k in ds.dims
        assert v == ds.sizes[k]


def test_explode_data_equal_one_first(air):
    ds = next(iter(explode(air)))
    iselection = {dim: slice(0, s) for dim, s in ds.sizes.items()}
    assert air.isel(iselection).equals(ds)


def test_explode_data_equal_one_last(air):
    dss = list(explode(air))
    ds = dss[-1]

    # For the last chunk, we need to calculate where it actually starts
    # The original logic slice(0, s) only works for the first chunk
    iselection = {}
    for dim in ds.dims:
        # Get chunk boundaries
        chunk_bounds = np.cumsum((0,) + air.chunks[dim])
        # Last chunk index
        last_chunk_idx = len(air.chunks[dim]) - 1
        # Calculate actual start and end positions
        start = chunk_bounds[last_chunk_idx]
        end = chunk_bounds[last_chunk_idx + 1]
        iselection[dim] = slice(start, end)

    assert air.isel(iselection).equals(ds)


def test_block_slices_scalar_dataset_yields_single_block():
    # A dimensionless dataset (e.g. scalar metadata variables) has exactly
    # one block: the whole, empty selection.
    ds = xr.Dataset({"projection": ((), 0)})
    assert list(block_slices(ds)) == [{}]


def test_block_slices_scalar_ignores_irrelevant_chunks():
    ds = xr.Dataset({"projection": ((), 0)})
    assert list(block_slices(ds, chunks={"time": 4})) == [{}]


def test_block_slices_filters_chunk_keys_to_dataset_dims(air_small):
    # A chunk key for a dimension the dataset doesn't have is ignored,
    # rather than raising.
    base = list(block_slices(air_small, chunks={"time": 4, "lat": 3, "lon": 4}))
    extra = list(
        block_slices(
            air_small, chunks={"time": 4, "lat": 3, "lon": 4, "absent": 2}
        )
    )
    assert len(extra) == len(base)


def test_block_slices_dimensional_unchunked_raises():
    # A dataset with dimensions but no chunking is still a user error.
    ds = xr.Dataset({"v": (["x"], np.arange(3))}, coords={"x": np.arange(3)})
    with pytest.raises(AssertionError):
        list(block_slices(ds))


def test_from_map_basic():
    def make_df(x):
        return pd.DataFrame({"value": [x, x * 2], "index": [0, 1]})

    result = from_map(make_df, [1, 2, 3])
    assert isinstance(result, pa.Table)
    assert len(result) == 6
    assert result.column_names == ["value", "index"]


def test_from_map_multiple_iterables():
    def add_values(x, y):
        return pd.DataFrame({"sum": [x + y], "x": [x], "y": [y]})

    result = from_map(add_values, [1, 2], [10, 20])
    assert isinstance(result, pa.Table)
    assert len(result) == 2

    df = result.to_pandas()
    assert list(df["sum"]) == [11, 22]


def test_from_map_with_args():
    def multiply_and_add(x, multiplier, add_value):
        return pd.DataFrame({"result": [x * multiplier + add_value]})

    result = from_map(multiply_and_add, [1, 2, 3], args=(2, 10))
    assert isinstance(result, pa.Table)
    assert len(result) == 3

    df = result.to_pandas()
    assert list(df["result"]) == [12, 14, 16]


def test_from_map_with_pyarrow_tables():
    def make_arrow_table(x):
        df = pd.DataFrame({"value": [x]})
        return pa.Table.from_pandas(df)

    result = from_map(make_arrow_table, [1, 2, 3])
    assert isinstance(result, pa.Table)
    assert len(result) == 3


def test_iter_record_batches_splits_into_multiple_batches(air_small):
    """iter_record_batches should emit >1 batch when partition exceeds batch_size."""
    schema = _parse_schema(air_small)
    block = next(
        block_slices(air_small, chunks={"time": 4, "lat": 3, "lon": 4})
    )
    ds_block = air_small.isel(block)
    total_rows = int(np.prod([ds_block.sizes[d] for d in ds_block.sizes]))

    small_batch = 16  # force many small batches
    batches = list(
        iter_record_batches(ds_block, schema, batch_size=small_batch)
    )

    assert len(batches) == -(-total_rows // small_batch)  # ceiling division
    assert all(b.num_rows <= small_batch for b in batches)
    assert sum(b.num_rows for b in batches) == total_rows


def test_iter_record_batches_matches_dataset_to_record_batch(air_small):
    """Concatenating all iter_record_batches output must equal dataset_to_record_batch."""
    schema = _parse_schema(air_small)
    dim_cols = [f.name for f in schema if f.name in air_small.dims]
    block = next(
        block_slices(air_small, chunks={"time": 4, "lat": 3, "lon": 4})
    )
    ds_block = air_small.isel(block)

    batches = list(iter_record_batches(ds_block, schema, batch_size=16))
    actual_df = (
        pa.Table.from_batches(batches)
        .to_pandas()
        .sort_values(dim_cols)
        .reset_index(drop=True)
    )
    expected_df = (
        dataset_to_record_batch(ds_block, schema)
        .to_pandas()
        .sort_values(dim_cols)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual_df, expected_df)


def test_iter_record_batches_default_batch_size():
    """A single-batch partition (rows <= DEFAULT_BATCH_SIZE) yields exactly one batch."""
    ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(0, 2))
    schema = _parse_schema(ds)
    total_rows = int(np.prod([ds.sizes[d] for d in ds.sizes]))
    assert total_rows <= DEFAULT_BATCH_SIZE, "fixture too large — adjust isel"
    batches = list(iter_record_batches(ds, schema))
    assert len(batches) == 1
    assert batches[0].num_rows == total_rows


def test_dataset_to_record_batch_matches_pivot(air_small):
    """dataset_to_record_batch should contain the same rows as pivot.

    Row ordering may differ (pivot uses ds.dims key order; dataset_to_record_batch
    uses the data variable's own dim order). Both orderings are valid for SQL, so
    we sort by the coordinate columns before comparing.
    """
    schema = _parse_schema(air_small)
    dim_cols = [f.name for f in schema if f.name in air_small.dims]
    blocks = list(
        block_slices(air_small, chunks={"time": 4, "lat": 3, "lon": 4})
    )

    for block in blocks:
        ds_block = air_small.isel(block)
        actual_df = (
            dataset_to_record_batch(ds_block, schema)
            .to_pandas()
            .sort_values(dim_cols)
            .reset_index(drop=True)
        )
        expected_df = (
            pa.RecordBatch.from_pandas(pivot(ds_block), schema=schema)
            .to_pandas()
            .sort_values(dim_cols)
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(actual_df, expected_df, check_like=False)


def test_dataset_to_record_batch_column_order(air_small):
    """Output column order must match schema (dims first, then data vars)."""
    schema = _parse_schema(air_small)
    block = next(
        block_slices(air_small, chunks={"time": 4, "lat": 3, "lon": 4})
    )
    batch = dataset_to_record_batch(air_small.isel(block), schema)
    assert batch.schema.names == schema.names


def test_dataset_to_record_batch_row_count(air_small):
    """Row count must equal the product of the block dimension sizes."""
    schema = _parse_schema(air_small)
    chunks = {"time": 4, "lat": 3, "lon": 4}
    for block in block_slices(air_small, chunks=chunks):
        ds_block = air_small.isel(block)
        expected_rows = int(
            np.prod([ds_block.sizes[d] for d in ds_block.sizes])
        )
        batch = dataset_to_record_batch(ds_block, schema)
        assert batch.num_rows == expected_rows


def test_from_map_batched_basic_functionality(air_small):
    blocks = list(
        block_slices(air_small, chunks={"time": 4, "lat": 3, "lon": 4})
    )

    first_block_df = pivot(air_small.isel(blocks[0]))
    expected_schema = pa.Schema.from_pandas(first_block_df)

    reader = from_map_batched(
        pivot,
        [air_small.isel(block) for block in blocks],
        schema=expected_schema,
    )

    assert isinstance(reader, pa.RecordBatchReader)
    assert reader.schema == expected_schema

    batches = list(reader)
    assert len(batches) > 0
    for batch in batches:
        assert batch.schema == expected_schema
        assert len(batch) > 0


def adding_function(x, y):
    """Simple function that adds two values and returns a DataFrame."""
    result = pd.DataFrame({"x": [x], "y": [y], "sum": [x + y]})
    return result


def test_from_map_batched_multiple_iterables():
    x_values = [1, 2, 3, 4, 5]
    y_values = [10, 20, 30, 40, 50]

    expected_schema = pa.schema(
        [("x", pa.int64()), ("y", pa.int64()), ("sum", pa.int64())]
    )

    reader = from_map_batched(
        adding_function, x_values, y_values, schema=expected_schema
    )
    table = reader.read_all()
    df = table.to_pandas()

    expected_df = pd.DataFrame(
        {
            "x": x_values,
            "y": y_values,
            "sum": [x + y for x, y in zip(x_values, y_values)],
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)


def test_from_map_batched_with_args_and_kwargs():
    def multiply_and_add(x, multiplier, offset=0):
        return pd.DataFrame({"x": [x], "result": [x * multiplier + offset]})

    values = [1, 2, 3]
    expected_schema = pa.schema([("x", pa.int64()), ("result", pa.int64())])

    reader = from_map_batched(
        multiply_and_add, values, args=(2,), offset=5, schema=expected_schema
    )
    table = reader.read_all()
    df = table.to_pandas()

    expected_df = pd.DataFrame({"x": [1, 2, 3], "result": [7, 9, 11]})
    pd.testing.assert_frame_equal(df, expected_df)


def test_from_map_batched_empty_iterables():
    empty_schema = pa.schema([("value", pa.int64())])

    reader = from_map_batched(
        lambda x: pd.DataFrame({"value": [x]}), [], schema=empty_schema
    )
    batches = list(reader)
    assert len(batches) == 0


def test_from_map_batched_consistency_with_regular_map(air_small):
    blocks = list(block_slices(air_small, chunks={"time": 4, "lat": 3}))
    datasets = [air_small.isel(block) for block in blocks]

    first_df = pivot(datasets[0])
    schema = pa.Schema.from_pandas(first_df)

    reader = from_map_batched(pivot, datasets, schema=schema)
    batched_table = reader.read_all()

    regular_dfs = [pivot(ds) for ds in datasets]
    regular_table = pa.Table.from_pandas(
        pd.concat(regular_dfs, ignore_index=True)
    )

    assert batched_table.schema == regular_table.schema
    assert len(batched_table) == len(regular_table)

    batched_df = (
        batched_table.to_pandas()
        .sort_values(["time", "lat", "lon"])
        .reset_index(drop=True)
    )
    regular_df = (
        regular_table.to_pandas()
        .sort_values(["time", "lat", "lon"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(batched_df, regular_df)


def test_from_map_batched_integration_with_datafusion_via_read_xarray():
    air = xr.tutorial.open_dataset("air_temperature")
    air_small = air.isel(time=slice(0, 50), lat=slice(0, 10), lon=slice(0, 15))
    air_chunked = air_small.chunk({"time": 25, "lat": 5, "lon": 8})

    arrow_stream = read_xarray(
        air_chunked, chunks={"time": 25, "lat": 5, "lon": 8}
    )

    assert hasattr(arrow_stream, "schema")
    assert hasattr(arrow_stream, "__iter__")

    table = arrow_stream.read_all()
    assert len(table) > 0

    expected_columns = {"time", "lat", "lon", "air"}
    actual_columns = set(table.column_names)
    assert expected_columns.issubset(actual_columns)


def test_read_xarray_loads_one_chunk_at_a_time(large_ds):
    tracemalloc.stop()  # reset any state left by a previously-failed test
    tracemalloc.start()
    try:
        iterable = read_xarray(large_ds)
        first_size, first_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

        sizes, peaks = [], []

        first_chunk = large_ds.isel(next(block_slices(large_ds)))
        chunk_size = first_chunk.nbytes

        # Creating the iterator should be inexpensive -- less than one chunk.
        # We multiply by constant factors because chunks have additional overhead
        assert first_size < chunk_size * 3
        assert first_peak < chunk_size * 6

        for it in iterable:
            _ = it
            cur_size, cur_peak = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            sizes.append(cur_size)
            peaks.append(cur_peak)

        for size in sizes:
            # iter_record_batches' whole-partition fast path holds the
            # data-variable arrays (≈1× chunk) plus repeat/tile-expanded
            # coordinate columns (n_dims × 8 bytes × rows, ≈1.5× chunk
            # for this 3-dim float64 dataset) for the partition being
            # streamed; batches themselves are zero-copy slices.
            assert chunk_size * 1.3 < size, f"size {size} unexpectedly low"
            assert chunk_size * 4.0 > size, f"size {size} unexpectedly high"

        for peak in peaks:
            # Peak adds transient buffers on top of the steady state:
            # np.repeat/np.tile intermediates for the coordinate columns
            # and Arrow's from_pandas null scan; the first batch of each
            # chunk is highest (Dask compute overhead). Observed ~5.04×
            # on macOS.
            assert chunk_size * 1.5 < peak, f"peak {peak} unexpectedly low"
            assert chunk_size * 6.5 > peak, f"peak {peak} unexpectedly high"

        assert max(peaks) < large_ds.nbytes
    finally:
        tracemalloc.stop()


def test_read_xarray_table_memory_bounds(large_ds):
    """read_xarray_table should not materialise data at registration time.

    Registration should only hold coordinate arrays and Rust partition metadata
    (no data variables).  Peak memory during a full-table query should be a
    small fraction of the whole dataset — i.e. partitions are processed without
    loading all of them simultaneously.
    """
    from datafusion import SessionContext

    first_chunk = large_ds.isel(next(block_slices(large_ds)))
    chunk_size = first_chunk.nbytes

    tracemalloc.stop()  # reset any state left by a previously-failed test
    # --- Registration phase ---
    tracemalloc.start()
    try:
        table = read_xarray_table(large_ds)
        reg_size, reg_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

        # The lazy generator only materialises coord arrays (~O(dim sizes)) and
        # factory closure objects — no data arrays.  Both metrics should be well
        # below one chunk of data.
        assert reg_size < chunk_size, (
            f"Registration held {reg_size} bytes >= chunk_size {chunk_size}: "
            "data may have been loaded eagerly"
        )
        assert reg_peak < chunk_size * 2, (
            f"Registration peak {reg_peak} too high (expected < 2× chunk_size {chunk_size})"
        )

        # --- Query phase ---
        ctx = SessionContext()
        ctx.register_table("weather", table)
        ctx.sql(
            "SELECT AVG(temperature), AVG(precipitation) FROM weather"
        ).collect()
        _, query_peak = tracemalloc.get_traced_memory()

        # tracemalloc measures Python-heap allocations, which include Arrow
        # buffer copies and object overhead on top of the raw data.  The
        # observed peak is typically 1.1–1.5× the raw dataset size; we use
        # 2× as a generous bound that would still catch catastrophic regressions
        # (e.g. loading all partitions twice simultaneously).
        assert query_peak < large_ds.nbytes * 2, (
            f"Query peak {query_peak} >= 2× dataset {large_ds.nbytes}: "
            "may be holding excessive data in memory"
        )
    finally:
        tracemalloc.stop()


# ---------------------------------------------------------------------------
# compute_chunks: arithmetic replacement for ds.chunk(...).chunks.
# Dask serves as the source of truth.
# ---------------------------------------------------------------------------


def _dask_chunks(ds: xr.Dataset, chunks: dict) -> dict:
    rechunked = ds.copy(data=None, deep=False).chunk(chunks)
    return {str(k): tuple(v) for k, v in rechunked.chunks.items()}


def _normalise(result: dict) -> dict:
    return {str(k): tuple(v) for k, v in result.items()}


def _simple_ds(shape: tuple[int, ...], dims: tuple[str, ...]) -> xr.Dataset:
    return xr.Dataset(
        {"v": (dims, np.zeros(shape))},
        coords={d: np.arange(s) for d, s in zip(dims, shape)},
    )


@pytest.mark.parametrize(
    "ds,chunks",
    [
        # Even divide on a single dim.
        (_simple_ds((10,), ("x",)), {"x": 5}),
        # Uneven divide: trailing remainder chunk.
        (_simple_ds((10,), ("x",)), {"x": 3}),
        # Requested chunk size larger than the dim → single chunk.
        (_simple_ds((5,), ("x",)), {"x": 100}),
        # Multi-dim spec with a dim left unspecified (kept as one chunk).
        (_simple_ds((4, 6), ("x", "y")), {"x": 2}),
        # Multi-dim spec rechunking every dim.
        (_simple_ds((7, 11, 13), ("a", "b", "c")), {"a": 3, "b": 4, "c": 5}),
    ],
)
def test_compute_chunks_matches_dask(ds, chunks):
    assert _normalise(compute_chunks(ds, chunks)) == _dask_chunks(ds, chunks)


def test_compute_chunks_preserves_existing_dask_chunking():
    # When the dataset is already dask-backed, rechunking one dim must
    # leave other dims' existing chunk tuples alone.
    ds = _simple_ds((4, 5), ("x", "y")).chunk({"x": 1, "y": 2})
    chunks = {"x": 2}
    assert _normalise(compute_chunks(ds, chunks)) == _dask_chunks(ds, chunks)


def test_compute_chunks_tuples_sum_to_dim_size():
    # Dask-independent invariant: every per-dim chunk tuple must fully
    # cover its dimension.
    ds = _simple_ds((7, 11, 13), ("a", "b", "c"))
    result = compute_chunks(ds, {"a": 3, "b": 4, "c": 5})
    for dim, tup in result.items():
        assert sum(tup) == ds.sizes[dim]
