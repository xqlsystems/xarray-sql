"""Lazy Arrow stream reader for xarray Datasets.

This module provides XarrayRecordBatchReader, which implements the Arrow
PyCapsule Interface (__arrow_c_stream__) to enable zero-copy, lazy streaming
of xarray data to DataFusion and other Arrow consumers.

The implementation delegates to PyArrow's RecordBatchReader for the
actual stream implementation, wrapping xarray block iteration in a generator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import xarray as xr

from .df import (
    Block,
    Chunks,
    DEFAULT_BATCH_SIZE,
    _block_len,
    _block_metadata,
    _block_slices_from_resolved,
    _ensure_default_indexes,
    _parse_schema,
    block_slices,
    iter_record_batches,
    resolve_chunks,
)

if TYPE_CHECKING:
    from ._native import LazyArrowStreamTable


class XarrayRecordBatchReader:
    """A lazy Arrow stream reader for xarray Datasets.

    Implements the Arrow PyCapsule Interface (__arrow_c_stream__) to enable
    zero-copy, lazy streaming of xarray data to DataFusion and other Arrow
    consumers.

    The key property is that xarray blocks are only converted to Arrow
    RecordBatches when the consumer calls get_next (e.g., during DataFusion's
    collect()), NOT when the reader is created or registered.

    Attributes:
        schema: The Arrow schema for the stream.

    Example:
        >>> import xarray as xr
        >>> from xarray_sql import XarrayRecordBatchReader
        >>> ds = xr.tutorial.open_dataset('air_temperature')
        >>> reader = XarrayRecordBatchReader(ds, chunks={'time': 240})
        >>> # At this point, NO data has been read from xarray
        >>> # Data is only read when consumed:
        >>> import pyarrow as pa
        >>> pa_reader = pa.RecordBatchReader.from_stream(reader)
        >>> for batch in pa_reader:
        ...     print(batch.num_rows)  # Data read here
    """

    def __init__(
        self,
        ds: xr.Dataset,
        chunks: Chunks = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        _iteration_callback: (
            Callable[[Block, list[str] | None], None] | None
        ) = None,
    ):
        """Initialize the lazy reader.

        Args:
            ds: An xarray Dataset. All data_vars must share the same dimensions.
            chunks: Xarray-like chunks specification. If not provided, uses
                the Dataset's existing chunks.
            batch_size: Maximum rows per emitted Arrow RecordBatch.  Smaller
                values let DataFusion start processing earlier at the cost of
                more Python→Arrow conversion calls.
            _iteration_callback: Internal callback for testing. Called with
                each block dict just before it's converted to Arrow. This
                allows tests to track when iteration actually occurs.
        """
        self._ds = ds
        self._chunks = chunks
        self._batch_size = batch_size
        self._schema = _parse_schema(ds)
        self._iteration_callback = _iteration_callback
        self._consumed = False

        # Validate dimensions
        fst = next(iter(ds.values())).dims
        if not all(da.dims == fst for da in ds.values()):
            raise ValueError(
                "All dimensions must be equal. Please filter data_vars in the Dataset."
            )

    @property
    def schema(self) -> pa.Schema:
        """The Arrow schema for this stream."""
        return self._schema

    def _generate_batches(self) -> Iterator[pa.RecordBatch]:
        """Generate RecordBatches lazily from xarray blocks.

        This generator is only consumed when the Arrow stream's get_next
        is called, ensuring true lazy evaluation.  Each xarray block is
        emitted as one or more RecordBatches of at most self._batch_size rows.
        """
        for block in block_slices(self._ds, self._chunks):
            # Call the iteration callback if provided (for testing).
            # XarrayRecordBatchReader has no projection concept, so always passes None.
            if self._iteration_callback is not None:
                self._iteration_callback(block, None)

            yield from iter_record_batches(
                self._ds.isel(block), self._schema, self._batch_size
            )

    def __arrow_c_stream__(
        self, requested_schema: object | None = None
    ) -> object:
        """Export as Arrow C Stream via PyCapsule.

        This method is called by Arrow consumers (like DataFusion) to get
        a C-level stream interface. The actual data iteration only begins
        when the consumer calls get_next on the stream.

        Args:
            requested_schema: Optional schema for type casting. Currently
                passed through to PyArrow's implementation.

        Returns:
            PyCapsule containing ArrowArrayStream pointer with name
            "arrow_array_stream".

        Raises:
            RuntimeError: If the stream has already been consumed.
        """
        if self._consumed:
            raise RuntimeError(
                "Stream already consumed. XarrayRecordBatchReader can only "
                "be iterated once. Create a new reader for additional iterations."
            )
        self._consumed = True

        # Create a PyArrow RecordBatchReader from our generator
        # The generator is NOT consumed here - only when get_next is called
        reader = pa.RecordBatchReader.from_batches(
            self._schema, self._generate_batches()
        )

        # Delegate to PyArrow's __arrow_c_stream__ implementation
        return reader.__arrow_c_stream__(requested_schema)

    def __arrow_c_schema__(
        self, requested_schema: object | None = None
    ) -> object:
        """Export the schema as Arrow C Schema via PyCapsule.

        This allows consumers to inspect the schema without consuming the stream.

        Args:
            requested_schema: Optional schema for negotiation (unused).

        Returns:
            PyCapsule containing ArrowSchema pointer.
        """
        return self._schema.__arrow_c_schema__()


def read_xarray(ds: xr.Dataset, chunks: Chunks = None) -> pa.RecordBatchReader:
    """Pivots an Xarray Dataset into a PyArrow Table, partitioned by chunks.

    Args:
      ds: An Xarray Dataset. All `data_vars` must share the same dimensions.
      chunks: Xarray-like chunks. If not provided, will default to the
        Dataset's chunks. The product of the chunk sizes becomes the
        standard length of each dataframe partition.

    Returns:
      A PyArrow RecordBatchReader, which is a table representation of the input
      Dataset.
    """
    ds = _ensure_default_indexes(ds)
    reader = XarrayRecordBatchReader(ds, chunks=chunks)
    return pa.RecordBatchReader.from_stream(reader)


def read_xarray_table(
    ds: xr.Dataset,
    chunks: Chunks = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    coord_arrays: dict[str, np.ndarray] | None = None,
    _iteration_callback: (
        Callable[[Block, list[str] | None], None] | None
    ) = None,
) -> "LazyArrowStreamTable":
    """Create a lazy DataFusion table from an xarray Dataset.

    This is the simplest way to register xarray data with DataFusion.
    Data is only read when queries are executed, not during registration.
    The table can be queried multiple times.

    Each chunk becomes a separate partition, enabling DataFusion's parallel
    execution across multiple cores.

    Note:
        SQL queries with WHERE clauses on dimension columns (time, lat, lon, etc.)
        automatically prune partitions that can't contain matching rows — this is
        called *filter pushdown*. For example:

            # This query will skip loading partitions with time < '2020-02-01'
            result = ctx.sql('SELECT * FROM air WHERE time > "2020-02-01"').collect()

        Supported operators: `=`, `<`, `>`, `<=`, `>=`, `BETWEEN`, `IN`, `AND`, `OR`.

    Args:
        ds: An xarray Dataset. All data_vars must share the same dimensions.
        chunks: Xarray-like chunks specification. If not provided, uses
            the Dataset's existing chunks.
        batch_size: Maximum rows per Arrow RecordBatch emitted per partition.
            Smaller values let DataFusion start processing earlier; the default
            (65 536) works well for most datasets.
        coord_arrays: Pre-materialised coordinate arrays keyed by dim-name
            string.  Hand in to share a single read across multiple tables
            built from the same parent Dataset (e.g. surface + atmosphere
            from ARCO-ERA5); the dim coords are otherwise read once per
            ``read_xarray_table`` call, which is a network round-trip for
            Zarr-backed datasets.
        _iteration_callback: Internal callback for testing. Called with
            each block dict just before it's converted to Arrow.

    Returns:
        A LazyArrowStreamTable ready for registration with DataFusion.

    Example:
        >>> from datafusion import SessionContext
        >>> import xarray as xr
        >>> from xarray_sql import read_xarray_table
        >>>
        >>> ds = xr.tutorial.open_dataset('air_temperature')
        >>> table = read_xarray_table(ds, chunks={'time': 240})
        >>>
        >>> ctx = SessionContext()
        >>> ctx.register_table('air', table)
        >>>
        >>> # Data is only read here, during query execution
        >>> # Filters on 'time' will prune partitions automatically!
        >>> result = ctx.sql('SELECT AVG(air) FROM air').collect()
    """
    from ._native import LazyArrowStreamTable

    ds = _ensure_default_indexes(ds)
    schema = _parse_schema(ds)

    # Hoist coordinate reads once; avoids N_partitions remote I/O calls for
    # Zarr-backed datasets (e.g. ARCO-ERA5 on GCS).  When the caller supplies
    # pre-materialised arrays (e.g. shared across surface + atmosphere
    # tables), reuse them and skip the extra read.
    if coord_arrays is None:
        coord_arrays = {str(dim): ds.coords[dim].values for dim in ds.dims}

    # Determine which column names are data variables (not dimension coordinates).
    # Used by the factory to skip loading unrequested variables.
    data_var_names = set(ds.data_vars.keys())

    def make_partition_factory(
        block: Block,
    ) -> Callable[[list[str] | None], pa.RecordBatchReader]:
        def make_stream(
            projection_names: list[str] | None,
        ) -> pa.RecordBatchReader:
            if _iteration_callback is not None:
                _iteration_callback(block, projection_names)

            if projection_names is not None:
                # Restrict to the data variables mentioned in the projection.
                # Dimension coordinates come along automatically via coords.
                data_vars_needed = [
                    c for c in projection_names if c in data_var_names
                ]
                if data_vars_needed:
                    ds_block = ds[data_vars_needed].isel(block)
                else:
                    # Only dimension coords requested — drop all data vars to avoid
                    # loading them unnecessarily (e.g. for queries like SELECT lat, lon).
                    ds_block = ds.drop_vars(list(ds.data_vars)).isel(block)
                batch_schema = pa.schema(
                    [schema.field(name) for name in projection_names]
                )
            else:
                ds_block = ds.isel(block)
                batch_schema = schema

            return pa.RecordBatchReader.from_batches(
                batch_schema,
                iter_record_batches(ds_block, batch_schema, batch_size),
            )

        return make_stream

    # Separate dims whose chunk bounds vary across partitions from those
    # whose bounds are constant (one chunk spanning the whole axis).  For the
    # latter we compute min/max once instead of re-scanning the full coord
    # array on every partition — dominant cost when registering hundreds of
    # thousands of single-time-step partitions on a 4-D dataset like ERA5.
    resolved = resolve_chunks(ds, chunks)
    varying_dims = [d for d, tup in resolved.items() if len(tup) > 1]
    static_dims = [d for d in ds.dims if d not in varying_dims]
    static_block: Block = {d: slice(None) for d in static_dims}
    static_ranges = _block_metadata(
        coord_arrays, static_block, dims=static_dims
    )

    def partition_pairs():
        """Lazily yield (factory, metadata, num_rows) for each partition.

        Consuming this generator one item at a time means Python never holds
        all N block dicts, metadata dicts, and factory closures simultaneously.
        Peak Python memory during registration is O(1) per partition instead
        of O(N_partitions).
        """
        for block in _block_slices_from_resolved(ds, resolved):
            dynamic = _block_metadata(coord_arrays, block, dims=varying_dims)
            yield (
                make_partition_factory(block),
                {**static_ranges, **dynamic},
                # Exact row count for this partition (product of the chunk's
                # per-dimension sizes), so the scan can report exact
                # Statistics::num_rows to the optimizer.
                _block_len(block),
            )

    return LazyArrowStreamTable(partition_pairs(), schema)
