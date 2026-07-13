"""DuckDB engine adapter with source-level filter and projection pushdown.

Registers a lazy ``xarray.Dataset`` on a ``duckdb.DuckDBPyConnection`` as
a :class:`XarrayPushdownDataset` — a ``pyarrow.dataset.Dataset`` subclass
in the same pattern Lance uses for its ``LanceDataset``. DuckDB
classifies the object with a real ``isinstance`` check against
``pyarrow.dataset.Dataset`` and calls ``scanner(columns=[...],
filter=<pyarrow.compute.Expression>)`` once per query, which lets the
adapter:

* **push projection to the source** — only the data variables a query
  mentions are loaded from storage;
* **prune chunks** — per-dimension shadow ``FileSystemDataset`` fragments
  carry each chunk's coordinate range as a ``partition_expression``, so
  Arrow's own guarantee simplification decides which chunks can match the
  pushed predicate (sound for every predicate shape, no expression
  parsing on our side);
* **parallelize production** — surviving chunks are loaded by a bounded
  thread pool ahead of the consumer.

Correctness contract: DuckDB deletes the filter conjuncts it pushes down
and does **not** re-apply them to returned batches, so the pushed
expression must be applied exactly. The scanner therefore hands the
expression to ``pyarrow.dataset.Scanner.from_batches(..., filter=...)``,
which row-filters exactly; pruning is only ever an optimization on top.

This adapter never imports the ``duckdb`` package — detection is by
connection type, and registration is a method call on the connection —
so DuckDB stays a purely optional dependency
(``pip install xarray-sql[duckdb]``).

Zarr-native scanning inside DuckDB is what the `duckdb-zarr
<https://github.com/xqlsystems/duckdb-zarr>`_ extension provides; this
adapter instead covers everything xarray can open (NetCDF, GRIB, Xee, CF
decoding, in-memory) and pairs with :func:`xarray_sql.to_dataset` for
the labeled round-trip.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.fs as pafs
import xarray as xr

from ..df import (
    Block,
    Chunks,
    DEFAULT_BATCH_SIZE,
    _ensure_default_indexes,
    _parse_schema,
    iter_record_batches,
    resolve_chunks,
)
from ..reader import XarrayRecordBatchReader
from ..sql import _group_vars_by_dims
from .base import register_adapter

DEFAULT_PREFETCH = 4
"""Chunk loads kept in flight ahead of the consumer during a scan."""

_SHADOW_FANOUT = 1024
"""Maximum fragments per shadow level.

A dimension with more chunks than this gets a two-level shadow: a coarse
level of at most this many buckets, refined per surviving bucket. This
bounds shadow construction cost for finely partitioned datasets (e.g.
hundreds of thousands of single-step time chunks) at registration and
query time alike.
"""

_REFINE_MAX_FRACTION = 0.25
"""Skip fine-level pruning when the coarse pass kept more buckets.

Refinement builds one sub-shadow per surviving bucket; when a predicate
matches most of the axis that cost cannot pay for itself, so the scan
falls back to the (sound) coarse answer.
"""


class _DimShadow:
    """Chunk-pruning index for one dimension of the source grid.

    Fragment ``i`` of a shadow ``FileSystemDataset`` carries the
    guarantee ``dim ∈ [min, max]`` of chunk-span ``i`` as its
    ``partition_expression``; ``get_fragments(filter=...)`` then lets
    Arrow's guarantee simplification decide which spans can satisfy a
    predicate — sound for every predicate shape, conservative on columns
    the guarantee does not mention, and the fragments' paths are never
    opened.

    Axes with more than ``_SHADOW_FANOUT`` chunks use two levels: a
    coarse shadow over buckets of consecutive chunks, plus per-bucket
    fine shadows built lazily for the buckets a query keeps.
    """

    def __init__(
        self,
        name: str,
        schema: pa.Schema,
        coord: np.ndarray,
        bounds: np.ndarray,
    ):
        self._name = name
        # The full table schema, not just this dimension's field: the
        # pushed predicate may reference any column, and get_fragments
        # must be able to bind all of them (guarantees stay per-dim;
        # unmentioned columns are conservatively unconstrained).
        self._schema = schema
        self._field_type = schema.field(name).type
        self._coord = coord
        self._bounds = bounds
        self._n = len(bounds) - 1
        self._step = max(1, math.ceil(self._n / _SHADOW_FANOUT))
        self._n_buckets = math.ceil(self._n / self._step)
        self._coarse = self._build(
            [
                (b * self._step, min((b + 1) * self._step, self._n))
                for b in range(self._n_buckets)
            ]
        )
        self._fine: dict[int, pads.FileSystemDataset] = {}

    def _build(self, spans: list[tuple[int, int]]) -> pads.FileSystemDataset:
        """A shadow whose fragment ``i`` guarantees chunk-span ``spans[i]``."""
        fmt = pads.IpcFileFormat()
        fs = pafs.LocalFileSystem()
        fragments = []
        for i, (lo_chunk, hi_chunk) in enumerate(spans):
            vals = self._coord[self._bounds[lo_chunk] : self._bounds[hi_chunk]]
            # min/max (not first/last) so descending axes like latitude
            # 90→-90 carry correct ranges.
            lo = pa.scalar(vals.min(), type=self._field_type)
            hi = pa.scalar(vals.max(), type=self._field_type)
            guarantee = (pc.field(self._name) >= lo) & (
                pc.field(self._name) <= hi
            )
            fragments.append(
                fmt.make_fragment(str(i), fs, partition_expression=guarantee)
            )
        return pads.FileSystemDataset(fragments, self._schema, fmt, fs)

    @staticmethod
    def _kept_indices(
        shadow: pads.FileSystemDataset, filter: pc.Expression
    ) -> list[int]:
        return sorted(
            int(frag.path) for frag in shadow.get_fragments(filter=filter)
        )

    def kept(self, filter: pc.Expression) -> list[int] | None:
        """Chunk indices that can satisfy ``filter``; ``None`` means all."""
        try:
            buckets = self._kept_indices(self._coarse, filter)
            if self._step == 1:
                return buckets if len(buckets) < self._n else None
            if len(buckets) > _REFINE_MAX_FRACTION * self._n_buckets:
                # Refining most of the axis costs more than it saves;
                # answer with the coarse buckets, which is still sound.
                return (
                    None
                    if len(buckets) == self._n_buckets
                    else [
                        i
                        for b in buckets
                        for i in range(
                            b * self._step,
                            min((b + 1) * self._step, self._n),
                        )
                    ]
                )
            kept: list[int] = []
            for b in buckets:
                fine = self._fine.get(b)
                if fine is None:
                    start = b * self._step
                    stop = min((b + 1) * self._step, self._n)
                    fine = self._build([(i, i + 1) for i in range(start, stop)])
                    self._fine[b] = fine
                start = b * self._step
                kept.extend(start + i for i in self._kept_indices(fine, filter))
            return kept
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError):
            return None  # conservative: scan every chunk of this dim


class XarrayArrowStream:
    """A re-scannable Arrow C-stream view over a lazy xarray Dataset.

    Arrow PyCapsule consumers (DuckDB among them) call
    ``__arrow_c_stream__`` once per scan. Each call constructs a fresh
    :class:`~xarray_sql.reader.XarrayRecordBatchReader` over the same
    lazy Dataset, so — unlike registering a ``pyarrow.RecordBatchReader``
    directly, which is exhausted after one query — the same registered
    table supports any number of queries, and data is only read while a
    query is executing.

    The PyCapsule scan path gets no source-level pushdown (the producer
    never sees the query's columns or filters), so
    :class:`XarrayPushdownDataset` is the default registration object;
    this class remains as the dependency-light fallback.
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
        # Validate eagerly (same checks XarrayRecordBatchReader runs) so
        # registration fails fast instead of erroring mid-query.
        probe = XarrayRecordBatchReader(ds, chunks, batch_size=batch_size)
        self._ds = ds
        self._chunks = chunks
        self._batch_size = batch_size
        self._schema = probe.schema
        self._iteration_callback = _iteration_callback

    def __arrow_c_stream__(
        self, requested_schema: object | None = None
    ) -> object:
        reader = XarrayRecordBatchReader(
            self._ds,
            self._chunks,
            batch_size=self._batch_size,
            _iteration_callback=self._iteration_callback,
        )
        return reader.__arrow_c_stream__(requested_schema)

    def __arrow_c_schema__(self) -> object:
        return self._schema.__arrow_c_schema__()


class XarrayPushdownDataset(pads.Dataset):
    """A pushdown-capable ``pyarrow.dataset.Dataset`` view of a Dataset.

    Consumers that speak the pyarrow dataset protocol (DuckDB, Polars,
    ...) call :meth:`scanner` with the columns a query needs and the
    predicate it pushed down; the scan then loads only the needed data
    variables from only the chunks whose coordinate ranges can satisfy
    the predicate.

    The base class is never initialized (there is no C++ dataset behind
    this object — the same construction Lance uses for ``LanceDataset``);
    every entry point consumers touch is overridden in Python, and the
    few inherited members that would read uninitialized native state are
    stubbed out.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        chunks: Chunks = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        prefetch: int = DEFAULT_PREFETCH,
        coord_arrays: dict[str, np.ndarray] | None = None,
        _iteration_callback: (
            Callable[[Block, list[str] | None], None] | None
        ) = None,
    ):
        # Deliberately no super().__init__() — see class docstring.
        ds = _ensure_default_indexes(ds)
        if ds.data_vars:
            fst = next(iter(ds.values())).dims
            if not all(da.dims == fst for da in ds.values()):
                raise ValueError(
                    "All dimensions must be equal. "
                    "Please filter data_vars in the Dataset."
                )
        self._ds = ds
        self._schema = _parse_schema(ds)
        self._resolved = resolve_chunks(ds, chunks)
        if not self._resolved and ds.sizes:
            raise ValueError(
                "Dataset `ds` must be chunked or `chunks` must be provided."
            )
        self._chunk_bounds = {
            d: np.cumsum((0, *sizes)) for d, sizes in self._resolved.items()
        }
        # Reuse pre-materialised coordinate arrays where the caller has
        # them (e.g. shared across the tables of a dim-group split); each
        # missing dim costs one read, a network round-trip for Zarr.
        self._coord_arrays = dict(coord_arrays or {})
        for d in ds.dims:
            if str(d) not in self._coord_arrays:
                self._coord_arrays[str(d)] = ds.coords[d].values
        self._batch_size = batch_size
        self._prefetch = prefetch
        self._iteration_callback = _iteration_callback
        self._shadows: dict[str, _DimShadow] | None = None

    # ------------------------------------------------------------------
    # The consumer-facing surface
    # ------------------------------------------------------------------

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def scanner(
        self,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
        **kwargs: Any,
    ) -> pads.Scanner:
        """Build a scanner for the requested columns and predicate.

        ``filter`` is applied exactly by the returned scanner (DuckDB
        deletes the conjuncts it pushes down and trusts the source to
        enforce them); chunk pruning and column selection only reduce
        how much data is read to get there. Extra keyword arguments from
        other pyarrow-dataset consumers are accepted and ignored.
        """
        proj = list(columns) if columns else list(self._schema.names)
        scan_names = self._scan_columns(proj, filter)
        scan_schema = pa.schema([self._schema.field(n) for n in scan_names])
        kept = None if filter is None else self._prune(filter)
        batches = self._batch_generator(scan_schema, kept)
        return pads.Scanner.from_batches(
            batches, schema=scan_schema, columns=proj, filter=filter
        )

    # Inherited convenience methods (to_table, head, count_rows,
    # to_batches, take) route through scanner() and keep working; the
    # members below would touch the uninitialized native dataset.

    @property
    def partition_expression(self) -> pc.Expression:
        # The dataset-level guarantee: trivially true. The base class
        # getter reads native state this object does not have.
        return pc.scalar(True)

    def get_fragments(self, filter: pc.Expression | None = None):
        raise NotImplementedError(
            "XarrayPushdownDataset is not fragment-based; use scanner()."
        )

    def filter(self, expression: pc.Expression):
        raise NotImplementedError(
            "Use scanner(filter=...) or the engine's WHERE clause."
        )

    def replace_schema(self, schema: pa.Schema):
        raise NotImplementedError

    def sort_by(self, sorting, **kwargs):
        raise NotImplementedError

    def join(self, *args, **kwargs):
        raise NotImplementedError

    def join_asof(self, *args, **kwargs):
        raise NotImplementedError

    def __reduce__(self):
        raise TypeError("XarrayPushdownDataset is not picklable.")

    # ------------------------------------------------------------------
    # Projection: which columns must be read
    # ------------------------------------------------------------------

    def _scan_columns(
        self, proj: list[str], filter: pc.Expression | None
    ) -> list[str]:
        """Columns to read: the projection plus any the filter references.

        The consumer's column list need not include filter-only columns
        (DuckDB drops pushed conjuncts from its plan and has no upstream
        use for them). Rather than parsing the expression, probe it
        against an empty table and grow the column set from the "no match
        for field" errors until it evaluates; on anything unexpected fall
        back to scanning every column, which is always correct.
        """
        if filter is None:
            return proj
        wanted = set(proj)
        for _ in range(len(self._schema.names) + 1):
            probe = pa.table(
                {
                    n: pa.array([], type=self._schema.field(n).type)
                    for n in self._schema.names
                    if n in wanted
                }
            )
            try:
                probe.filter(filter)
            except pa.lib.ArrowInvalid as exc:
                match = re.search(r"FieldRef\.Name\((.*?)\)", str(exc))
                name = match.group(1) if match else None
                if name in set(self._schema.names) - wanted:
                    wanted.add(name)
                    continue
                return list(self._schema.names)
            else:
                return [n for n in self._schema.names if n in wanted]
        return list(self._schema.names)

    # ------------------------------------------------------------------
    # Pruning: which chunks can satisfy the predicate
    # ------------------------------------------------------------------

    def _dim_shadows(self) -> dict[str, _DimShadow]:
        """One pruning index per prunable dimension, built lazily.

        Keeping one shadow per dimension (Σ n_d fragments) instead of one
        per chunk (Π n_d) is what keeps this cheap for finely partitioned
        datasets, and is sound: a chunk is dropped only when the full
        predicate is provably false given that single dimension's range.
        """
        if self._shadows is not None:
            return self._shadows
        shadows: dict[str, _DimShadow] = {}
        for dim in self._resolved:
            name = str(dim)
            if name not in self._schema.names:
                continue
            coord = self._coord_arrays[name]
            if coord.dtype.kind not in ("i", "u", "f", "M"):
                continue  # strings/objects/cftime: never prune this dim
            try:
                shadows[name] = _DimShadow(
                    name, self._schema, coord, self._chunk_bounds[dim]
                )
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError):
                continue  # conservative: no pruning on this dim
        self._shadows = shadows
        return shadows

    def _prune(self, filter: pc.Expression) -> dict[str, list[int]]:
        """Per-dimension chunk indices that can satisfy ``filter``.

        Satisfiability is delegated to Arrow's guarantee simplification
        (see :class:`_DimShadow`) — no expression decoding here, and
        predicates on columns a shadow knows nothing about are
        conservatively kept. Dimensions without a shadow, or where every
        chunk survives, are absent from the result.
        """
        kept: dict[str, list[int]] = {}
        for name, shadow in self._dim_shadows().items():
            indices = shadow.kept(filter)
            if indices is not None:
                kept[name] = indices
        return kept

    # ------------------------------------------------------------------
    # Scan: load surviving chunks, prefetching ahead of the consumer
    # ------------------------------------------------------------------

    def _blocks(self, kept: dict[str, list[int]] | None) -> Iterator[Block]:
        """Yield isel-able block slices for the surviving chunk grid."""
        dims = list(self._resolved.keys())
        if not dims:
            yield {}
            return
        index_ranges = [
            (kept or {}).get(str(d), range(len(self._resolved[d])))
            for d in dims
        ]
        for combo in itertools.product(*index_ranges):
            block: Block = {d: slice(None) for d in self._ds.dims}
            for d, i in zip(dims, combo):
                bounds = self._chunk_bounds[d]
                block[d] = slice(int(bounds[i]), int(bounds[i + 1]))
            yield block

    def _batch_generator(
        self,
        scan_schema: pa.Schema,
        kept: dict[str, list[int]] | None,
    ) -> Iterator[pa.RecordBatch]:
        names = list(scan_schema.names)
        data_vars = [n for n in names if n in self._ds.data_vars]
        # Select only the needed variables before slicing so unrequested
        # variables are never loaded (dimension coords come via coords).
        base = (
            self._ds[data_vars]
            if data_vars
            else self._ds.drop_vars(list(self._ds.data_vars))
        )

        def load(block: Block) -> list[pa.RecordBatch]:
            if self._iteration_callback is not None:
                self._iteration_callback(block, names)
            return list(
                iter_record_batches(
                    base.isel(block), scan_schema, self._batch_size
                )
            )

        def generate() -> Iterator[pa.RecordBatch]:
            blocks = self._blocks(kept)
            if self._prefetch <= 1:
                for block in blocks:
                    yield from load(block)
                return
            pool = ThreadPoolExecutor(max_workers=self._prefetch)
            pending: deque = deque()
            try:
                for block in blocks:
                    pending.append(pool.submit(load, block))
                    if len(pending) >= self._prefetch:
                        yield from pending.popleft().result()
                while pending:
                    yield from pending.popleft().result()
            finally:
                # Consumer may stop early (e.g. LIMIT): drop queued work
                # without waiting for in-flight loads.
                pool.shutdown(wait=False, cancel_futures=True)

        return generate()


@register_adapter
class DuckDBAdapter:
    """Registers Datasets on ``duckdb.DuckDBPyConnection`` connections."""

    @staticmethod
    def matches(con: Any) -> bool:
        # The connection class lives in ``duckdb`` or, in newer releases,
        # the ``_duckdb`` C-extension module.
        root = type(con).__module__.split(".")[0]
        return root in ("duckdb", "_duckdb")

    @staticmethod
    def register(
        con: Any,
        name: str,
        ds: xr.Dataset,
        *,
        chunks: Chunks = None,
        **kwargs: Any,
    ) -> Any:
        """Register ``ds`` on a DuckDB connection.

        Datasets whose variables all share the same dimensions become a
        single table named ``name``. Mixed-dimension datasets are split
        into one table per dimension group, named
        ``<name>_<dim1>_<dim2>_...`` (DuckDB registration has no schema
        namespace to mirror the DataFusion adapter's ``name.group``
        layout). Extra keyword arguments (``batch_size``, ``prefetch``)
        are forwarded to :class:`XarrayPushdownDataset`.
        """
        groups = _group_vars_by_dims(ds)
        if len(groups) <= 1:
            con.register(name, XarrayPushdownDataset(ds, chunks, **kwargs))
            return con
        # Materialise dim coordinates once and share across sub-tables.
        coord_arrays = {
            str(dim): ds.coords[dim].values
            for dim in ds.dims
            if dim in ds.coords
        }
        for dims, var_names in groups.items():
            suffix = "_".join(dims) or "scalar"
            con.register(
                f"{name}_{suffix}",
                XarrayPushdownDataset(
                    ds[var_names],
                    chunks,
                    coord_arrays=coord_arrays,
                    **kwargs,
                ),
            )
        return con
