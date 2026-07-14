"""Engine-neutral pyarrow views of lazy xarray Datasets.

Two ways to hand a lazy ``xarray.Dataset`` to an Arrow-speaking query
engine:

* :class:`XarrayPushdownDataset` — a real ``pyarrow.dataset.Dataset``
  subclass (the pattern Lance uses for ``LanceDataset``). Consumers of
  the pyarrow dataset protocol — DuckDB via ``con.register``, Polars via
  ``pl.scan_pyarrow_dataset``, or pyarrow itself — call
  :meth:`~XarrayPushdownDataset.scanner` with the columns a query needs
  and the predicate it pushed down, so the scan loads only the needed
  data variables from only the chunks whose coordinate ranges can
  satisfy the predicate. Construct one with :func:`arrow_dataset`.
* :class:`XarrayArrowStream` — a re-scannable Arrow C-stream
  (PyCapsule) view. No source-level pushdown, but works with any
  PyCapsule consumer; the dependency-light fallback.

Correctness contract shared by all consumers of the pushdown dataset:
engines may delete the filter conjuncts they push down (DuckDB does),
so the returned scanner applies the expression exactly via
``pyarrow.dataset.Scanner``; chunk pruning is only ever an optimization
on top.
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

_STRICT_MAX_BLOCKS = 4096
"""Cap on surviving chunks for the strictness (provably-true) analysis.

Deciding strictness builds one guarantee fragment per surviving chunk;
above this many survivors the analysis costs more than it saves, so
every survivor is treated as a boundary chunk (sound, just no fast
path).
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
            if (vals.dtype.kind == "f" and np.isnan(vals).any()) or (
                vals.dtype.kind == "M" and np.isnat(vals).any()
            ):
                # NaN/NaT poisons min/max into a (dim >= NaN) guarantee
                # that Arrow simplifies every predicate against as false,
                # silently pruning rows. An always-true guarantee keeps
                # the span unprunable instead.
                guarantee = pc.scalar(True)
            else:
                # min/max (not first/last) so descending axes like
                # latitude 90→-90 carry correct ranges.
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
        coalesce_rows: int | None = None,
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
        self._coalesce_rows = coalesce_rows
        self._iteration_callback = _iteration_callback
        self._shadows: dict[str, _DimShadow] | None = None
        # One long-lived pool shared by every scan, its threads spawned
        # NOW — never from inside an engine's scan callback. Creating a
        # pool (and its OS threads) per scan deadlocks when the scan is
        # driven from an engine executing under another thread pool
        # (dask computing chunks of a lazy round-trip): thread startup
        # and concurrent.futures' global shutdown lock interleave with
        # the engine's callback needing the GIL. Scans that stop early
        # cancel their queued loads instead of tearing the pool down.
        self._pool: ThreadPoolExecutor | None = None
        if self._prefetch > 1:
            self._pool = ThreadPoolExecutor(max_workers=self._prefetch)
            spawn = [
                self._pool.submit(lambda: None)
                for _ in range(self._prefetch)
            ]
            for f in spawn:
                f.result()

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
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> pads.Scanner:
        """Build a scanner for the requested columns and predicate.

        ``filter`` is applied exactly by the returned scanner (DuckDB
        deletes the conjuncts it pushes down and trusts the source to
        enforce them); chunk pruning and column selection only reduce
        how much data is read to get there. ``batch_size`` caps rows per
        emitted batch (Polars passes it through ``to_batches``). Extra
        keyword arguments from other pyarrow-dataset consumers are
        accepted and ignored.
        """
        kept = None if filter is None else self._prune(filter)
        blocks = (
            self._coalesced_blocks(kept)
            if self._coalesce_rows
            else self._blocks(kept)
        )
        return self._scanner_for_blocks(blocks, columns, filter, batch_size)

    def _scanner_for_blocks(
        self,
        blocks: Iterator[Block] | list[Block],
        columns: list[str] | None,
        filter: pc.Expression | None,
        batch_size: int | None = None,
    ) -> pads.Scanner:
        """A scanner over the given blocks; shared by dataset and fragments."""
        # ``None`` means every column (the pyarrow convention); an
        # explicitly empty list is a real projection ("no payload"), not
        # a request for the full schema.
        proj = list(self._schema.names) if columns is None else list(columns)
        scan_names = self._scan_columns(proj, filter)
        scan_schema = pa.schema([self._schema.field(n) for n in scan_names])
        batches = self._batch_generator(
            scan_schema, blocks, batch_size or self._batch_size
        )
        return pads.Scanner.from_batches(
            batches, schema=scan_schema, columns=proj, filter=filter
        )

    def get_fragments(
        self, filter: pc.Expression | None = None
    ) -> list["_XarrayFragment"]:
        """One fragment per chunk of the source grid, pruned by ``filter``.

        This is how DataFusion consumes the dataset
        (``SessionContext.register_dataset`` plans one partition per
        fragment and scans them in parallel), and enables the Dask
        pattern ``from_map(lambda f: f.to_table().to_pandas(),
        ds.get_fragments())``.
        """
        kept = None if filter is None else self._prune(filter)
        return [_XarrayFragment(self, block) for block in self._blocks(kept)]

    def count_rows(
        self, filter: pc.Expression | None = None, **kwargs: Any
    ) -> int:
        """Count rows, reading as little data as possible.

        Without a filter the count is pure chunk arithmetic — no I/O at
        all. With a filter, chunks are split three ways: pruned chunks
        contribute nothing, chunks whose coordinate ranges *prove* the
        filter true contribute their exact size arithmetically, and only
        the undecided boundary chunks are scanned (reading just the
        columns the filter references).
        """
        if not self._ds.sizes:
            return self.scanner(columns=[], filter=filter).count_rows()
        if filter is None:
            return int(np.prod([self._ds.sizes[d] for d in self._ds.dims]))
        kept = self._prune(filter)
        strict, boundary = self._split_strict_blocks(kept, filter)
        count = sum(self._block_rows(b) for b in strict)
        if boundary:
            count += self._scanner_for_blocks(
                boundary, [], filter
            ).count_rows()
        return count

    # Inherited convenience methods (to_table, head, to_batches, take)
    # route through scanner() and keep working; the members below would
    # touch the uninitialized native dataset.

    @property
    def partition_expression(self) -> pc.Expression:
        # The dataset-level guarantee: trivially true. The base class
        # getter reads native state this object does not have.
        return pc.scalar(True)

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
    # Strictness: which surviving chunks satisfy the filter entirely
    # ------------------------------------------------------------------

    def _chunk_guarantee(self, name: str, i: int) -> pc.Expression | None:
        """``name ∈ [min, max]`` for chunk ``i``, or ``None`` for no info.

        Mirrors :class:`_DimShadow`'s bounds logic, including the NaN/NaT
        poisoning guard (a NaN span carries no usable guarantee).
        """
        if name not in self._schema.names:
            return None
        coord = self._coord_arrays[name]
        if coord.dtype.kind not in ("i", "u", "f", "M"):
            return None
        bounds = self._chunk_bounds[name]
        vals = coord[bounds[i] : bounds[i + 1]]
        if (vals.dtype.kind == "f" and np.isnan(vals).any()) or (
            vals.dtype.kind == "M" and np.isnat(vals).any()
        ):
            return None
        field_type = self._schema.field(name).type
        lo = pa.scalar(vals.min(), type=field_type)
        hi = pa.scalar(vals.max(), type=field_type)
        return (pc.field(name) >= lo) & (pc.field(name) <= hi)

    def _split_strict_blocks(
        self,
        kept: dict[str, list[int]] | None,
        filter: pc.Expression,
    ) -> tuple[list[Block], list[Block]]:
        """Split surviving blocks into (provably-true, boundary).

        A chunk with conjunctive guarantee ``G`` satisfies ``filter``
        everywhere iff ``G ∧ ¬filter`` is unsatisfiable; Arrow's
        guarantee simplification decides that when the strict blocks'
        shadow is pruned with the *inverted* filter. Everything
        undecidable — NaN spans, non-prunable dims, oversized grids,
        expression shapes the simplifier rejects — lands conservatively
        in the boundary set, which the caller scans exactly.
        """
        combos = list(self._combos(kept))
        if not combos or len(combos) > _STRICT_MAX_BLOCKS:
            return [], [self._block_for_combo(c) for c in combos]
        dims = list(self._resolved.keys())
        try:
            inverted = ~filter
            fmt = pads.IpcFileFormat()
            fs = pafs.LocalFileSystem()
            fragments = []
            decidable: list[int] = []
            for pos, combo in enumerate(combos):
                guarantee: pc.Expression | None = None
                for d, i in zip(dims, combo):
                    g = self._chunk_guarantee(str(d), i)
                    if g is None:
                        guarantee = None
                        break
                    guarantee = g if guarantee is None else guarantee & g
                if guarantee is None:
                    continue  # no usable guarantee: stays boundary
                decidable.append(pos)
                fragments.append(
                    fmt.make_fragment(
                        str(pos), fs, partition_expression=guarantee
                    )
                )
            strict_pos: set[int] = set()
            if fragments:
                shadow = pads.FileSystemDataset(
                    fragments, self._schema, fmt, fs
                )
                survivors = {
                    int(frag.path)
                    for frag in shadow.get_fragments(filter=inverted)
                }
                strict_pos = {p for p in decidable if p not in survivors}
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError):
            return [], [self._block_for_combo(c) for c in combos]
        strict = [
            self._block_for_combo(c)
            for pos, c in enumerate(combos)
            if pos in strict_pos
        ]
        boundary = [
            self._block_for_combo(c)
            for pos, c in enumerate(combos)
            if pos not in strict_pos
        ]
        return strict, boundary

    def _block_rows(self, block: Block) -> int:
        rows = 1
        for d, sl in block.items():
            size = self._ds.sizes[d]
            start, stop, _ = sl.indices(size)
            rows *= stop - start
        return rows

    # ------------------------------------------------------------------
    # Scan: load surviving chunks, prefetching ahead of the consumer
    # ------------------------------------------------------------------

    def _combos(
        self, kept: dict[str, list[int]] | None
    ) -> Iterator[tuple[int, ...]]:
        """Surviving chunk-index combinations, in grid order."""
        dims = list(self._resolved.keys())
        if not dims:
            return
        index_ranges = [
            (kept or {}).get(str(d), range(len(self._resolved[d])))
            for d in dims
        ]
        yield from itertools.product(*index_ranges)

    def _block_for_combo(self, combo: tuple[int, ...]) -> Block:
        block: Block = {d: slice(None) for d in self._ds.dims}
        for d, i in zip(self._resolved.keys(), combo):
            bounds = self._chunk_bounds[d]
            block[d] = slice(int(bounds[i]), int(bounds[i + 1]))
        return block

    def _blocks(self, kept: dict[str, list[int]] | None) -> Iterator[Block]:
        """Yield isel-able block slices for the surviving chunk grid."""
        if not self._resolved:
            yield {}
            return
        for combo in self._combos(kept):
            yield self._block_for_combo(combo)

    def _coalesced_blocks(
        self, kept: dict[str, list[int]] | None
    ) -> Iterator[Block]:
        """Blocks with runs of consecutive chunks merged along one dim.

        Runs are merged along the most finely chunked dimension while
        the merged block stays under ``coalesce_rows`` rows. One merged
        block is one ``isel`` — on Zarr sources its member chunks are
        fetched by the store's own concurrent batch read instead of one
        request per chunk through the prefetch pool.
        """
        if not self._resolved:
            yield {}
            return
        dims = list(self._resolved.keys())
        merge_dim = max(dims, key=lambda d: len(self._resolved[d]))
        others = [d for d in dims if d != merge_dim]
        ranges = {
            d: list(
                (kept or {}).get(str(d), range(len(self._resolved[d])))
            )
            for d in dims
        }
        merge_bounds = self._chunk_bounds[merge_dim]
        # Rows contributed per merge-dim row by dims outside the merge
        # axis (unresolved dims span their full extent in every block).
        outer_rows = 1
        for d in self._ds.dims:
            if d not in self._resolved:
                outer_rows *= self._ds.sizes[d]

        def flush(prefix: tuple[int, ...], run: list[int]) -> Block:
            block: Block = {d: slice(None) for d in self._ds.dims}
            for d, i in zip(others, prefix):
                bounds = self._chunk_bounds[d]
                block[d] = slice(int(bounds[i]), int(bounds[i + 1]))
            block[merge_dim] = slice(
                int(merge_bounds[run[0]]), int(merge_bounds[run[-1] + 1])
            )
            return block

        for prefix in itertools.product(*(ranges[d] for d in others)):
            per_row = outer_rows
            for d, i in zip(others, prefix):
                bounds = self._chunk_bounds[d]
                per_row *= int(bounds[i + 1] - bounds[i])
            run: list[int] = []
            run_rows = 0
            for i in ranges[merge_dim]:
                rows = int(merge_bounds[i + 1] - merge_bounds[i]) * per_row
                if run and (
                    i != run[-1] + 1
                    or run_rows + rows > self._coalesce_rows
                ):
                    yield flush(prefix, run)
                    run, run_rows = [], 0
                run.append(i)
                run_rows += rows
            if run:
                yield flush(prefix, run)

    def _batch_generator(
        self,
        scan_schema: pa.Schema,
        blocks: Iterator[Block] | list[Block],
        batch_size: int,
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
            if not names:
                # Zero-column projection: row counts are chunk
                # arithmetic; no coordinate or variable data is read.
                out = []
                rows = self._block_rows(block)
                while rows > 0:
                    n = min(rows, batch_size)
                    out.append(
                        pa.table({"_": np.empty(n, np.int8)})
                        .select([])
                        .to_batches()[0]
                    )
                    rows -= n
                return out
            return list(
                iter_record_batches(base.isel(block), scan_schema, batch_size)
            )

        def generate() -> Iterator[pa.RecordBatch]:
            block_iter = iter(blocks)
            first = next(block_iter, None)
            if first is None:
                return
            second = next(block_iter, None)
            if self._pool is None or second is None:
                # Single-block scans (a lazy round-trip window that maps
                # onto one source chunk) skip the pool entirely.
                yield from load(first)
                if second is not None:
                    yield from load(second)
                    for block in block_iter:
                        yield from load(block)
                return
            pending: deque = deque()
            try:
                pending.append(self._pool.submit(load, first))
                pending.append(self._pool.submit(load, second))
                for block in block_iter:
                    pending.append(self._pool.submit(load, block))
                    if len(pending) >= self._prefetch:
                        yield from pending.popleft().result()
                while pending:
                    yield from pending.popleft().result()
            finally:
                # Consumer may stop early (e.g. LIMIT): drop queued work
                # without waiting for in-flight loads. The pool itself is
                # shared across scans and stays up.
                for f in pending:
                    f.cancel()

        return generate()


class _XarrayFragment:
    """One chunk of the source grid, presented as a dataset fragment.

    Fragment consumers (DataFusion's ``DatasetExec`` plans one partition
    per fragment; Dask maps over them) call :meth:`scanner` with the
    columns and predicate for this piece; the pushed filter is applied
    row-exactly, same as the parent dataset's scanner.
    """

    def __init__(self, dataset: XarrayPushdownDataset, block: Block):
        self._dataset = dataset
        self._block = block

    @property
    def physical_schema(self) -> pa.Schema:
        return self._dataset.schema

    def scanner(
        self,
        schema: pa.Schema | None = None,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> pads.Scanner:
        return self._dataset._scanner_for_blocks(
            [self._block], columns, filter, batch_size
        )

    def to_batches(self, **kwargs: Any) -> Iterator[pa.RecordBatch]:
        return self.scanner(**kwargs).to_batches()

    def to_table(self, **kwargs: Any) -> pa.Table:
        return self.scanner(**kwargs).to_table()

    def count_rows(self, **kwargs: Any) -> int:
        return self.scanner(**kwargs).count_rows()

    def __dask_tokenize__(self) -> tuple:
        # Dask hashes from_map inputs; the parent dataset is unpicklable,
        # so provide a deterministic token from the fragment's identity.
        return (
            "xarray_sql._XarrayFragment",
            repr(self._block),
            self._dataset.schema.to_string(),
        )

    def __repr__(self) -> str:
        return f"_XarrayFragment({self._block!r})"


def arrow_dataset(
    ds: xr.Dataset,
    chunks: Chunks = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    prefetch: int = DEFAULT_PREFETCH,
    coalesce_rows: int | None = None,
) -> XarrayPushdownDataset:
    """A pushdown-capable ``pyarrow.dataset.Dataset`` view of ``ds``.

    The returned object works anywhere a pyarrow dataset does, keeping
    projection pushdown and coordinate-range chunk pruning::

        import polars as pl
        lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(ds))

        import duckdb
        duckdb.connect().register("t", xql.arrow_dataset(ds))

        xql.arrow_dataset(ds).to_table(columns=["t2m"], filter=...)

    Args:
        ds: An xarray Dataset. All data variables must share the same
            dimensions (select a variable subset first otherwise).
        chunks: Xarray-like chunks specification controlling partition
            granularity. Defaults to the Dataset's existing chunks.
        batch_size: Maximum rows per emitted Arrow RecordBatch.
        prefetch: Chunk loads kept in flight ahead of the consumer
            (memory scales with ``prefetch`` x pivoted chunk size).
        coalesce_rows: When set, merge runs of consecutive surviving
            chunks along the most finely chunked dimension into single
            reads of at most this many rows. Fewer, larger source
            requests — the win on remote stores, where each merged read
            fetches its member chunks through the store's own concurrent
            batching. Memory scales with ``prefetch`` x the *merged*
            block size, so size accordingly (e.g. ``8_000_000``).

    Returns:
        An :class:`XarrayPushdownDataset`.
    """
    return XarrayPushdownDataset(
        ds,
        chunks,
        batch_size=batch_size,
        prefetch=prefetch,
        coalesce_rows=coalesce_rows,
    )
