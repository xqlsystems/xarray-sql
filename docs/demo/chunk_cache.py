"""Lazy SQL over remote Zarr in Pyodide, by faulting chunks in between runs.

In the browser, Zarr reads are asynchronous (``fetch``). zarr's
synchronous API blocks on them with Pyodide's JSPI support, but that
cannot suspend while the engine is calling back into Python: DuckDB and
pyarrow pull record batches through C++ frames that JSPI cannot unwind.
So a query cannot read chunks in the middle of a scan.

Instead, every scan reads from a cache of decoded chunks:

1. Run the query. A chunk that is not cached is recorded as missing and
   scanned as fill values, so the scan keeps going and records every
   chunk the query touches in one pass. xarray-sql's chunk pruning still
   applies, so only chunks that survive the query's coordinate filters
   are touched.
2. If anything was missing, fetch those chunks concurrently with
   ``Dataset.load_async`` (plain top-level ``await``, no JSPI), then run
   the query again.
3. A run with no misses saw only real data, so its result is exact.

Placeholder data can change data-dependent filters in the discarded
runs, which may fetch a few extra chunks, or make a run fail outright
(casting NaN to an integer, say). A failed run that missed chunks is
treated like any other: fetch, then run again. Neither changes the final
answer.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

ChunkKey = tuple[str, tuple[tuple[int, int], ...]]


def _fill_value(dtype: np.dtype) -> Any:
    if dtype.kind in "fc":
        return np.nan
    if dtype.kind == "M":
        return np.datetime64("NaT")
    if dtype.kind == "m":
        return np.timedelta64("NaT")
    return 0


class _CachedVariable(xr.backends.BackendArray):
    """A lazily indexed variable whose reads are served from the cache."""

    def __init__(self, cache: ChunkCache, name: str, source: xr.Variable):
        self._cache = cache
        self._name = name
        self.shape = source.shape
        self.dtype = source.dtype
        self._chunks = tuple(
            cache.chunks.get(str(d), n)
            for d, n in zip(source.dims, source.shape)
        )

    def __getitem__(self, key: Any) -> np.ndarray:
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.BASIC,
            self._raw_getitem,
        )

    def _raw_getitem(self, key: tuple) -> np.ndarray:
        """Assemble a basic selection from the cache chunks it overlaps."""
        ranges, steps, squeeze = [], [], []
        for axis, (k, n) in enumerate(zip(key, self.shape)):
            if isinstance(k, (int, np.integer)):
                k = slice(int(k), int(k) + 1)
                squeeze.append(axis)
            start, stop, step = k.indices(n)
            ranges.append((start, max(start, stop)))
            steps.append(step)
        out = np.empty([hi - lo for lo, hi in ranges], dtype=self.dtype)
        per_axis = [
            range(lo // c, (hi - 1) // c + 1) if hi > lo else range(0)
            for (lo, hi), c in zip(ranges, self._chunks)
        ]
        for index in itertools.product(*per_axis):
            bounds = tuple(
                (i * c, min((i + 1) * c, n))
                for i, c, n in zip(index, self._chunks, self.shape)
            )
            chunk = self._cache._get((self._name, bounds))
            src, dst = [], []
            for (lo, hi), (b0, b1) in zip(ranges, bounds):
                a, b = max(lo, b0), min(hi, b1)
                src.append(slice(a - b0, b - b0))
                dst.append(slice(a - lo, b - lo))
            out[tuple(dst)] = chunk[tuple(src)]
        out = out[tuple(slice(None, None, s) for s in steps)]
        return out.squeeze(axis=tuple(squeeze)) if squeeze else out


@dataclass
class QueryStats:
    rounds: int = 0
    chunks_fetched: int = 0
    seconds_fetching: float = 0.0
    seconds_querying: float = 0.0
    misses_per_round: list[int] = field(default_factory=list)


class ChunkCache:
    """Serves a remote Dataset's chunks to synchronous scans from memory.

    Args:
        source: A lazily opened Dataset (e.g. ``xr.open_zarr(...,
            chunks=None)``). Its coordinates should already be in memory;
            only data variables are fetched through the cache.
        chunks: Chunk sizes per dimension. Align them with the store's
            (inner) chunks so one cached chunk is one fetch.
        max_bytes: Memory budget for cached chunks; least recently used
            chunks are evicted first.
    """

    def __init__(
        self,
        source: xr.Dataset,
        chunks: dict[str, int],
        max_bytes: int = 512 * 2**20,
    ) -> None:
        self.source = source
        self.chunks = chunks
        self.max_bytes = max_bytes
        self._chunks: OrderedDict[ChunkKey, np.ndarray] = OrderedDict()
        self._nbytes = 0
        self.missing: set[ChunkKey] = set()

    def __len__(self) -> int:
        """Number of chunks currently cached."""
        return len(self._chunks)

    def dataset(self) -> xr.Dataset:
        """The source Dataset with every data variable read through the cache.

        Register it with ``chunks=`` set to this cache's chunks, so each
        scanned block is one cached chunk.
        """
        variables = {
            name: xr.Variable(
                var.dims,
                xr.core.indexing.LazilyIndexedArray(
                    _CachedVariable(self, str(name), var.variable)
                ),
                attrs=var.attrs,
            )
            for name, var in self.source.data_vars.items()
        }
        return self.source.assign(variables)

    def _get(self, key: ChunkKey) -> np.ndarray:
        if key in self._chunks:
            self._chunks.move_to_end(key)
            return self._chunks[key]
        self.missing.add(key)
        dtype = self.source[key[0]].dtype
        shape = tuple(hi - lo for lo, hi in key[1])
        return np.full(shape, _fill_value(dtype), dtype=dtype)

    async def _fetch(self, key: ChunkKey) -> None:
        name, bounds = key
        var = self.source[name]
        region = {d: slice(lo, hi) for d, (lo, hi) in zip(var.dims, bounds)}
        loaded = await var.isel(region).load_async()
        values = np.asarray(loaded.values)
        self._chunks[key] = values
        self._nbytes += values.nbytes
        while self._nbytes > self.max_bytes and len(self._chunks) > 1:
            _, evicted = self._chunks.popitem(last=False)
            self._nbytes -= evicted.nbytes

    def _nbytes_of(self, keys: set[ChunkKey]) -> int:
        return sum(
            int(np.prod([hi - lo for lo, hi in bounds]))
            * self.source[name].dtype.itemsize
            for name, bounds in keys
        )

    async def fill(self) -> int:
        """Fetch every chunk recorded as missing; returns how many."""
        keys = list(self.missing)
        self.missing.clear()
        await asyncio.gather(*(self._fetch(k) for k in keys))
        return len(keys)

    async def query(self, run: Any) -> tuple[Any, QueryStats]:
        """Call ``run()`` until it completes without a cache miss.

        ``run`` executes the query and materializes its result, e.g.
        ``lambda: con.sql(sql).to_arrow_table()``.

        Every retry fetches chunks this query has not fetched before, so
        the loop ends. If a run misses a chunk this query already fetched,
        the cache evicted it to make room for the others: the query needs
        more than the cache holds.
        """
        stats = QueryStats()
        fetched: set[ChunkKey] = set()
        self.missing.clear()
        while True:
            stats.rounds += 1
            start = time.perf_counter()
            try:
                result = run()
            except Exception:
                # Fill values can fail a query that real data would not.
                if not self.missing:
                    raise
            stats.seconds_querying += time.perf_counter() - start
            stats.misses_per_round.append(len(self.missing))
            if not self.missing:
                return result, stats
            if (
                self._nbytes_of(self.missing) > self.max_bytes
                or self.missing & fetched
            ):
                needed = self._nbytes_of(fetched | self.missing)
                raise MemoryError(
                    f"The query needs about {needed / 2**20:.0f} MiB of "
                    f"chunks, more than the cache's "
                    f"{self.max_bytes / 2**20:.0f} MiB. "
                    "Narrow its filters on the dimension columns."
                )
            fetched |= self.missing
            start = time.perf_counter()
            stats.chunks_fetched += await self.fill()
            stats.seconds_fetching += time.perf_counter() - start
