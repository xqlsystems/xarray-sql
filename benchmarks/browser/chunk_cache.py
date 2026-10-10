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
runs, which may fetch a few extra chunks; it never changes the final
answer.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import dask.array as da
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


class _CachedVariable:
    """Array-like that dask slices one chunk at a time; serves the cache."""

    def __init__(self, cache: ChunkCache, name: str, source: xr.Variable):
        self._cache = cache
        self._name = name
        self.shape = source.shape
        self.dtype = source.dtype
        self.ndim = source.ndim

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        bounds = tuple(
            (s.start or 0, self.shape[i] if s.stop is None else s.stop)
            for i, s in enumerate(key)
        )
        return self._cache._get((self._name, bounds))


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
        """The source Dataset with every data variable read through the cache."""
        variables = {}
        for name, var in self.source.data_vars.items():
            data = da.from_array(
                _CachedVariable(self, name, var.variable),
                chunks=tuple(self.chunks.get(d, -1) for d in var.dims),
                name=f"chunk-cache-{id(self)}-{name}",
                asarray=False,
                meta=np.empty((0,) * var.ndim, dtype=var.dtype),
            )
            variables[name] = var.copy(data=data)
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

    async def fill(self) -> int:
        """Fetch every chunk recorded as missing; returns how many."""
        keys = list(self.missing)
        self.missing.clear()
        await asyncio.gather(*(self._fetch(k) for k in keys))
        return len(keys)

    async def query(
        self, run: Any, max_rounds: int = 6
    ) -> tuple[Any, QueryStats]:
        """Call ``run()`` until it completes without a cache miss.

        ``run`` executes the query and materializes its result, e.g.
        ``lambda: con.sql(sql).to_arrow_table()``.
        """
        stats = QueryStats()
        self.missing.clear()
        for _ in range(max_rounds):
            stats.rounds += 1
            start = time.perf_counter()
            result = run()
            stats.seconds_querying += time.perf_counter() - start
            stats.misses_per_round.append(len(self.missing))
            if not self.missing:
                return result, stats
            needed = sum(
                int(np.prod([hi - lo for lo, hi in b]))
                * self.source[n].dtype.itemsize
                for n, b in self.missing
            )
            if needed > self.max_bytes:
                raise MemoryError(
                    f"The query needs {needed / 2**20:.0f} MiB of chunks, "
                    f"more than the cache's {self.max_bytes / 2**20:.0f} MiB. "
                    "Narrow its filters on the dimension columns."
                )
            start = time.perf_counter()
            stats.chunks_fetched += await self.fill()
            stats.seconds_fetching += time.perf_counter() - start
        raise RuntimeError(
            f"The query still read uncached chunks after {max_rounds} runs."
        )
