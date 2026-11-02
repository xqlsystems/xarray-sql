"""Experimental compact execution of SQL numerical programs.

[EinFold](https://github.com/xqlsystems/einfold) lowers DataFusion plans to
array operations. Its optional ``dense_program`` module retains compact
arrays between shared SQL and [ddx](https://github.com/xqlsystems/ddx)
automatic differentiation stages. Public results remain Arrow tables.
The original SQL executor handles unsupported plans without changing SQL
or gradient definitions. Install the patch in ``accelerators/einfold_mlx``.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pyarrow as pa
from datafusion import SessionContext
from ddxdb import ad as _ddx
from ddxdb import prepare as _prepare_ad
from einfold.dense_program import (
    _CACHE,
    _EXECUTION,
    _MAX_BYTES,
    _OPS,
    DenseProgram,
    _ComputedCache,
    _Decline,
    _read_dataset,
)

from xarray_sql.ad import calls_ad
from xarray_sql.shared import SharedExecutor, _Backend, _Graph, _snapshot

__all__ = ["DenseExecutor"]

_runtime_factory: type[Any] | None = None
if os.environ.get("MLX_ENABLE_TF32") == "0":
    try:
        from xarray_sql._dense_mlx import _Runtime
    except ImportError:
        pass
    else:
        _runtime_factory = _Runtime


class _Compiled:
    def __init__(self, owner, frame):
        self.schema = frame.schema()
        self.program = DenseProgram(frame.logical_plan(), self.schema)
        self.epoch = owner._epoch
        owner._counts["dense_programs"] += 1

    def run(self, owner):
        started = time.perf_counter()
        result = self.program.run(owner._read)
        owner._counts["dense_executions"] += 1
        owner._times["dense_lowering_seconds"] += time.perf_counter() - started
        return result

    def store(self, owner, name, table):
        owner._store[name] = table
        owner._stub(name, self.schema)


class _DenseBackend(_Backend):
    def compile(self, plan):
        return _Compiled(self.owner, _ddx._dataframe(self.ctx, plan))

    def materialize_compiled(self, name, plan, compiled):
        compiled.store(self.owner, name, compiled.run(self.owner))

    def returns_rows_compiled(self, plan, compiled):
        table = compiled.run(self.owner)
        # Checks may have arbitrary relational shapes; materialize their
        # actual values, never infer emptiness from schema-only providers.
        return table.arrow(compiled.schema).num_rows > 0

    def drop_table(self, name):
        self.owner._store.pop(name, None)
        self.owner._stubs.discard(name)
        super().drop_table(name)


class _DenseGraph(_Graph):
    def run(self, stack=(), *, rebind=False):
        owner, aliases = self.owner, {}
        for name in sorted(self.tables & owner._views.keys()):
            if name in stack:
                raise ValueError("cyclic virtual views cannot be staged")
            graph = owner._view_graph(name)
            table = graph.run((*stack, name))
            graph.compiled[""].store(owner, graph.names[""], table)
            aliases[name] = f'SELECT * FROM "{graph.names[""]}"'
        for name in self.order:
            compiled = self.compiled.get(name)
            if compiled is None or compiled.epoch != owner._epoch:
                compiled = self.compiled[name] = _Compiled(
                    owner,
                    SessionContext.sql(owner.ctx, self._sql(name, aliases)),
                )
            table = compiled.run(owner)
            if name:
                compiled.store(owner, self.names[name], table)
        return table


class DenseExecutor(SharedExecutor):
    """Execute supported SQL regions using compact arrays across ddx stages.

    Input Datasets are read afresh on every call, including in-place
    writes. All final Arrow tables are realized before returning, and no
    parameter array is mutated during execution. ``fallback=False`` is a
    diagnostic mode that reports unsupported regions immediately.
    """

    def __init__(
        self,
        ctx,
        *,
        precision="float64",
        fallback=True,
        max_array_bytes=1 << 30,
        max_cached_bytes=8 << 30,
        **options,
    ):
        if precision not in {"float64", "float32"}:
            raise ValueError("precision must be float64 or float32")
        if precision == "float32" and os.environ.get("MLX_ENABLE_TF32") != "0":
            raise ValueError("launch Python with MLX_ENABLE_TF32=0")
        if precision == "float32" and _runtime_factory is None:
            raise ImportError("MLX is required for float32 GPU execution")
        if max_array_bytes <= 0:
            raise ValueError("max_array_bytes must be positive")
        if max_cached_bytes <= 0:
            raise ValueError("max_cached_bytes must be positive")
        super().__init__(ctx, **options)
        self._max_array_bytes = max_array_bytes
        self._max_cached_bytes = max_cached_bytes
        self._runtime = (
            _runtime_factory()
            if precision == "float32" and _runtime_factory is not None
            else None
        )
        self._backend = _DenseBackend(self)
        self._store = {}
        self._stubs = set()
        self._fallback = SharedExecutor(ctx, **options) if fallback else None

    def register_view(self, name, sql):
        super().register_view(name, sql)
        if self._fallback is not None:
            self._fallback.register_view(name, sql)

    def _view_graph(self, name):
        if name not in self._graphs:
            self._graphs[name] = _DenseGraph(self, self._views[name])
        return self._graphs[name]

    def _stub(self, name, schema):
        if name not in self._stubs:
            empty = pa.RecordBatch.from_arrays(
                [pa.array([], type=f.type) for f in schema], schema=schema
            )
            self.ctx.deregister_table(name)
            self.ctx.register_record_batches(name, [[empty]])
            self._stubs.add(name)

    def _read(self, name):
        if name not in self._store:
            if name not in self.ctx._registered_datasets:
                raise _Decline(
                    f"source {name} has no rectangular Dataset contract"
                )
            self._store[name] = _read_dataset(
                name, self.ctx._registered_datasets[name]
            )
        return self._store[name]

    def _invoke(self, execute, fallback, tables):
        with self._lock:
            started = time.perf_counter()
            self.ctx._write(tables)
            self._refresh()
            self._store.clear()
            token = _EXECUTION.set(self._runtime)
            limit_token = _MAX_BYTES.set(self._max_array_bytes)
            cache = _ComputedCache(self._max_cached_bytes)
            cache_token = _CACHE.set(cache)
            try:
                overridden = (
                    self.ctx._custom_udfs - self.ctx._trusted_math_udfs
                ) & (
                    set(_OPS)
                    | {
                        "coalesce",
                        "nullif",
                        "sum",
                        "avg",
                        "min",
                        "max",
                        "count",
                    }
                )
                if overridden:
                    raise _Decline(
                        "custom scalar functions require SQL execution"
                    )
                return execute()
            except _Decline as error:
                self._counts["dense_fallbacks"] += 1
                self._last_decline = str(error)
                if fallback is None:
                    raise
                # Bind current function definitions when restarting in SQL.
                self._fallback._epoch += 1
                self._fallback._backend._catalog = []
                return fallback()
            finally:
                self._store.clear()
                _EXECUTION.reset(token)
                _MAX_BYTES.reset(limit_token)
                _CACHE.reset(cache_token)
                self._counts["dense_cache_evictions"] += cache.evictions
                self._counts["dense_peak_cached_bytes"] = max(
                    self._counts["dense_peak_cached_bytes"], cache.peak
                )
                self._times["dense_call_seconds"] += (
                    time.perf_counter() - started
                )

    def prepare(self, sql):
        if calls_ad(sql):
            run = self.prepare_all([sql])
            return lambda **tables: run(**tables)[0]
        graph = _DenseGraph(self, sql)
        fallback = (
            self._fallback.prepare(sql) if self._fallback is not None else None
        )

        def execute():
            result = graph.run()
            return _snapshot(result.arrow(graph.compiled[""].schema))

        return lambda **tables: self._invoke(execute, fallback, tables)

    def prepare_all(self, statements):
        prepared = _prepare_ad(self._backend, statements)
        finals = {}
        fallback = (
            self._fallback.prepare_all(statements)
            if self._fallback is not None
            else None
        )

        def execute():
            with prepared.run() as rewritten:
                results = []
                for sql in rewritten.statements:
                    if sql not in finals or finals[sql].epoch != self._epoch:
                        finals[sql] = _Compiled(
                            self, SessionContext.sql(self.ctx, sql)
                        )
                    compiled = finals[sql]
                    results.append(
                        _snapshot(compiled.run(self).arrow(compiled.schema))
                    )
                return results

        return lambda **tables: self._invoke(execute, fallback, tables)

    def statistics(self):
        return (
            super().statistics()
            | (dict(self._runtime.counts) if self._runtime is not None else {})
            | {"dense_last_decline": getattr(self, "_last_decline", None)}
        )
