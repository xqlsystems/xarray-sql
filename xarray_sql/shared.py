"""Explicit shared-stage execution for deterministic SQL training graphs.

[EinFold](https://github.com/xqlsystems/einfold) recognizes array operations
inside DataFusion plans. This executor materializes reused common table
expressions (CTEs) between those plans. An installed
[MLX](https://github.com/ml-explore/mlx) extension can execute the numerical
kernels on Apple's GPU. Stage boundaries currently exchange Arrow tables.

[ddx](https://github.com/xqlsystems/ddx) differentiates the original SQL,
including virtual view definitions, before choosing its own materialization
boundaries. Forward staging never replaces the query ddx differentiates.

This experimental API accepts SELECTs with flat, nonrecursive WITH clauses
and the deterministic numerical functions listed below. It requires the
native feature build in ``accelerators/einfold_mlx``. Tables may be updated through
``Context._write`` or registered again between calls; calls on this executor
are serialized. Coordinate/schema changes rebuild physical plans.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Sequence

import einfold._native as _einfold
import numpy as np
import pyarrow as pa
from datafusion import SessionContext
from ddxdb import UnsupportedExpression
from ddxdb import ad as _ddx
from ddxdb import prepare as _prepare_ad
from ddxdb._ddxdb import to_sql as _to_sql

from xarray_sql._native import _SortedArrowTable
from xarray_sql._native import shared_ctes as _shared_ctes
from xarray_sql.ad import calls_ad
from xarray_sql.sql import _copy, _tables_read

__all__ = ["SharedExecutor"]

_FUNCTIONS = frozenset(
    [
        "abs",
        "avg",
        "ceil",
        "coalesce",
        "count",
        "ddx_stop_gradient",
        "exp",
        "erf",
        "floor",
        "greatest",
        "least",
        "ln",
        "log",
        "max",
        "min",
        "nullif",
        "pow",
        "power",
        "round",
        "sign",
        "sin",
        "cos",
        "sqrt",
        "sum",
        "tanh",
        "row_number",
    ]
)
_DYNAMIC_FILTERS = tuple(
    f"datafusion.optimizer.enable{kind}_dynamic_filter_pushdown"
    for kind in ("", "_join", "_topk", "_aggregate")
)


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _snapshot(rows):
    return pa.Table.from_arrays(
        [
            pa.concat_arrays(c.chunks)
            if c.chunks
            else pa.array([], type=c.type)
            for c in rows.columns
        ],
        schema=rows.schema,
    )


def _canonical(rows, keys):
    """Avoid sorting the already ordered positional output of dense kernels."""
    if rows.num_rows < 2:
        return rows
    tied = None
    for name, _ in keys:
        column = rows[name]
        if column.null_count or not pa.types.is_integer(column.type):
            return rows.sort_by(keys)
        values = column.to_numpy(zero_copy_only=False)
        descending = values[1:] < values[:-1]
        if tied is not None:
            descending &= tied
        if descending.any():
            return rows.sort_by(keys)
        equal = values[1:] == values[:-1]
        tied = equal if tied is None else tied & equal
        if not np.any(tied):
            return rows
    return rows


def _join_only(plan):
    """Join rows can remain logical until a consumer reduces them."""
    joined = False
    stack = [plan]
    allowed = {
        "Join",
        "CrossJoin",
        "Projection",
        "Filter",
        "SubqueryAlias",
        "TableScan",
    }
    while stack:
        node = stack.pop()
        kind = type(node.to_variant()).__name__
        if kind not in allowed:
            return False
        joined |= kind in {"Join", "CrossJoin"}
        stack.extend(node.inputs())
    return joined


class _Compiled:
    def __init__(self, owner, frame, *, deferred=False, graph=None):
        self.frame, self.schema = frame, frame.schema()
        self.deferred = deferred
        self.graph = graph
        self.plan = None
        self.epoch = owner._epoch
        self.batch = None
        self.inputs = {
            name: owner._versions.get(name, 0)
            for name in _tables_read(frame.logical_plan())
        }
        if not deferred and graph is None:
            self._physical(owner)

    def _physical(self, owner):
        start = time.perf_counter()
        self.plan = self.frame.execution_plan()
        owner._counts["physical_plans"] += 1
        owner._times["planning_seconds"] += time.perf_counter() - start

    def valid(self, owner):
        return self.epoch == owner._epoch and all(
            owner._versions.get(name, 0) == generation
            for name, generation in self.inputs.items()
        )

    def run(self, owner):
        if self.graph is not None:
            # ddx recreates transient view providers each pass. Rebind the
            # compound stage to those providers while retaining its buffers.
            return self.graph.run(rebind=True)
        if self.plan is None:
            self._physical(owner)
        start = time.perf_counter()
        plan = _copy(owner.ctx, self.plan)
        batches = [
            b.to_pyarrow()
            for p in range(plan.partition_count)
            for b in owner.ctx.execute(plan, p)
        ]
        # Physical simplification can prove a boolean nonnullable while
        # the logical schema still permits NULL. Keep the declared schema
        # when collecting, without copying or converting column values.
        batches = [
            pa.RecordBatch.from_arrays(b.columns, schema=self.schema)
            for b in batches
        ]
        owner._counts["plan_executions"] += 1
        owner._times["execution_seconds"] += time.perf_counter() - start
        return pa.Table.from_batches(batches, self.schema)

    def store(self, owner, name, rows):
        started = time.perf_counter()
        keys = [
            (f.name, "ascending")
            for f in rows.schema
            if pa.types.is_integer(f.type) or pa.types.is_string(f.type)
        ]
        if keys:
            rows = _canonical(rows, keys)
        arrays = [c.combine_chunks() for c in rows.columns]
        compatible = (
            self.batch is not None
            and self.batch.schema == rows.schema
            and self.batch.num_rows == rows.num_rows
        )
        if compatible and self.batch is not None:
            # Integer/string keys must retain both their domain and order:
            # cached scan statistics were derived from these coordinates.
            compatible = all(
                new.null_count == 0 and old.null_count == 0
                if pa.types.is_floating(new.type)
                else new.equals(old)
                for new, old in zip(arrays, self.batch.columns)
            )
        if compatible and self.batch is not None:
            for new, old in zip(arrays, self.batch.columns):
                if pa.types.is_floating(new.type):
                    width = new.type.byte_width
                    a = memoryview(new.buffers()[1])[
                        new.offset * width : (new.offset + len(new)) * width
                    ]
                    memoryview(old.buffers()[1])[:] = a
            owner._counts["stage_buffer_writes"] += 1
        else:
            if self.batch is not None:
                if self.batch.num_rows != rows.num_rows:
                    reason = "rows"
                elif self.batch.schema != rows.schema:
                    reason = "schema"
                else:
                    reason = "keys_or_nulls"
                owner._counts[f"stage_replacements_{reason}"] += 1
                owner._versions[name] = owner._versions.get(name, 0) + 1
            # concat_arrays always owns its value buffers, including a
            # single input chunk. No stage overwrites its producer's data.
            self.batch = pa.RecordBatch.from_arrays(
                [owner._intern_key(a) for a in arrays], schema=rows.schema
            )
            owner._counts["stage_buffer_allocations"] += 1
        owner.ctx.deregister_table(name)
        if keys:
            owner.ctx.register_table(
                name, _SortedArrowTable(self.batch, [k for k, _ in keys])
            )
        else:
            owner.ctx.register_record_batches(name, [[self.batch]])
        owner._times["stage_storage_seconds"] += time.perf_counter() - started


class _Backend(_ddx.DataFusionBackend):
    def __init__(self, owner):
        super().__init__(owner.ctx)
        self.owner = owner
        self._catalog: list = []
        self._changes = 0
        self._deferred: dict[str, _Compiled] = {}

    def select_all(self, name):
        deferred = self._deferred.get(name)
        if deferred is None:
            return super().select_all(name)
        # ddx asks for a named Read with the saved table's schema. A view
        # would expand to its entire definition in Substrait. Serialize a
        # schema-only table, then restore the view before any execution.
        empty = pa.RecordBatch.from_arrays(
            [pa.array([], type=f.type) for f in deferred.schema],
            schema=deferred.schema,
        )
        self.ctx.deregister_table(name)
        try:
            self.ctx.register_record_batches(name, [[empty]])
            return super().select_all(name)
        finally:
            self.ctx.deregister_table(name)
            self.ctx.register_view(name, deferred.frame)

    def drop_table(self, name):
        self._deferred.pop(name, None)
        super().drop_table(name)

    def materialize(self, name, plan):
        super().materialize(name, plan)
        self.owner._versions[name] = self.owner._versions.get(name, 0) + 1

    def view_sql(self, name):
        if name in self.owner._views:
            return self.owner._views[name]
        return super().view_sql(name)

    def _bind_views(self, sql):
        # ddx synthesizes nested WITH queries while extracting shared CTEs.
        # Rejecting those probes makes it abandon staging and inline the
        # entire model. The restriction on user forward graphs is separate.
        parts = _shared_ctes(sql, allow_nested=True)
        defined = {name for name, *_ in parts}
        needed = {
            ref
            for _, _, refs, _ in parts
            for ref in refs
            if ref in self.owner._views and ref not in defined
        }
        if not needed:
            return sql
        # ddx extracts reused CTEs from large views. A simple view may be
        # left to the engine; bind its original SELECT in that case too.
        definitions = ", ".join(
            f"{_quote(name)} AS ({self.owner._views[name]})"
            for name in sorted(needed)
        )
        return f'WITH {definitions} SELECT * FROM ({sql}) AS "__xql_original"'

    def plan_sql(self, sql):
        return super().plan_sql(self._bind_views(sql))

    def plan_key(self, sql):
        return super().plan_key(self._bind_views(sql))

    def session_key(self):
        # Internal shared stages may be created while preparing a program.
        # They are execution state, rather than changes to its public inputs.
        try:
            settings = (
                SessionContext.sql(
                    self.ctx,
                    "SELECT name,value FROM information_schema.df_settings ORDER BY name",
                )
                .to_arrow_table()
                .to_pylist()
            )
        except ValueError:
            return None
        catalog = []
        for c in sorted(self.ctx.catalog_names()):
            for s in sorted(self.ctx.catalog(c).schema_names()):
                if s == "information_schema":
                    continue
                schema = self.ctx.catalog(c).schema(s)
                for t in sorted(schema.table_names()):
                    if t.startswith(("__ddx_", "__xql_shared_")):
                        continue
                    table = schema.table(t)
                    path = ".".join(_quote(p) for p in (c, s, t))
                    value = (
                        self.ctx.table(path).logical_plan()
                        if table.kind == "view"
                        else table.schema
                    )
                    catalog.append((path, value))
        if catalog != self._catalog:
            self._catalog, self._changes = catalog, self._changes + 1
        return repr((self._changes, settings, self.owner._views))

    def compile(self, plan):
        # DataFusion 55's Substrait consumer applies Read projections only
        # to TableScan roots; a deferred view expands to a join instead.
        # ddx's SQL writer preserves the same plan and handles those reads.
        frame = (
            SessionContext.sql(self.ctx, _to_sql(plan, "datafusion"))
            if self._deferred
            else _ddx._dataframe(self.ctx, plan)
        )
        stack = [frame.logical_plan()]
        unions = 0
        while stack:
            node = stack.pop()
            inputs = node.inputs()
            if type(node.to_variant()).__name__ == "Union":
                unions += max(0, len(inputs) - 1)
            stack.extend(inputs)
        if unions >= 32:
            # Large cotangent sums have many independently reduced UNION
            # branches. Materialize those reductions sequentially; running
            # all their join build sides together can exhaust memory.
            graph = self._stage_graph(plan, reductions=True)
            if graph is not None:
                self.owner._counts["union_reduction_graphs"] += 1
                return _Compiled(self.owner, frame, graph=graph)
        deferred = _join_only(frame.logical_plan())
        if deferred:
            try:
                parts = _shared_ctes(
                    _to_sql(plan, "datafusion"), allow_nested=True
                )
                deferred = all(set(fns) <= _FUNCTIONS for _, _, _, fns in parts)
            except (UnsupportedExpression, ValueError):
                deferred = False
        return _Compiled(self.owner, frame, deferred=deferred)

    def _stage_graph(self, plan, **options):
        try:
            sql = _to_sql(plan, "datafusion")
            return _Graph(self.owner, sql, **options)
        except (UnsupportedExpression, ValueError):
            return None

    def _refresh_compiled(self, plan, compiled):
        if compiled.valid(self.owner):
            return False
        fresh = self.compile(plan)
        batch = compiled.batch
        if (
            compiled.graph is not None
            and fresh.graph is not None
            and compiled.graph.sql == fresh.graph.sql
        ):
            fresh.graph = compiled.graph
        compiled.__dict__.update(fresh.__dict__)
        compiled.batch = batch
        return True

    def materialize_compiled(self, name, plan, compiled):
        changed = self._refresh_compiled(plan, compiled)
        if compiled.deferred:
            if changed:
                self.owner._versions[name] = (
                    self.owner._versions.get(name, 0) + 1
                )
            self._deferred[name] = compiled
            self.ctx.deregister_table(name)
            self.ctx.register_view(name, compiled.frame)
            self.owner._counts["deferred_stages"] += 1
            return
        rows = compiled.run(self.owner)
        compiled.store(self.owner, name, rows)
        compiled.epoch = self.owner._epoch

    def returns_rows_compiled(self, plan, compiled):
        self._refresh_compiled(plan, compiled)
        return compiled.run(self.owner).num_rows > 0


class _Graph:
    def __init__(self, owner, sql, *, reductions=False):
        self.owner = owner
        self.reductions = reductions
        self.sql = sql
        self.parts = _shared_ctes(sql)
        self.defs = {name: (query, refs) for name, query, refs, _ in self.parts}
        if len(self.defs) != len(self.parts):
            raise ValueError("duplicate CTE names cannot be staged")
        for _, _, _, functions in self.parts:
            unknown = set(functions) - _FUNCTIONS
            if unknown:
                raise ValueError(
                    "shared execution requires deterministic numerical SQL; "
                    f"unsupported functions: {', '.join(sorted(unknown))}"
                )
        needed = set()

        def visit(name, stack):
            if name in stack:
                raise ValueError("cyclic CTE dependencies cannot be staged")
            if name in needed:
                return
            needed.add(name)
            for dep in self.defs[name][1]:
                if dep in self.defs:
                    visit(dep, stack | {name})

        visit("", set())
        counts = Counter(
            dep
            for name in needed
            for dep in self.defs[name][1]
            if dep in needed
        )
        self.boundaries = {
            name
            for name, count in counts.items()
            if count > 1 and name not in owner.inline_ctes
        }
        if reductions:
            self.boundaries.update(
                name
                for name, _, _, functions in self.parts
                if name
                and name in needed
                and set(functions) & {"sum", "avg", "min", "max", "count"}
            )
        self.names = {
            name: owner._prefix + uuid.uuid4().hex
            for name in self.boundaries | {""}
        }
        self.order = [name for name, *_ in self.parts if name in self.names]
        self.compiled = {}
        self.tables = {
            dep
            for name in needed
            for dep in self.defs[name][1]
            if dep not in self.defs
        }

    def _sql(self, target, aliases):
        definitions = dict(aliases)

        def include(name):
            if name in definitions:
                return
            if name in self.boundaries and name != target:
                definitions[name] = f"SELECT * FROM {_quote(self.names[name])}"
                return
            for dep in self.defs[name][1]:
                if dep in self.defs:
                    include(dep)
            definitions[name] = self.defs[name][0]

        for dep in self.defs[target][1]:
            if dep in self.defs:
                include(dep)
        prefix = ", ".join(
            f"{_quote(name)} AS ({query})"
            for name, query in definitions.items()
        )
        return (f"WITH {prefix} " if prefix else "") + self.defs[target][0]

    def run(self, stack=(), *, rebind=False):
        owner = self.owner
        aliases = {}
        stages = 0
        for name in sorted(self.tables & owner._views.keys()):
            if name in stack:
                raise ValueError("cyclic virtual views cannot be staged")
            graph = owner._view_graph(name)
            rows = graph.run((*stack, name))
            stages += graph.stages
            graph.compiled[""].store(owner, graph.names[""], rows)
            aliases[name] = f"SELECT * FROM {_quote(graph.names[''])}"
        for name in self.order:
            compiled = self.compiled.get(name)
            if rebind or compiled is None or not compiled.valid(owner):
                frame = SessionContext.sql(owner.ctx, self._sql(name, aliases))
                fresh = _Compiled(owner, frame)
                if compiled is not None:
                    fresh.batch = compiled.batch
                compiled = self.compiled[name] = fresh
            rows = compiled.run(owner)
            stages += 1
            if name:
                compiled.store(owner, self.names[name], rows)
            compiled.epoch = owner._epoch
        self.stages = stages
        if self.reductions:
            owner._counts["reduction_stage_executions"] += stages
        else:
            owner._counts["forward_stages"] = stages
        return rows


class SharedExecutor:
    """Cache shared SQL stages and ddx programs over live parameter tables.

    Install EinFold or MLX on ``ctx`` before constructing this executor.
    Register model SQL with ``register_view``; it stays as SQL until ddx
    has differentiated it. ``prepare`` returns a function producing an
    Arrow table, and ``prepare_all`` produces a list of Arrow tables.

    ``inline_ctes`` names CTEs to keep inside consumers, for example an
    attention score/softmax chain that EinFold can fuse as one operator.
    Only reused, reachable CTEs outside that set become stage boundaries.

    Physical plans retain input buffers, with fresh join state each run.
    Re-registering a public table invalidates those plans. Returned results
    are independent snapshots. Changes to virtual views require preparing
    forward queries again. Do not modify context settings while executing.
    """

    def __init__(
        self,
        ctx,
        *,
        inline_ctes: Sequence[str] = (),
        max_key_cache_bytes: int = 256 << 20,
    ):
        if not getattr(_einfold, "RESET_ON_DECODE", False):
            raise ImportError(
                "rebuild accelerators/einfold_mlx for safe plan reuse"
            )
        self.ctx = ctx
        if max_key_cache_bytes < 0:
            raise ValueError("max_key_cache_bytes must be nonnegative")
        self._max_key_bytes = max_key_cache_bytes
        self._key_bytes = 0
        self._keys: OrderedDict[tuple, list] = OrderedDict()
        self.inline_ctes = frozenset(inline_ctes)
        self._views: dict[str, str] = {}
        self._graphs: dict[str, _Graph] = {}
        self._counts: Counter[str] = Counter()
        self._times: Counter[str] = Counter()
        self._epoch = 0
        self._versions: dict[str, int] = {}
        self._prefix = "__xql_shared_" + uuid.uuid4().hex + "_"
        self._lock = threading.RLock()
        self._catalog: tuple | None = None
        for key in _DYNAMIC_FILTERS:
            SessionContext.sql(ctx, f"SET {key} = false").collect()
        self._backend = _Backend(self)

    def _intern_key(self, array):
        size = array.nbytes
        if (
            not (
                pa.types.is_integer(array.type)
                or pa.types.is_string(array.type)
            )
            or size > self._max_key_bytes
        ):
            return pa.concat_arrays([array])
        # The fingerprint is only an index: equality checks every value and
        # NULL before sharing a buffer. Owned integer/string buffers are
        # immutable; floating point values always get separate storage.
        fingerprint = (
            str(array.type),
            len(array),
            array.null_count,
            tuple(
                array[i].as_py() for i in (0, len(array) // 2, len(array) - 1)
            )
            if len(array)
            else (),
        )
        candidates = self._keys.get(fingerprint, [])
        for old in candidates:
            if array.equals(old):
                self._keys.move_to_end(fingerprint)
                self._counts["key_cache_hits"] += 1
                self._counts["key_copy_bytes_avoided"] += size
                return old
        owned = pa.concat_arrays([array])
        while self._keys and self._key_bytes + size > self._max_key_bytes:
            _, evicted = self._keys.popitem(last=False)
            self._key_bytes -= sum(a.nbytes for a in evicted)
        # Eviction may have removed this fingerprint's bucket.
        self._keys.setdefault(fingerprint, []).append(owned)
        self._keys.move_to_end(fingerprint)
        self._key_bytes += size
        return owned

    def register_view(self, name: str, sql: str):
        """Keep a deterministic SELECT as a virtual view for execution and ddx."""
        with self._lock:
            _Graph(self, sql)  # Validate before changing the view catalog.
            if self._views.get(name) != sql:
                self._views[name] = sql
                self._graphs.clear()
                self._epoch += 1

    def _view_graph(self, name):
        if name not in self._graphs:
            self._graphs[name] = _Graph(self, self._views[name])
        return self._graphs[name]

    def _refresh(self):
        catalog = tuple(
            sorted(
                (name, generation)
                for name, generation in self.ctx._deregistrations.items()
                if not name.startswith(("__ddx_", "__xql_shared_"))
            )
        )
        if catalog != self._catalog:
            self._catalog = catalog
            self._epoch += 1

    def prepare(self, sql: str):
        """Prepare one SELECT; differentiation always uses its original SQL."""
        if calls_ad(sql):
            run = self.prepare_all([sql])
            return lambda **tables: run(**tables)[0]
        graph = _Graph(self, sql)

        def execute(**tables):
            with self._lock:
                self.ctx._write(tables)
                self._refresh()
                return _snapshot(graph.run())

        return execute

    def prepare_all(self, statements: Sequence[str]):
        """Prepare ddx statements sharing a backward pass over original views."""
        prepared = _prepare_ad(self._backend, statements)
        finals: dict[str, _Compiled] = {}

        def execute(**tables):
            with self._lock:
                self.ctx._write(tables)
                self._refresh()
                with prepared.run() as rewritten:
                    results = []
                    for sql in rewritten.statements:
                        compiled = finals.get(sql)
                        if compiled is None or not compiled.valid(self):
                            compiled = finals[sql] = _Compiled(
                                self, SessionContext.sql(self.ctx, sql)
                            )
                        results.append(_snapshot(compiled.run(self)))
                    return results

        return execute

    def statistics(self):
        """Executed stages, physical-plan builds and cumulative timings."""
        return (
            dict(self._counts)
            | dict(self._times)
            | {"key_cache_bytes": self._key_bytes}
        )
