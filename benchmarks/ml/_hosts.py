"""The SQL engines ("hosts") the ML demos run on.

einfold (https://github.com/xqlsystems/einfold), a query optimizer that
rewrites SQL plans into array algorithms, calls the engine that runs a
plan its *host*. Every demo writes
its model once, as SQL, and runs that same text on each host, so a
difference in time is a difference between engines, never between
queries. A host does these things:

* ``load(name, data, **options)`` registers an ``xr.Dataset`` as the
  table ``name`` through xarray-sql, replacing any table of that name
  (``options`` such as ``table_names`` go to xarray-sql);
* ``execute(sql)`` runs a statement for its effect, such as
  ``CREATE VIEW``;
* ``query(sql)`` runs a statement and returns its rows as Arrow;
* ``query_all(sqls)`` runs several statements, and, on a host that
  differentiates, shares the backward pass between those that take
  ``grad`` of the same loss;
* ``operators(sql)`` counts the operators in the physical plan it would
  run, which shows, for one, whether a CTE read twice is computed twice.

Forward queries run everywhere. Gradients (``grad``, ``vjp``, ``jvp``)
need ddx (https://github.com/xqlsystems/ddx), which differentiates SQL
queries and runs its programs on DataFusion and DuckDB; a host that
cannot differentiate says so with ``differentiates = False``, and the
demos report its gradient steps as not run rather than skip it silently.
A gradient that ddx (or the engine under it) refuses to plan raises one
of the host's ``refusals``, and the demos report that step with the reason.

Optional Apple silicon hosts use MLX (https://github.com/ml-explore/mlx),
Apple's array framework, through xarray-sql's native dense-kernel bridge.
They pair prepared queries with reusable parameter buffers. Forced variants
lower kernel thresholds to measure small operations on each backend.

Adding a host is one class here and one entry in ``HOSTS``. ``DEFAULT_HOSTS``
keeps optional device dependencies out of ordinary runs.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Sequence

import duckdb
import pyarrow as pa
import xarray as xr
from ddxdb import DdxError
from ddxdb import duckdb as ddx_duckdb
from einfold.datafusion import EinFold as EinFoldExtension

import xarray_sql as xql

try:
    from xarray_sql.mlx import MLX
except ImportError:
    MLX = None  # Optional Apple silicon dependency and Metal device.

try:
    from xarray_sql.shared import SharedExecutor
except ImportError:
    SharedExecutor = None

try:
    from xarray_sql.dense import DenseExecutor
except ImportError:
    DenseExecutor = None


class Host:
    """An engine that runs the demos' SQL. Subclasses fill in the rest."""

    name = ""
    differentiates = False
    refusals: tuple[type[Exception], ...] = ()
    rtol = 1e-9
    atol = 1e-9

    def measurements(self) -> dict:
        return {}

    def load(self, name: str, data: xr.Dataset, **options) -> None:
        raise NotImplementedError

    def execute(self, sql: str) -> None:
        raise NotImplementedError

    def query(self, sql: str) -> pa.Table:
        raise NotImplementedError

    def query_all(self, sqls: Sequence[str]) -> list[pa.Table]:
        return [self.query(sql) for sql in sqls]

    def operators(self, sql: str) -> int:
        raise NotImplementedError


class DataFusion(Host):
    """xarray-sql's ``Context``, with ddx's gradients (the ``ml`` extra)."""

    name = "datafusion"
    differentiates = True
    refusals = (DdxError,)

    def __init__(self):
        self.ctx = xql.Context()

    def load(self, name, data, **options):
        self.ctx.deregister_table(name)
        self.ctx.from_dataset(name, data, **options)

    def execute(self, sql):
        self.ctx.sql(sql).collect()

    def query(self, sql):
        return self.ctx.sql(sql).to_arrow_table()

    def query_all(self, sqls):
        return [df.to_arrow_table() for df in xql.ad.sql_all(self.ctx, sqls)]

    def operators(self, sql):
        plan = self.ctx.sql(sql).execution_plan().display_indent()
        return len(plan.splitlines())


class EinFold(DataFusion):
    """xarray-sql's ``Context`` with einfold installed as its query
    planner, and ddx's gradients.

    Where einfold finds a computation in a plan that it has an array
    algorithm for, such as a sum of products over a join or a
    softmax-weighted average of one (attention), it runs that algorithm;
    DataFusion runs the rest of the plan. ``einfold`` counts what it found:
    ``plans_with_pipelines()`` is the number of plans in which it ran a
    fused pipeline, such as attention tile by tile.
    """

    name = "einfold"

    def __init__(self):
        super().__init__()
        self.einfold = EinFoldExtension()
        self.ctx = self.ctx.with_extensions(self.einfold)

    def operators(self, sql):
        return None  # Native physical stages cross the FFI as opaque nodes.


class DataFusionPrepared(DataFusion):
    """Cached SQL plans and ddx programs over reusable parameter buffers.

    Data is copied on first registration so in-place training updates never
    mutate the caller's initial weights or the batches used by other hosts.
    A schema/coordinate change refreshes the tables, views and query caches.
    """

    name = "datafusion_prepared"

    def __init__(self):
        super().__init__()
        for kind in ("", "_join", "_topk", "_aggregate"):
            self.ctx.sql(
                f"SET datafusion.optimizer.enable{kind}_dynamic_filter_pushdown = false"
            ).collect()
        self._tables: set[str] = set()
        self._queries: dict = {}
        self._programs: dict = {}
        self._last_view: str | None = None
        self._counts: Counter[str] = Counter()
        self.accelerator = None

    def load(self, name, data, **options):
        if name in self._tables:
            try:
                self.ctx._write({name: data})
            except ValueError:
                pass
            else:
                self._counts["inplace_updates"] += 1
                return
        self._queries.clear()
        self._programs.clear()
        self._last_view = None
        super().load(name, data.copy(deep=True), **options)
        self._tables.add(name)
        self._counts["table_registers"] += 1

    def execute(self, sql):
        # The GPT recreates this same view after loading each batch. When
        # tables retain their buffers, its existing plan reads current values.
        view = sql.lstrip().upper().startswith("CREATE OR REPLACE VIEW ")
        if view and sql == self._last_view:
            return
        super().execute(sql)
        self._queries.clear()
        self._programs.clear()
        self._last_view = sql if view else None

    def query(self, sql):
        if xql.ad.calls_ad(sql):
            return self.query_all([sql])[0]
        if sql not in self._queries:
            self._queries[sql] = self.ctx.prepare(sql)
            self._counts["query_preparations"] += 1
        return self._queries[sql]()

    def query_all(self, sqls):
        key = tuple(sqls)
        if key not in self._programs:
            self._programs[key] = xql.ad.prepare(self.ctx, key)
            self._counts["gradient_preparations"] += 1
        return self._programs[key]()

    def measurements(self):
        values = dict(self._counts)
        if self.accelerator is not None:
            counters = self.accelerator.statistics()
            values.update(
                {
                    key: counters.get(key, 0)
                    for key in (
                        "cpu_matmul",
                        "gpu_matmul",
                        "gpu_attention",
                        "declined",
                    )
                }
            )
        return values

    def operators(self, sql):
        return None if self.accelerator is not None else super().operators(sql)


class CPUOptimized(DataFusionPrepared):
    """Prepared execution plus the float64 CPU dense-kernel bridge."""

    name = "cpu_optimized"
    precision = "float64"
    force_kernels = False

    def __init__(self):
        super().__init__()
        if MLX is None:
            raise ImportError(
                "this host needs the Apple silicon MLX extra and native hook"
            )
        options = {"min_ops": 0, "min_gpu_ops": 0} if self.force_kernels else {}
        self.accelerator = MLX(precision=self.precision, **options)
        self.ctx = self.ctx.with_extensions(self.accelerator)


class CPUForced(CPUOptimized):
    """Force eligible small products through the float64 CPU backend."""

    name = "cpu_forced"
    force_kernels = True


class GPUOptimized(CPUOptimized):
    """Prepared execution with the measured automatic GPU thresholds."""

    name = "gpu_optimized"
    precision = "float32"
    rtol = 3e-5
    atol = 3e-5


class GPUForced(GPUOptimized):
    """Force eligible operations onto MLX to expose small-kernel overhead."""

    name = "gpu_forced"
    force_kernels = True


class SharedEinFold(DataFusionPrepared):
    """Shared forward CTEs and reusable ddx physical plans through EinFold."""

    name = "shared_einfold"
    precision = None

    def __init__(self):
        super().__init__()
        if SharedExecutor is None:
            raise ImportError("rebuild xarray-sql for shared SQL execution")
        extension = EinFoldExtension()
        if self.precision is not None:
            if MLX is None:
                raise ImportError(
                    "shared CPU/GPU kernels require the MLX extra"
                )
            extension = self.accelerator = MLX(
                precision=self.precision,
                min_ops=0,
                min_gpu_ops=0,
                max_working_bytes=4 << 30,
            )
        self.ctx = self.ctx.with_extensions(extension)
        self.executor = SharedExecutor(self.ctx, max_key_cache_bytes=4 << 30)

    def load(self, name, data, **options):
        # Large stacked weights remain lazy Arrow streams. One layer per
        # partition lets a block's constant layer filter prune the others.
        if "layer" in data.dims and "chunks" not in options:
            options["chunks"] = {"layer": 1}
        super().load(name, data, **options)

    def execute(self, sql):
        view = re.fullmatch(
            r"\s*CREATE OR REPLACE VIEW (\w+) AS\s*(.*)",
            sql,
            re.IGNORECASE | re.DOTALL,
        )
        if view:
            name, query = view.groups()
            if sql != self._last_view:
                # Preserve the score/softmax chain as one EinFold region.
                self.executor.inline_ctes = frozenset(
                    re.findall(
                        r"\b(l\d+_(?:scores|weights|norm))\s+AS",
                        query,
                        re.IGNORECASE,
                    )
                )
                self.executor.register_view(name, query)
                self._queries.clear()
                self._programs.clear()
                self._last_view = sql
            return
        super().execute(sql)

    def query(self, sql):
        if xql.ad.calls_ad(sql):
            return self.query_all([sql])[0]
        if sql not in self._queries:
            self._queries[sql] = self.executor.prepare(sql)
            self._counts["query_preparations"] += 1
        return self._queries[sql]()

    def query_all(self, sqls):
        key = tuple(sqls)
        if key not in self._programs:
            self._programs[key] = self.executor.prepare_all(sqls)
            self._counts["gradient_preparations"] += 1
        return self._programs[key]()

    def operators(self, sql):
        # The FFI planner exposes each stage as an opaque node. Its node
        # count cannot be compared with ordinary DataFusion's operators.
        return None

    def measurements(self):
        return super().measurements() | self.executor.statistics()


class SharedCPU(SharedEinFold):
    """Shared execution with double-precision dense CPU kernels."""

    name = "shared_cpu"
    precision = "float64"


class SharedGPU(SharedEinFold):
    """Shared execution with explicitly enabled MLX float32 GPU kernels."""

    name = "shared_gpu"
    precision = "float32"
    rtol = 3e-5
    atol = 3e-5


class DenseCPU(SharedEinFold):
    """Compact double-precision arrays across SQL and ddx stages."""

    name = "dense_cpu"
    dense_precision = "float64"

    def __init__(self):
        super().__init__()
        if DenseExecutor is None:
            raise ImportError("install EinFold's optional dense program module")
        self.executor = DenseExecutor(
            self.ctx,
            precision=self.dense_precision,
            fallback=False,
            max_cached_bytes=int(
                os.environ.get("XQL_DENSE_CACHE_BYTES", str(8 << 30))
            ),
        )


class DenseGPU(DenseCPU):
    """Compact MLX arrays across SQL and ddx stages, with float32 arithmetic."""

    name = "dense_gpu"
    dense_precision = "float32"
    rtol = 3e-5
    atol = 3e-5


class DuckDB(Host):
    """DuckDB, reading the same Datasets through xarray-sql's adapter, with
    ddx's gradients (``ddxdb[duckdb]``)."""

    name = "duckdb"
    differentiates = True
    # ddx reads DuckDB's plans as Substrait, a portable format for query
    # plans, and DuckDB's Substrait producer raises its own error for a
    # plan it can't write.
    refusals = (DdxError, duckdb.NotImplementedException)

    def __init__(self):
        self.con = duckdb.connect()

    def load(self, name, data, **options):
        xql.register(self.con, name, data, **options)

    def execute(self, sql):
        self.con.execute(sql)

    def query(self, sql):
        # As xarray-sql's Context does, send ddx only the statements that
        # take a gradient.
        if xql.ad.calls_ad(sql):
            return self.query_all([sql])[0]
        return self.con.sql(sql).fetch_arrow_table()

    def query_all(self, sqls):
        return [
            rel.fetch_arrow_table()
            for rel in ddx_duckdb.sql_all(self.con, sqls)
        ]

    def operators(self, sql):
        (_, plan), *_ = self.con.sql(f"EXPLAIN (FORMAT JSON) {sql}").fetchall()
        return sum(_count(node) for node in json.loads(plan))


def _count(node: dict) -> int:
    return 1 + sum(_count(child) for child in node.get("children", []))


DEFAULT_HOSTS = (DataFusion.name, EinFold.name, DuckDB.name)
HOSTS = {
    host.name: host
    for host in (
        DataFusion,
        EinFold,
        DuckDB,
        DataFusionPrepared,
        CPUOptimized,
        CPUForced,
        GPUOptimized,
        GPUForced,
        SharedEinFold,
        SharedCPU,
        SharedGPU,
        DenseCPU,
        DenseGPU,
    )
}
