"""Owned array state for training programs written in SQL.

EinFold (https://github.com/xqlsystems/einfold) executes the numerical
plans and ddx (https://github.com/xqlsystems/ddx) supplies every gradient.
This opt-in session keeps complete, nonnullable rectangular tables as
arrays between calls. Use a dedicated Context; ordinary query APIs retain
their independent live-Dataset and Arrow snapshot contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from datafusion import SessionContext
from ddxdb import prepare as _prepare
from einfold.dense_program import (
    _align,
    _axes,
    _Axis,
    _Column,
    _const,
    _Decline,
    _expr,
    _is,
    _read_dataset,
    _Sorted,
    _Table,
    _Tensor,
)

from xarray_sql.dense import (
    DenseExecutor,
    _Compiled,
    _DenseBackend,
    _DenseGraph,
)
from xarray_sql.shared import _snapshot

__all__ = ["TrainingResult", "TrainingSession"]


def _compact(table):
    if isinstance(table, _Sorted):
        sorts, table = table.sorts, table.table
        if not isinstance(table, _Table):
            raise _Decline("training requires a rectangular ordered result")
        order = []
        for expr in sorts:
            column = _expr(expr.expr(), table)
            axis = column.tensor.axis
            if axis is None or not _is(column.valid, True):
                raise _Decline(
                    "training order requires nonnullable coordinate keys"
                )
            if axis in order:
                continue
            indexes = np.argsort(axis.values, kind="stable")
            if not expr.ascending():
                indexes = indexes[::-1]
            if not np.array_equal(indexes, np.arange(len(axis))):
                replacement = _Axis(axis.values[indexes])
                table = table.take(axis, indexes, (replacement,))
                axis = replacement
            order.append(axis)
        table = _Table(
            _axes(order, table.axes), table.columns.items(), table.present
        )
    if not isinstance(table, _Table):
        raise _Decline("training requires a rectangular result")
    return table


@dataclass(frozen=True)
class _DeviceTable:
    table: _Table
    schema: pa.Schema

    def select(self, names):
        return _DeviceTable(
            _Table(
                self.table.axes,
                [((None, n), self.table.resolve(None, n)) for n in names],
                self.table.present,
            ),
            pa.schema([self.schema.field(n) for n in names]),
        )


class _TrainingBackend(_DenseBackend):
    def returns_rows_compiled(self, plan, compiled):
        table = _compact(compiled.run(self.owner))
        if any(len(axis) == 0 for axis in table.axes):
            return False
        flags = table.present.get()
        runtime = self.owner._runtime
        if runtime is not None and runtime.owns(flags):
            if self.owner._tracing:
                raise _Decline(
                    "compiled training needs static structural checks"
                )
            return runtime.boolean(flags.any())
        return bool(np.any(flags))


class TrainingResult:
    """A completed update's loss and immutable device table snapshots.

    Full arrays reach Arrow only through ``export``. The loss may be an
    MLX scalar; converting it to a Python number synchronizes that scalar.
    """

    def __init__(self, session, loss, outputs, version, measurements=None):
        self._session, self._outputs = session, outputs
        runtime = session.executor._runtime
        self.loss = (
            runtime.copy(loss)
            if runtime is not None
            else np.array(loss, copy=True)
        )
        self.version = version
        self.measurements = measurements or {}

    def export(self, *names):
        """Export independent Arrow snapshots of selected intermediate tables."""
        return self._session._export({n: self._outputs[n] for n in names})

    def loss_value(self):
        """Read the loss scalar, recording this explicit synchronization."""
        runtime = self._session.executor._runtime
        if runtime is not None and runtime.owns(self.loss):
            runtime.counts["gpu_loss_readbacks"] += 1
            self.measurements["gpu_loss_readbacks"] = (
                self.measurements.get("gpu_loss_readbacks", 0) + 1
            )
        return float(self.loss)


class TrainingSession:
    """Own numerical tables and execute SQL updates without Arrow intermediates.

    ``bind`` imports a private copy. Later caller writes do not change it;
    call ``bind`` again for explicit replacement. All persisted tables
    must retain complete coordinate domains, non-NULL finite values and
    their original schemas. An update replaces every assigned table
    together after validation. A failed update leaves state unchanged.
    """

    def __init__(
        self, ctx, *, precision="float32", max_cached_bytes=16 << 30, **options
    ):
        self.executor = DenseExecutor(
            ctx,
            precision=precision,
            fallback=False,
            max_cached_bytes=max_cached_bytes,
            **options,
        )
        self.executor._backend = _TrainingBackend(self.executor)
        self.executor._tracing = False
        self._tables = {}
        self._closed = False
        self.version = 0
        self._queries = {}
        self._index_domains = {}
        self._dropouts = ()
        self._random_step = 0

    def _invoke(self, function):
        if self._closed:
            raise RuntimeError("training session is closed")

        def execute():
            self.executor._store.update(
                {n: t.table for n, t in self._tables.items()}
            )
            return function()

        return self.executor._invoke(execute, None, {})

    def _seal(self, table, schema):
        table = _compact(table)
        if len(schema) != len(table.columns):
            raise _Decline("training result schema changed")
        columns = []
        for field, column in zip(schema, table.columns.values()):
            tensor = column.tensor
            value = tensor.get()
            if tensor.axis is not None or tensor.host_only:
                runtime = self.executor._runtime
                if runtime is not None and runtime.owns(value):
                    raise _Decline(
                        "device-dependent columns cannot be frozen as static coordinates"
                    )
                frozen = _Tensor(
                    "array",
                    tensor.axes,
                    value=np.asarray(value).copy(),
                    axis=tensor.axis,
                )
            else:
                frozen = _Tensor("device", tensor.axes, value=value)
            frozen.domain = tensor.domain
            columns.append(((None, field.name), _Column(frozen, column.valid)))
        return _DeviceTable(_Table(table.axes, columns, table.present), schema)

    def _values_and_flags(self, tables):
        values, flags = [], []
        for item in tables:
            table = item.table
            flags.append(table.present.get())
            for col in table.columns.values():
                flags.append(col.valid.get())
                if col.tensor.axis is None:
                    values.append(col.tensor.get())
        return values, flags

    def _validate(self, tables):
        runtime = self.executor._runtime
        values, flags = self._values_and_flags(tables)
        if runtime is not None:
            runtime.validate(values, flags)
        elif not all(np.all(f) for f in flags) or not all(
            np.isfinite(v).all() for v in values
        ):
            raise ValueError(
                "training update must be rectangular, nonnullable and finite"
            )

    def bind(self, name, dataset=None):
        """Import or explicitly replace a table, independent of caller buffers."""
        with self.executor._lock:
            return self._bind(name, dataset)

    def _bind(self, name, dataset):
        ctx = self.executor.ctx
        if dataset is None:
            dataset = ctx._registered_datasets[name]
        private = dataset.copy(deep=True)
        if any(size == 0 for size in private.sizes.values()):
            raise ValueError(
                "training tables require nonempty coordinate domains"
            )
        if any(
            tuple(v.dims) != tuple(private.sizes)
            for v in private.data_vars.values()
        ):
            raise ValueError(
                "owned training columns must share the complete table dimensions"
            )
        if name not in ctx._registered_datasets:
            ctx.deregister_table(name)
            ctx.from_dataset(name, private)
        schema = ctx.table(name).schema()
        source = _read_dataset(name, private)
        if schema.names != [n for _, n in source.columns] or any(
            f.type != pa.from_numpy_dtype(c.tensor.value.dtype)
            for f, c in zip(schema, source.columns.values())
        ):
            raise ValueError("replacement changes the training table schema")

        def execute():
            item = self._seal(source, schema)
            for column, (reference, dim) in self._index_domains.get(
                name, {}
            ).items():
                domain = (
                    self._tables[reference].table.resolve(None, dim).tensor.axis
                )
                col = item.table.resolve(None, column)
                values = col.tensor.get()
                if (
                    domain is None
                    or values.dtype.kind not in "iu"
                    or not np.isin(values, domain.values).all()
                ):
                    raise ValueError(
                        "index values are outside the declared coordinate domain"
                    )
                tensor = _Tensor(
                    "device",
                    col.tensor.axes,
                    value=self.executor._runtime.indices(values),
                )
                tensor.domain = domain
                item.table.columns[None, column] = _Column(tensor, col.valid)
            self._validate([item])
            return item

        item = self._invoke(execute)
        old = self._tables.get(name)
        if old is not None and (
            not old.schema.equals(item.schema)
            or len(old.table.axes) != len(item.table.axes)
            or any(
                not np.array_equal(a.values, b.values)
                for a, b in zip(old.table.axes, item.table.axes)
            )
        ):
            raise ValueError(
                "replacement changes the training table schema or coordinate domains"
            )
        self._tables[name] = item

    def index_domains(self, name, columns):
        """Declare GPU integer input columns bounded by owned coordinate axes.

        The next ``bind`` validates the actual integer values and uploads
        them as dynamic device inputs. ``columns`` maps each field to a
        ``(reference_table, coordinate)`` pair.
        """
        if self.executor._runtime is None:
            raise ValueError(
                "device index domains require float32 GPU execution"
            )
        self._index_domains[name] = dict(columns)
        self.executor._epoch += 1

    def dropout(self, names, *, probability, seed, step=0):
        """Generate owned mask tables on GPU using an explicit seed and step.

        Each named table must have one numerical field. The counter advances
        only after a successful update. Save seed and next counter with the
        checkpoint; identical inputs reproduce masks after resume.
        """
        if self.executor._runtime is None or not 0 <= probability < 1:
            raise ValueError(
                "GPU dropout requires float32 execution and probability in [0,1)"
            )
        if not 0 <= seed < 2**32 or not 0 <= step < 2**32:
            raise ValueError("dropout seed and step must be uint32 values")
        for name in names:
            columns = [
                c
                for c in self._tables[name].table.columns.values()
                if c.tensor.axis is None
            ]
            if len(columns) != 1:
                raise ValueError(
                    "each dropout table needs exactly one numerical field"
                )
        self._dropouts = tuple(sorted(names))
        self._dropout_probability, self._random_seed, self._random_step = (
            probability,
            seed,
            step,
        )
        self.executor._epoch += 1

    def _make_masks(self, key):
        templates = [self._tables[name] for name in self._dropouts]
        cols = [
            next(c for c in t.table.columns.values() if c.tensor.axis is None)
            for t in templates
        ]
        arrays = self.executor._runtime.dropout(
            key,
            [tuple(len(a) for a in c.tensor.axes) for c in cols],
            self._dropout_probability,
        )
        results = {}
        for name, item, col, array in zip(
            self._dropouts, templates, cols, arrays
        ):
            columns = [
                (
                    k,
                    _Column(
                        _Tensor("device", c.tensor.axes, value=array),
                        _const(True),
                    )
                    if c is col
                    else c,
                )
                for k, c in item.table.columns.items()
            ]
            results[name] = _DeviceTable(
                _Table(item.table.axes, columns), item.schema
            )
        return results

    def prepare_step(
        self,
        *,
        gradients,
        stages,
        updates,
        assignments,
        loss_column="l",
        compile=False,
    ):
        """Prepare one shared ddx backward followed by ordinary SQL statements.

        ``gradients``, ``stages`` and ``updates`` map result names to SQL.
        ``assignments`` maps each owned state table to ``(result, columns)``.
        All update SELECTs read the same old state. The primal loss is read
        from ddx's saved output, without an independent forward query.
        """
        step = _TrainingStep(
            self, gradients, stages, updates, assignments, loss_column
        )
        step.compile = compile
        return step

    def query(self, sql):
        """Evaluate SQL against current owned state and export its result."""
        if sql not in self._queries:
            self._queries[sql] = _DenseGraph(self.executor, sql)
        graph = self._queries[sql]

        def execute():
            table = graph.run()
            return _snapshot(table.arrow(graph.compiled[""].schema))

        return self._invoke(execute)

    def _export(self, tables):
        def execute():
            self.executor._counts["training_exports"] += len(tables)
            return {
                n: _snapshot(t.table.arrow(t.schema)) for n, t in tables.items()
            }

        return self._invoke(execute)

    def export(self, *names):
        """Export independent Arrow snapshots of owned tables."""
        return self._export(
            {n: self._tables[n] for n in (names or self._tables)}
        )

    def close(self):
        """Release this session's owned arrays."""
        with self.executor._lock:
            self._tables.clear()
            self._closed = True

    def statistics(self):
        return self.executor.statistics() | {"training_version": self.version}


class _TrainingStep:
    def __init__(
        self, session, gradients, stages, updates, assignments, loss_column
    ):
        self.session, self.assignments, self.loss_column = (
            session,
            dict(assignments),
            loss_column,
        )
        self.gradients, self.stages, self.updates = (
            dict(gradients),
            dict(stages),
            dict(updates),
        )
        self.prepared = _prepare(
            session.executor._backend, list(gradients.values())
        )
        self.compiled = {}

    def _run(self, sql):
        owner = self.session.executor
        if sql not in self.compiled or self.compiled[sql].epoch != owner._epoch:
            self.compiled[sql] = _Compiled(
                owner, SessionContext.sql(owner.ctx, sql)
            )
        compiled = self.compiled[sql]
        return self.session._seal(compiled.run(owner), compiled.schema)

    def _store(self, name, item):
        owner = self.session.executor
        owner._store[name] = item.table
        owner._stub(name, item.schema)

    def _build(self):
        session = self.session
        results = {}
        with self.prepared.run() as rewritten:
            if not hasattr(self.prepared, "values"):
                raise ImportError(
                    "rebuild accelerators/ddx_math for prepared primal outputs"
                )
            values = [
                v
                for v in self.prepared.values
                if tuple(v.columns) == (self.loss_column,)
            ]
            if len(values) != 1:
                raise ValueError(
                    "training requires one unambiguous scalar loss output"
                )
            loss_table = self._run(values[0].query())
            for name, sql in zip(self.gradients, rewritten.statements):
                results[name] = self._run(sql)
                self._store(name, results[name])
            for name, sql in self.stages.items():
                results[name] = self._run(sql)
                self._store(name, results[name])
            for name, sql in self.updates.items():
                results[name] = self._run(sql)
            next_state = {
                name: results[result].select(columns)
                for name, (result, columns) in self.assignments.items()
            }
            for name, item in next_state.items():
                old = session._tables[name]
                if old.schema.names != item.schema.names or any(
                    a.type != b.type for a, b in zip(old.schema, item.schema)
                ):
                    raise ValueError("update changes a state table schema")
                if len(old.table.axes) != len(item.table.axes) or any(
                    not np.array_equal(a.values, b.values)
                    for a, b in zip(old.table.axes, item.table.axes)
                ):
                    raise ValueError("update changes a state coordinate domain")
                shape = tuple(len(a) for a in item.table.axes)
                columns = []
                for key, col in item.table.columns.items():
                    if col.tensor.axis is None:
                        value = col.tensor.get()
                        runtime = session.executor._runtime
                        value = (
                            runtime.broadcast(
                                value, col.tensor.axes, item.table.axes, shape
                            )
                            if runtime is not None
                            else np.broadcast_to(
                                _align(value, col.tensor.axes, item.table.axes),
                                shape,
                            )
                        )
                        col = _Column(
                            _Tensor("device", item.table.axes, value=value),
                            col.valid,
                        )
                    columns.append((key, col))
                next_state[name] = _DeviceTable(
                    _Table(item.table.axes, columns, item.table.present),
                    item.schema,
                )
            return loss_table, results, next_state

    def _pack(self, tables):
        arrays, layouts = [], {}
        for name, item in tables.items():
            columns = []
            for key, col in item.table.columns.items():
                t = col.tensor
                if t.axis is None and not t.host_only:
                    index = len(arrays)
                    arrays.append(t.get())
                    columns.append((key, t.axes, t.domain, index, None))
                else:
                    columns.append((key, t.axes, t.domain, None, col))
            layouts[name] = (item.table.axes, item.schema, columns)
        return arrays, layouts

    @staticmethod
    def _unpack(arrays, layouts):
        tables = {}
        for name, (axes, schema, columns) in layouts.items():
            values = []
            for key, local, domain, index, static in columns:
                if index is None:
                    value = static
                else:
                    tensor = _Tensor("device", local, value=arrays[index])
                    tensor.domain = domain
                    value = _Column(tensor, _const(True))
                values.append((key, value))
            tables[name] = _DeviceTable(_Table(axes, values), schema)
        return tables

    def _compiled_build(self):
        session, owner = self.session, self.session.executor
        runtime = owner._runtime
        if runtime is None:
            raise ValueError("compiled training requires float32 MLX execution")
        sources = {
            n: t
            for n, t in sorted(session._tables.items())
            if n not in session._dropouts
        }
        for item in sources.values():
            if any(
                c.tensor.axis is None and c.tensor.host_only
                for c in item.table.columns.values()
            ):
                raise ValueError(
                    "compiled integer inputs require explicit index_domains"
                )
        arrays, layouts = self._pack(sources)
        if session._dropouts:
            arrays.append(
                runtime.random_key(session._random_seed, session._random_step)
            )
        guard = (owner._epoch, owner._backend.session_key())
        if getattr(self, "_compiled_guard", None) != guard:

            def function(*args):
                bound = self._unpack(
                    args[:-1] if session._dropouts else args, layouts
                )
                masks = (
                    session._make_masks(args[-1]) if session._dropouts else {}
                )
                owner._store.clear()
                owner._store.update({n: t.table for n, t in bound.items()})
                owner._store.update({n: t.table for n, t in masks.items()})
                owner._tracing = True
                runtime.tracing = True
                try:
                    loss, results, next_state = self._build()
                    results.update(masks)
                    packed, self._output_layouts = self._pack(
                        {
                            ("loss",): loss,
                            **{("result", n): t for n, t in results.items()},
                            **{("state", n): t for n, t in next_state.items()},
                        }
                    )
                    values, flags = session._values_and_flags(
                        [loss, *results.values()]
                    )
                    status = runtime.status(values, flags)
                    return packed, status
                finally:
                    owner._tracing = False
                    runtime.tracing = False

            self._function = runtime.compile_training(function)
        packed, status = self._function(*arrays)
        runtime.finish_training(packed, status)
        restored = self._unpack(packed, self._output_layouts)
        loss = restored[("loss",)]
        results = {k[1]: v for k, v in restored.items() if k[0] == "result"}
        next_state = {k[1]: v for k, v in restored.items() if k[0] == "state"}
        owner._refresh()
        self._compiled_guard = (owner._epoch, owner._backend.session_key())
        return loss, results, next_state

    def __call__(self):
        session = self.session

        def execute():
            counters = session.executor.statistics()
            if session._dropouts and session._random_step >= 2**32:
                raise ValueError("GPU dropout counter is exhausted")
            if getattr(self, "compile", False):
                loss_table, results, next_state = self._compiled_build()
            else:
                masks = (
                    session._make_masks(
                        session.executor._runtime.random_key(
                            session._random_seed, session._random_step
                        )
                    )
                    if session._dropouts
                    else {}
                )
                session.executor._store.update(
                    {n: t.table for n, t in masks.items()}
                )
                loss_table, results, next_state = self._build()
                results.update(masks)
                session._validate([loss_table, *results.values()])
            # Validation proved these full domains have no NULLs.
            next_state = {
                n: _DeviceTable(
                    _Table(
                        t.table.axes,
                        [
                            (k, _Column(c.tensor, _const(True)))
                            for k, c in t.table.columns.items()
                        ],
                    ),
                    t.schema,
                )
                for n, t in next_state.items()
            }
            loss = loss_table.table.resolve(None, self.loss_column).tensor.get()
            session._tables.update(next_state)
            session.version += 1
            session._random_step += 1
            session.executor._counts["training_steps"] += 1
            measured = session.executor.statistics()
            delta = {
                name: measured.get(name, 0) - counters.get(name, 0)
                for name in (
                    "gpu_readback_bytes",
                    "gpu_input_bytes",
                    "gpu_status_readbacks",
                    "training_exports",
                    "gpu_training_compilations",
                )
            }
            return TrainingResult(
                session, loss, results, session.version, delta
            )

        return session._invoke(execute)
