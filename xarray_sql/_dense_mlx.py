"""MLX array operations for EinFold's compact numerical programs.

[MLX](https://github.com/ml-explore/mlx) evaluates the arithmetic chosen by
[EinFold](https://github.com/xqlsystems/einfold). Compiled functions take
every numerical leaf as an argument, so a cached function never captures
parameter values from an earlier training step. Coordinates stay on the CPU.
"""

from __future__ import annotations

import os
import string
from collections import Counter

import mlx.core as mx
import numpy as np
from einfold.dense_program import _OPS, _align, _Decline, _op


def _remainder(a, b):
    result = mx.remainder(a, b)
    return mx.where((result != 0) & ((a < 0) != (b < 0)), result - b, result)


_FUNCTIONS = {
    "+": mx.add,
    "-": mx.subtract,
    "*": mx.multiply,
    "/": mx.divide,
    "%": _remainder,
    "=": mx.equal,
    "!=": mx.not_equal,
    "<": mx.less,
    "<=": mx.less_equal,
    ">": mx.greater,
    ">=": mx.greater_equal,
    "and": mx.logical_and,
    "or": mx.logical_or,
    "not": mx.logical_not,
    "neg": mx.negative,
    "where": mx.where,
    "abs": mx.abs,
    "sqrt": mx.sqrt,
    "exp": mx.exp,
    "erf": mx.erf,
    "ln": mx.log,
    "log": mx.log10,
    "tanh": mx.tanh,
    "isnan": mx.isnan,
    "iszero": lambda x: x == 0,
    "signum": mx.sign,
    "sign": mx.sign,
    "sin": mx.sin,
    "cos": mx.cos,
    "power": mx.power,
    "pow": mx.power,
    "floor": mx.floor,
    "ceil": mx.ceil,
    "minimum": mx.minimum,
    "maximum": mx.maximum,
    "float64": lambda x: x.astype(mx.float32),
    "int64": lambda x: x.astype(mx.int64),
    "bool": lambda x: x.astype(mx.bool_),
}


class _Runtime:
    def __init__(self):
        if os.environ.get("MLX_ENABLE_TF32") != "0":
            raise ValueError("launch Python with MLX_ENABLE_TF32=0")
        self.compiled = {}
        self.counts = Counter()
        self.tracing = False

    def owns(self, array):
        return isinstance(array, mx.array)

    def boolean(self, value):
        self.counts["gpu_status_readbacks"] += 1
        return bool(value.item())

    def status(self, values, flags):
        status = mx.array(True)
        for flag in flags:
            status = mx.logical_and(status, mx.all(self._mx(flag)))
        for value in values:
            status = mx.logical_and(
                status, mx.all(mx.isfinite(self._mx(value)))
            )
        return status

    def validate(self, values, flags):
        status = self.status(values, flags)
        mx.eval(values, status)
        if not self.boolean(status):
            raise ValueError(
                "training update must be rectangular, nonnullable and finite"
            )

    def compile_training(self, function):
        self.counts["gpu_training_compilations"] += 1
        return mx.compile(function)

    def finish_training(self, arrays, status):
        self.counts["gpu_training_executions"] += 1
        mx.eval(arrays, status)
        if not self.boolean(status):
            raise ValueError(
                "training update must be rectangular, nonnullable and finite"
            )

    def indices(self, values):
        if values.size and (
            int(values.min()) < -(1 << 63) or int(values.max()) >= (1 << 63)
        ):
            raise _Decline("device index is outside exact int64 range")
        self.counts["gpu_input_arrays"] += 1
        self.counts["gpu_input_bytes"] += values.nbytes
        return mx.array(values, dtype=mx.int64)

    def random_key(self, seed, step):
        return mx.bitwise_xor(
            mx.random.key(seed), mx.array([0, step], dtype=mx.uint32)
        )

    def dropout(self, key, shapes, probability):
        keys = mx.random.split(key, len(shapes))
        return [
            mx.random.bernoulli(1 - probability, shape=shape, key=k).astype(
                mx.float32
            )
            / (1 - probability)
            for k, shape in zip(keys, shapes)
        ]

    def memory(self):
        return {
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        }

    def copy(self, value):
        return mx.array(value)

    def numpy(self, value):
        if self.owns(value):
            if self.tracing:
                raise _Decline(
                    "compiled numerical graph requires a host conversion"
                )
            self.counts["gpu_readbacks"] += 1
            self.counts["gpu_readback_bytes"] += value.size * value.itemsize
        return np.asarray(value)

    def array(self, value):
        array = np.asarray(value)
        if array.dtype.kind == "f":
            # ddx uses the extreme float64 value as a masked MAX sentinel.
            # It stays beyond every supported finite float32 input. An
            # observable infinity is rejected at the final SQL boundary.
            if array.ndim == 0 and abs(array) == np.finfo(np.float64).max:
                return mx.array(float("-inf") if array < 0 else float("inf"))
            finite = np.isfinite(array)
            if (array.ndim and not finite.all()) or np.any(
                np.abs(array[finite]) > np.finfo(np.float32).max
            ):
                raise _Decline(
                    f"GPU input outside finite float32 range: {array.dtype}, shape={array.shape}, finite={finite.sum()}/{array.size}"
                )
            self.counts["gpu_input_arrays"] += 1
            self.counts["gpu_input_bytes"] += array.nbytes
            return mx.array(array, dtype=mx.float32)
        return array

    def _mx(self, value):
        return value if self.owns(value) else mx.array(value)

    def pointwise(self, tensor):
        leaves, instructions, seen = [], [], {}

        def describe(node):
            if node in seen:
                return seen[node]
            if node.kind in _OPS and node.cached is None and len(seen) < 64:
                children = tuple(describe(a) for a in node.args)
                instruction = node.kind, children
            else:
                leaves.append(_align(node.get(), node.axes, tensor.axes))
                instruction = "leaf", len(leaves) - 1
            at = len(instructions)
            seen[node] = at
            instructions.append(instruction)
            return at

        describe(tensor)
        signature = tuple(instructions)
        if not any(self.owns(a) for a in leaves):
            return None
        if signature not in self.compiled:

            def execute(*args):
                values = []
                for kind, children in signature:
                    if kind == "leaf":
                        values.append(args[children])
                    else:
                        values.append(
                            _FUNCTIONS[kind](*(values[c] for c in children))
                        )
                return values[-1]

            self.compiled[signature] = mx.compile(execute)
            self.counts["gpu_fused_programs"] += 1
        self.counts["gpu_fused_executions"] += 1
        return self.compiled[signature](*(self._mx(a) for a in leaves))

    def take(self, data, indexes, axis):
        return (
            mx.take(self._mx(data), self._mx(indexes), axis=axis)
            if self.owns(data) or self.owns(indexes)
            else np.take(data, indexes, axis=axis)
        )

    def indexes(self, domain, values):
        if domain.dtype.kind not in "iu" or not len(domain):
            raise _Decline("GPU index domain must contain integers")
        if int(domain.min()) < -(1 << 63) or int(domain.max()) >= (1 << 63):
            raise _Decline("GPU index domain is outside exact int64 range")
        domain = domain.astype(np.int64, copy=False)
        order = np.argsort(domain)
        sorted_domain = mx.array(domain[order])
        at = mx.minimum(mx.searchsorted(sorted_domain, values), len(domain) - 1)
        indexes = mx.take(mx.array(order), at)
        return indexes, mx.take(mx.array(domain), indexes) == values

    def index(self, data, indexes):
        return (
            self._mx(data)[tuple(self._mx(i) for i in indexes)]
            if self.owns(data) or any(self.owns(i) for i in indexes)
            else data[tuple(indexes)]
        )

    def contract(self, arrays, labels, output):
        converted = [self._mx(a) for a in arrays]
        if not any(mx.issubdtype(a.dtype, mx.floating) for a in converted):
            for array in arrays:
                if self.owns(array):
                    if array.dtype != mx.bool_:
                        raise _Decline(
                            "GPU integer contraction requires a count mask"
                        )
                elif np.any((array < -1) | (array > 1)):
                    raise _Decline(
                        "GPU integer contraction requires unit factors"
                    )
            dimensions = {
                label: size
                for a, names in zip(converted, labels)
                for label, size in zip(names, a.shape)
            }
            used = tuple(sorted(dimensions))
            value = mx.array(1, dtype=mx.int64)
            for a, names in zip(converted, labels):
                ordered = [n for n in used if n in names]
                a = a.transpose(tuple(names.index(n) for n in ordered))
                a = a.reshape(
                    tuple(dimensions[n] if n in names else 1 for n in used)
                )
                value = value * a.astype(mx.int64)
            value = mx.sum(
                value,
                axis=tuple(i for i, n in enumerate(used) if n not in output),
            )
            remaining = [n for n in used if n in output]
            return value.transpose(tuple(remaining.index(n) for n in output))
        equation = (
            ",".join(
                "".join(string.ascii_letters[i] for i in label)
                for label in labels
            )
            + "->"
            + "".join(string.ascii_letters[i] for i in output)
        )
        # Mask/count operands are promoted before contraction; a boolean
        # reduction otherwise counts presence rather than SQL rows.
        arrays = [a.astype(mx.float32) for a in converted]
        self.counts["gpu_contractions"] += 1
        return mx.einsum(equation, *arrays)

    def broadcast(self, data, source, target, shape):
        return mx.broadcast_to(_align(self._mx(data), source, target), shape)

    def extreme(self, kind, data, source, mask, mask_axes, used, kept):
        shape = tuple(len(a) for a in used)
        if any(size == 0 for size in shape):
            raise _Decline("empty GPU extremum")
        data = self.broadcast(data, source, used, shape)
        mask = self.broadcast(mask, mask_axes, used, shape)
        data = mx.where(
            mask, data, -float("inf") if kind == "max" else float("inf")
        )
        axes = tuple(i for i, a in enumerate(used) if a not in kept)
        return (mx.max if kind == "max" else mx.min)(data, axis=axes)

    def scatter(self, data, indexes, size, *, count_bound):
        data = self._mx(data)
        integer = data.dtype == mx.int64
        if integer:
            if count_bound > np.iinfo(np.int32).max:
                raise _Decline("GPU integer scatter exceeds exact int32 range")
            data = data.astype(mx.int32)
        result = mx.zeros((data.shape[0], size), dtype=data.dtype)
        self.counts["gpu_scatter_reductions"] += 1
        result = result.at[
            mx.arange(data.shape[0])[:, None], mx.array(indexes)
        ].add(data)
        return result.astype(mx.int64) if integer else result

    def merge(self, parts, keys, key, axes, *, flags=None):
        result = mx.zeros(tuple(len(a) for a in axes), dtype=mx.float32)
        for part in parts:
            source = tuple(part.columns[k].tensor.axis for k in keys)
            indexes = [
                np.searchsorted(a.values, s.values)
                for a, s in zip(axes, source)
            ]
            shape = tuple(len(a) for a in source)
            if flags is not None:
                mask = (
                    part.present
                    if flags == "rows"
                    else _op("and", part.present, part.columns[key].valid)
                )
                data = mx.ones(shape, dtype=mx.float32)
            else:
                col = part.columns[key]
                value = col.tensor.get()
                if self.owns(value) and not mx.issubdtype(
                    value.dtype, mx.floating
                ):
                    raise _Decline(
                        "GPU integer partial aggregate requires relational execution"
                    )
                mask = _op("and", part.present, col.valid)
                data = self.broadcast(value, col.tensor.axes, source, shape)
            valid = self.broadcast(mask.get(), mask.axes, source, shape)
            data = mx.where(valid, data, 0)
            if axes:
                grid = [
                    mx.array(
                        i.reshape(
                            tuple(
                                len(i) if j == at else 1
                                for j in range(len(axes))
                            )
                        )
                    )
                    for at, i in enumerate(indexes)
                ]
                result = result.at[tuple(grid)].add(data)
            else:
                result = result + data
        self.counts["gpu_union_reductions"] += 1
        return result
