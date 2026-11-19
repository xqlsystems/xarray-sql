"""Profile one warm compact SQL loss/backward call, outside benchmarks.

GPU wait time includes pending device computation. NumPy view creation is
reported separately; these are host-observed phases, not GPU kernel events.
The profile leaves input parameters unchanged and is not a training run.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "benchmarks/ml"))
from _hosts import DenseCPU, DenseGPU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--cache-gib", type=float, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["XQL_DENSE_CACHE_BYTES"] = str(int(args.cache_gib * (1 << 30)))
    spec = importlib.util.spec_from_file_location(
        "dense_profile_model", _ROOT / "benchmarks/ml/02_nanogpt.py"
    )
    n = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n)
    n.LAYERS, n.EMBED, n.HEADS, n.HEAD, n.HIDDEN, n.CONTEXT = (
        12,
        768,
        12,
        64,
        3072,
        args.context,
    )
    host = DenseGPU() if args.gpu else DenseCPU()
    parameters = n.init_params(50257, np.random.default_rng(0))
    for name, values in parameters.items():
        host.load(name, n.as_table(name, values))
    host.load(
        "batch", next(n.batches(n.shakespeare(), 1, np.random.default_rng(1)))
    )
    host.execute(n.model())
    queries = n.step_statements(0.01)
    host.query(n.loss_statement())
    host.query_all(queries)
    phases = {
        "input_conversion_seconds": 0.0,
        "device_wait_seconds": 0.0,
        "numpy_view_seconds": 0.0,
    }
    if args.gpu:
        runtime = host.executor._runtime
        array, numpy = runtime.array, runtime.numpy

        def import_array(value):
            start = time.perf_counter()
            result = array(value)
            phases["input_conversion_seconds"] += time.perf_counter() - start
            return result

        def export_array(value):
            if runtime.owns(value):
                start = time.perf_counter()
                mx.eval(value)
                phases["device_wait_seconds"] += time.perf_counter() - start
            start = time.perf_counter()
            result = numpy(value)
            phases["numpy_view_seconds"] += time.perf_counter() - start
            return result

        runtime.array, runtime.numpy = import_array, export_array
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    host.query(n.loss_statement())
    rows = host.query_all(queries)
    profiler.disable()
    elapsed = time.perf_counter() - start
    profiler.dump_stats(str(args.output.with_suffix(".pstats")))
    report = io.StringIO()
    pstats.Stats(profiler, stream=report).strip_dirs().sort_stats(
        "cumtime"
    ).print_stats(40)
    args.output.with_suffix(".txt").write_text(report.getvalue())
    record = {
        "diagnostic": True,
        "host": host.name,
        "context": args.context,
        "parameters": sum(v.size for v in parameters.values()),
        "cache_gib": args.cache_gib,
        "paired_seconds": elapsed,
        "public_output_bytes": sum(t.nbytes for t in rows),
        "phases": phases,
        "statistics": host.measurements(),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
