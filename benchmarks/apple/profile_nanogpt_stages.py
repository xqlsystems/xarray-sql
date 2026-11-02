"""Attribute one SQL nanoGPT backward pass to stages and physical plans.

Diagnostic timings include instrumentation and are separate from performance
comparisons. Saved SQL is ddx's generated program, not a substitute model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from ddxdb._ddxdb import to_sql

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "benchmarks/ml"))
from _hosts import SharedCPU, SharedGPU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "xql_profile_nanogpt", _ROOT / "benchmarks/ml/02_nanogpt.py"
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
    host = SharedGPU() if args.gpu else SharedCPU()
    for name, values in n.init_params(50257, np.random.default_rng(0)).items():
        host.load(name, n.as_table(name, values))
    host.load(
        "batch", next(n.batches(n.shakespeare(), 1, np.random.default_rng(1)))
    )
    host.execute(n.model())
    backend = host.executor._backend
    records = []
    materialize = backend.materialize_compiled
    compile_plan = backend.compile

    def save():
        args.output.write_text(
            json.dumps(
                {
                    "host": host.name,
                    "context": args.context,
                    "instrumented": True,
                    "stages": records,
                    "statistics": host.measurements(),
                },
                indent=2,
            )
            + "\n"
        )

    def compile(plan):
        start = time.perf_counter()
        compiled = compile_plan(plan)
        compiled._profile_compile_seconds = time.perf_counter() - start
        return compiled

    def execute(name, plan, compiled):
        print("BEGIN", name, flush=True)
        start = time.perf_counter()
        materialize(name, plan, compiled)
        elapsed = time.perf_counter() - start
        record = {
            "name": name,
            "seconds": elapsed,
            "compile_seconds": getattr(compiled, "_profile_compile_seconds", 0),
            "deferred": compiled.deferred,
            "graph": compiled.graph is not None,
            "rows": None if compiled.batch is None else compiled.batch.num_rows,
            "bytes": None if compiled.batch is None else compiled.batch.nbytes,
        }
        if elapsed > 0.2:
            record["sql"] = to_sql(plan, "datafusion")
            if compiled.plan is not None:
                record["physical"] = compiled.plan.display_indent()
        records.append(record)
        print("END", name, elapsed, record["rows"], flush=True)
        save()

    backend.compile = compile
    backend.materialize_compiled = execute
    print("FORWARD", host.query(n.loss_statement()).to_pydict(), flush=True)
    host.query_all(n.step_statements(0.01))
    save()
    for r in sorted(records, key=lambda r: r["seconds"], reverse=True)[:20]:
        print(r["name"], r["seconds"], r["rows"], flush=True)


if __name__ == "__main__":
    main()
