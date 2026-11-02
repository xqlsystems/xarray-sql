"""Measure SQL -> DataFusion -> EinFold -> MLX, including Arrow results.

Run from the repository root with:
  .venv/bin/python -m benchmarks.apple.benchmark --output benchmarks/apple/results.json

Every host executes identical SQL. GPU kernels synchronize before returning
Arrow buffers. Registration is excluded; SQL planning, native/Python buffer
copies, execution, sorting and Arrow materialization are included. Warmups
are excluded from medians. ddx gradients are checked against closed-form
NumPy derivatives; attention and products against float64 NumPy results.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import xarray as xr
from einfold.datafusion import EinFold

from xarray_sql import Context, ad
from xarray_sql.mlx import MLX

PRODUCT = """SELECT a.i, b.k, SUM(a.v*b.v) AS v FROM a JOIN b ON a.j=b.j
GROUP BY a.i,b.k ORDER BY a.i,b.k"""
LOSS = """WITH pred AS (
SELECT a.i,b.k,SUM(a.v*b.v) AS p FROM a JOIN b ON a.j=b.j GROUP BY a.i,b.k),
loss AS (SELECT SUM(p*p)/2 AS l FROM pred)
SELECT j,k,grad(loss,v) AS v FROM b ORDER BY j,k"""
ATTENTION = """WITH scores AS (
SELECT q.i,k.j,SUM(q.v*k.v)/8.0 AS s FROM q JOIN k ON q.d=k.d
WHERE q.i>=k.j GROUP BY q.i,k.j),
weights AS (SELECT i,j,exp(s-MAX(s) OVER (PARTITION BY i)) AS w FROM scores),
norm AS (SELECT i,SUM(w) AS z FROM weights GROUP BY i)
SELECT w.i,v.d,SUM(w.w/n.z*v.v) AS v FROM weights w
JOIN v ON w.j=v.j JOIN norm n ON w.i=n.i GROUP BY w.i,v.d ORDER BY w.i,v.d"""


def _dataset(a, dims):
    return xr.Dataset(
        {"v": (dims, a)},
        coords={d: np.arange(n) for d, n in zip(dims, a.shape)},
    )


def _context(host, min_gpu_ops, threads=None):
    ctx = Context(threads=threads)
    extension = None
    if host == "einfold":
        extension = EinFold()
    elif host in ("einfold_blas", "einfold_mlx"):
        extension = MLX(
            precision="float32" if host == "einfold_mlx" else "float64",
            min_gpu_ops=min_gpu_ops,
        )
    if extension is not None:
        ctx = ctx.with_extensions(extension)
    return ctx, extension


def _measure(run, want, reps, tolerance):
    run()  # warm up compilation and the runtime
    times = []
    error = 0.0
    for _ in range(reps):
        start = time.perf_counter()
        rows = run()
        times.append(time.perf_counter() - start)
        got = rows["v"].to_numpy().reshape(want.shape)
        np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance)
        error = max(error, float(np.max(np.abs(got - want), initial=0)))
    return {
        "seconds": statistics.median(times),
        "samples": times,
        "max_abs_error": error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hosts",
        nargs="+",
        choices=["datafusion", "einfold", "einfold_blas", "einfold_mlx"],
        default=["datafusion", "einfold", "einfold_blas", "einfold_mlx"],
    )
    parser.add_argument(
        "--matrix-rows", nargs="+", type=int, default=[256, 2048]
    )
    parser.add_argument(
        "--attention-tokens", nargs="+", type=int, default=[256, 1024]
    )
    parser.add_argument(
        "--gradient-rows", nargs="+", type=int, default=[256, 2048]
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="DataFusion target partitions; default is CPU count",
    )
    parser.add_argument("--min-gpu-ops", type=int, default=1 << 33)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--outputs", type=int, default=128)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(20261031)
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "gpu": mx.device_info(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in [
                "datafusion",
                "einfold",
                "ddxdb",
                "mlx",
                "numpy",
                "pyarrow",
            ]
        },
        "min_gpu_ops": args.min_gpu_ops,
        "threads": args.threads or os.cpu_count(),
        "matrix_features": args.features,
        "matrix_outputs": args.outputs,
        "reps": args.reps,
        "warmups": 1,
        "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"),
    }
    report = {"environment": metadata, "results": []}
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    problems = []
    for n in args.matrix_rows:
        a, b = (
            rng.normal(size=(n, args.features)),
            rng.normal(size=(args.features, args.outputs)),
        )
        problems.append(
            (
                "matmul",
                n,
                [("a", a, ("i", "j")), ("b", b, ("j", "k"))],
                PRODUCT,
                a @ b,
            )
        )
    for n in args.attention_tokens:
        q, k, v = (rng.normal(size=(n, 64)) for _ in range(3))
        score = q @ k.T / 8.0
        score[np.triu_indices(n, 1)] = -np.inf
        weights = np.exp(score - np.max(score, axis=1, keepdims=True))
        want = weights / weights.sum(axis=1, keepdims=True) @ v
        problems.append(
            (
                "attention",
                n,
                [
                    ("q", q, ("i", "d")),
                    ("k", k, ("j", "d")),
                    ("v", v, ("j", "d")),
                ],
                ATTENTION,
                want,
            )
        )
    for n in args.gradient_rows:
        a, b = rng.normal(size=(n, 64)), rng.normal(size=(64, 32))
        problems.append(
            (
                "gradient",
                n,
                [("a", a, ("i", "j")), ("b", b, ("j", "k"))],
                LOSS,
                a.T @ (a @ b),
            )
        )
    for task, n, tables, sql, want in problems:
        for host in args.hosts:
            ctx, extension = _context(host, args.min_gpu_ops, args.threads)
            for name, a, dims in tables:
                ctx.from_dataset(name, _dataset(a, dims))
            tolerance = 3e-4 if host == "einfold_mlx" else 1e-9
            modes = ["sql", "prepared"] if task == "gradient" else ["sql"]
            for mode in modes:
                if mode == "prepared":
                    prepared = ad.prepare(ctx, [sql])
                    run = lambda prepared=prepared: prepared()[0]
                else:
                    run = lambda ctx=ctx, sql=sql: ctx.sql(sql).to_arrow_table()
                result = _measure(run, want, args.reps, tolerance)
                result.update(task=task, size=n, host=host, mode=mode)
                if isinstance(extension, MLX):
                    result["kernels"] = extension.statistics()
                report["results"].append(result)
                print(
                    f"{task:10} {n:5} {host:14} {mode:8} {result['seconds'] * 1000:9.3f} ms error={result['max_abs_error']:.3g}",
                    flush=True,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    if profiler:
        profiler.disable()
        profiler.dump_stats(args.profile)


if __name__ == "__main__":
    main()
