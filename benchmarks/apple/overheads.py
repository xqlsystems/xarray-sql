"""Measure training overhead and CPU partition counts independently of GPU math.

Run from the checkout root with ``python -m benchmarks.apple.overheads``.
Times include checked Arrow results. Registration and in-place writes have
separate measurements. References use NumPy float64 arithmetic.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import xarray as xr
from einfold.datafusion import EinFold

from xarray_sql import Context, ad

_PRODUCT = """SELECT a.i,b.k,SUM(a.v*b.v) AS v FROM a JOIN b ON a.j=b.j
GROUP BY a.i,b.k ORDER BY a.i,b.k"""
_SHARED = """WITH p AS (
SELECT a.i,b.k,SUM(a.v*b.v) AS v FROM a JOIN b ON a.j=b.j GROUP BY a.i,b.k)
SELECT x.i,x.k,x.v+y.v AS v FROM p x JOIN p y ON x.i=y.i AND x.k=y.k
ORDER BY x.i,x.k"""
_LOSS = """WITH p AS (
SELECT a.i,b.k,SUM(a.v*b.v) AS v FROM a JOIN b ON a.j=b.j GROUP BY a.i,b.k),
loss AS (SELECT SUM(v*v)/2 AS l FROM p)
SELECT j,k,grad(loss,v) AS v FROM b ORDER BY j,k"""
_ATTENTION = """WITH scores AS (
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


def _time(run, check, reps):
    check(run())
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        got = run()
        times.append(time.perf_counter() - start)
        check(got)
    return {"seconds": statistics.median(times), "samples": times}


def _check(want):
    def check(rows):
        np.testing.assert_allclose(
            rows["v"].to_numpy().reshape(want.shape), want, rtol=1e-9, atol=1e-9
        )

    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument(
        "--threads", nargs="+", type=int, default=[1, 10, 20, 30, 60]
    )
    parser.add_argument("--thread-tokens", type=int, default=2048)
    parser.add_argument("--label", default="current")
    parser.add_argument("--skip-thread-sweep", action="store_true")
    parser.add_argument("--only-thread-sweep", action="store_true")
    args = parser.parse_args()
    report = {
        "environment": {
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "reps": args.reps,
            "label": args.label,
        },
        "results": [],
    }

    def record(task, run, check, **fields):
        result = {"task": task, **fields, **_time(run, check, args.reps)}
        report["results"].append(result)
        print(task, fields, f"{result['seconds'] * 1000:.3f} ms", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    rng = np.random.default_rng(804)
    if not args.only_thread_sweep:
        for rows in (32, 4096, 65536):
            data = _dataset(np.ones((rows, 16)), ("i", "j"))
            ctx = Context()

            def register(ctx=ctx, data=data):
                ctx.deregister_table("w")
                ctx.from_dataset("w", data)

            record("register_dataset", register, lambda _: None, rows=rows)
            query = ctx.prepare("SELECT SUM(v) AS v FROM w")
            replacement = data * 2
            record(
                "prepared_write_and_sum",
                lambda query=query, replacement=replacement: query(
                    w=replacement
                ),
                _check(np.array([rows * 32.0])),
                rows=rows,
            )

            def replace_and_sum(ctx=ctx, data=replacement):
                ctx.deregister_table("w")
                ctx.from_dataset("w", data)
                return ctx.sql("SELECT SUM(v) AS v FROM w").to_arrow_table()

            record(
                "register_and_sum",
                replace_and_sum,
                _check(np.array([rows * 32.0])),
                rows=rows,
            )
        data = _dataset(np.ones((4096, 16)), ("i", "j"))
        ctx = Context()

        def fine_chunks():
            ctx.deregister_table("w")
            ctx.from_dataset("w", data, chunks={"i": 1, "j": 1})

        record(
            "register_fine_chunks",
            fine_chunks,
            lambda _: None,
            rows=4096,
            columns=16,
        )
        assert ctx.sql("SELECT SUM(v) AS v FROM w").to_pylist() == [
            {"v": 65536.0}
        ]
        rng = np.random.default_rng(804)
        a, b = rng.normal(size=(128, 32)), rng.normal(size=(32, 16))
        for host in ("datafusion", "einfold"):
            ctx = Context()
            if host == "einfold":
                ctx = ctx.with_extensions(EinFold())
            ctx.from_dataset("a", _dataset(a, ("i", "j")))
            ctx.from_dataset("b", _dataset(b, ("j", "k")))
            for name, sql, want in [
                ("product", _PRODUCT, a @ b),
                ("repeated_cte", _SHARED, 2 * (a @ b)),
            ]:
                record(
                    "sql",
                    lambda ctx=ctx, sql=sql: ctx.sql(sql).to_arrow_table(),
                    _check(want),
                    host=host,
                    query=name,
                )
                prepared = ctx.prepare(sql)
                record(
                    "prepared", prepared, _check(want), host=host, query=name
                )
            prepared = ctx.prepare(_LOSS)
            record(
                "gradient_context_prepared",
                prepared,
                _check(a.T @ (a @ b)),
                host=host,
            )
            prepared = ad.prepare(ctx, [_LOSS])
            record(
                "gradient_prepared",
                lambda prepared=prepared: prepared()[0],
                _check(a.T @ (a @ b)),
                host=host,
            )
    if args.skip_thread_sweep:
        return
    n = args.thread_tokens
    q, k, v = (rng.normal(size=(n, 64)) for _ in range(3))
    score = q @ k.T / 8
    score[np.triu_indices(n, 1)] = -np.inf
    weights = np.exp(score - np.max(score, axis=1, keepdims=True))
    want = weights / weights.sum(axis=1, keepdims=True) @ v
    for threads in args.threads:
        for host in ("datafusion", "einfold"):
            ctx = Context(threads=threads)
            if host == "einfold":
                ctx = ctx.with_extensions(EinFold())
            for name, values, dims in [
                ("q", q, ("i", "d")),
                ("k", k, ("j", "d")),
                ("v", v, ("j", "d")),
            ]:
                ctx.from_dataset(name, _dataset(values, dims))
            record(
                "thread_sweep",
                lambda ctx=ctx: ctx.sql(_ATTENTION).to_arrow_table(),
                _check(want),
                host=host,
                threads=threads,
                tokens=n,
            )


if __name__ == "__main__":
    main()
