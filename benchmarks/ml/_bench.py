"""Timing, checking and reporting for the ML demos.

Each demo is a script with the same command line::

    uv run benchmarks/ml/01_attention.py --hosts datafusion duckdb \\
        --sizes 64 256 --reps 3 --csv attention.csv

It runs every step on every chosen host (an engine that runs SQL; see
``_hosts.py``) and size, checks each result against the same model in
JAX, Google's library for differentiable array programs, and ends with
one table: a row per step, host
and size, with the median time and whether the result matched. Checks
come first: a wrong answer fails the run, however fast it was.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from _hosts import HOSTS, Host


def cli(description: str, sizes: Sequence[int], size_name: str = "n"):
    """The shared command line, for a demo to extend and parse."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=list(HOSTS),
        choices=list(HOSTS),
        help="engines to run on (default: all)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=list(sizes),
        help=f"problem sizes, as {size_name} (default: %(default)s)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="timed repetitions per step; the median is reported",
    )
    parser.add_argument("--csv", help="also write the results table here")
    return parser


def hosts(names: Sequence[str]) -> list[Host]:
    """A fresh instance of each named host."""
    return [HOSTS[name]() for name in names]


def assert_close(name: str, got, want, *, rtol=1e-9, atol=1e-9) -> float:
    """Assert two arrays match, NaN for NaN. Returns the largest error,
    relative to the largest value."""
    got, want = np.asarray(got), np.asarray(want)
    np.testing.assert_allclose(
        got, want, rtol=rtol, atol=atol, equal_nan=True, err_msg=name
    )
    finite = np.isfinite(want)
    scale = np.max(np.abs(want[finite]), initial=0.0) or 1.0
    return float(np.max(np.abs(got - want)[finite], initial=0.0) / scale)


@dataclass
class Report:
    """The demo's results: one row per step, host and size."""

    demo: str
    size_name: str
    reps: int = 1
    rows: list[dict] = field(default_factory=list)

    def time(self, step: str, host: Host, size: int, run: Callable, *args):
        """Run ``run(*args)`` ``reps`` times, record the median, and return
        the last result."""
        times = []
        for _ in range(self.reps):
            start = time.perf_counter()
            result = run(*args)
            times.append(time.perf_counter() - start)
        self.rows.append(
            {
                "demo": self.demo,
                "step": step,
                "host": host.name,
                self.size_name: size,
                "seconds": statistics.median(times),
                "check": "",
            }
        )
        print(
            f"  {step:<28} {host.name:<11} {self.size_name}={size:<6} "
            f"{statistics.median(times):8.3f}s",
            flush=True,
        )
        return result

    def check(self, error: float) -> None:
        """Mark the last row's result as checked, with its error (from
        :func:`assert_close`)."""
        self.rows[-1]["check"] = f"ok ({error:.0e})"

    def skip(self, step: str, host: Host, size: int, why: str) -> None:
        """Record a step the host cannot run, and why."""
        self.rows.append(
            {
                "demo": self.demo,
                "step": step,
                "host": host.name,
                self.size_name: size,
                "seconds": None,
                "check": why,
            }
        )

    def note(self, key: str, value) -> None:
        """Attach a measurement besides time to the last row."""
        self.rows[-1][key] = value

    def finish(self, csv_path: str | None = None) -> None:
        """Print the table, and write it as CSV if asked."""
        columns = list(dict.fromkeys(k for row in self.rows for k in row))
        print(f"\n| {' | '.join(columns)} |")
        print(f"|{'|'.join('---' for _ in columns)}|")
        for row in self.rows:
            cells = [_cell(row.get(c)) for c in columns]
            print(f"| {' | '.join(cells)} |")
        if csv_path:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(self.rows)
            print(f"\nwrote {csv_path}", file=sys.stderr)


def _cell(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
