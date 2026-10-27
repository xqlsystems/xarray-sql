"""The ML demos in ``benchmarks/ml`` run, at their smallest sizes.

Each demo checks its results against JAX and exits with an error on a
mismatch, so this runs each one and checks that it succeeds. It needs the
``ml`` extra, ddx's DuckDB backend and JAX, which the ``dev`` dependency
group installs.

A new demo adds a line to ``DEMOS``: its script, and for each run of
it, the arguments that make it small enough to run in a few seconds. A
demo fails if a host refuses a step the demo doesn't expect it to.
"""

import pathlib
import subprocess
import sys

import pytest

DEMOS_DIR = pathlib.Path(__file__).parent.parent / "benchmarks" / "ml"

DEMOS = {
    "01_attention.py": [["--sizes", "16"]],
    "02_nanogpt.py": [
        ["--steps", "1"],
        ["--steps", "1", "--optimizer", "adamw"],
    ],
}


def test_every_demo_is_listed():
    scripts = {p.name for p in DEMOS_DIR.glob("[0-9][0-9]_*.py")}
    assert scripts == set(DEMOS)


@pytest.mark.parametrize(
    "script, args",
    [(script, args) for script in sorted(DEMOS) for args in DEMOS[script]],
    ids=lambda v: " ".join(v) if isinstance(v, list) else v,
)
def test_the_demo_runs_and_matches_jax(script, args):
    for module in ("ddxdb", "duckdb", "jax"):
        pytest.importorskip(module)
    result = subprocess.run(
        [sys.executable, str(DEMOS_DIR / script), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
