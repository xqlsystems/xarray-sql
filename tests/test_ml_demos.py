"""The ML demos in ``benchmarks/ml`` run, at their smallest sizes.

Each demo checks its results against JAX and exits with an error on a
mismatch, so this runs each one and checks that it succeeds. It needs the
``ml`` extra, DuckDB and JAX; CI's ``ml demos`` job installs them.

A new demo adds a line to ``DEMOS``: its script, and the arguments that
make it small enough to run in a few seconds.
"""

import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("ddxdb")
pytest.importorskip("duckdb")
pytest.importorskip("jax")

DEMOS_DIR = pathlib.Path(__file__).parent.parent / "benchmarks" / "ml"

DEMOS = {
    "01_attention.py": ["--sizes", "16"],
    "02_nanogpt.py": ["--steps", "1"],
}


def test_every_demo_is_listed():
    scripts = {p.name for p in DEMOS_DIR.glob("[0-9][0-9]_*.py")}
    assert scripts == set(DEMOS)


@pytest.mark.parametrize("script", sorted(DEMOS))
def test_the_demo_runs_and_matches_jax(script):
    result = subprocess.run(
        [sys.executable, str(DEMOS_DIR / script), *DEMOS[script]],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
