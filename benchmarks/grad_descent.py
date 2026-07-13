# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "xarray_sql",
#   "xarray",
#   "numpy",
# ]
#
# [tool.uv.sources]
# xarray_sql = { path = "..", editable = true }
# ///
"""Gradient descent as a single declarative SQL query.

Fits a line ``y ~= a*x + b`` by minimising the mean squared error — with the
**entire training loop expressed as one recursive CTE**, no Python iteration.

Two pieces:

1. **grad compiles the update rule.** ``differentiate_sql`` turns the per-row
   loss into the symbolic derivative *as SQL text* — the autograd engine acting
   as a calculus compiler:

       da = differentiate_sql("(y-(a*x+b))^2", "a")   # -> "-2*((a*x+b)-y)*x", etc.

2. **A recursive CTE is the optimiser.** ``params(step, a, b)`` starts at one
   row and each recursion appends the next generation, descending along the
   gradient (``AVG`` of the compiled rule over the data):

       params.a - lr * AVG(da)

   So the whole loop — gradient, update, and iteration — is declarative SQL;
   the optimisation trajectory is the rows of one query.

Why two pieces instead of ``grad(...)`` directly inside the recursion? ``grad``
needs the Substrait round-trip, and Substrait has no recursion — so ``grad``
can't live inside a recursive CTE (tracked in #194 / a follow-up). Differentiating
once to plain SQL sidesteps that: the recursive query contains no ``grad`` marker.

Run standalone:

    uv run benchmarks/grad_descent.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import xarray_sql as xql

# Per-row loss r^2 with residual r = y - (a*x + b), over columns a, b, x, y.
RESIDUAL = "(y - (a * x + b))"
LOSS = f"{RESIDUAL} * {RESIDUAL}"
COLUMNS = ["a", "b", "x", "y"]
LR = 0.4
STEPS = 200


def main() -> None:
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0.0, 1.0, n)
    a_true, b_true = 2.0, -1.0
    y = a_true * x + b_true + rng.normal(0.0, 0.01, n)

    ctx = xql.XarrayContext()
    ctx.from_dataset(
        "d",
        xr.Dataset(
            {"x": (("i",), x), "y": (("i",), y)}, coords={"i": np.arange(n)}
        ),
        chunks={"i": n},
    )

    # grad compiles the per-row update rule to SQL, once.
    da = xql.differentiate_sql(LOSS, "a", COLUMNS)
    db = xql.differentiate_sql(LOSS, "b", COLUMNS)
    print(f"d(loss)/da = {da}")
    print(f"d(loss)/db = {db}\n")

    # The entire training loop is one declarative recursive query: each step
    # appends the next generation, descending along the SQL-computed gradient.
    trajectory = ctx.sql(
        f"""
        WITH RECURSIVE params(step, a, b) AS (
          SELECT 0 AS step, CAST(0.0 AS DOUBLE) AS a, CAST(0.0 AS DOUBLE) AS b
          UNION ALL
          SELECT params.step + 1               AS step,
                 params.a - {LR} * AVG({da})    AS a,
                 params.b - {LR} * AVG({db})    AS b
          FROM params CROSS JOIN d
          WHERE params.step < {STEPS}
          GROUP BY params.step, params.a, params.b
        )
        SELECT step, a, b FROM params ORDER BY step
        """
    ).to_pandas()

    print("trajectory (every 40th generation):")
    print(trajectory.iloc[::40].to_string(index=False))

    a, b = float(trajectory["a"].iloc[-1]), float(trajectory["b"].iloc[-1])
    a_ols, b_ols = np.polyfit(x, y, 1)
    print(
        f"\nSQL gradient descent:  a={a:.4f}  b={b:.4f}  ({len(trajectory)} generations)"
    )
    print(f"least-squares (numpy): a={a_ols:.4f}  b={b_ols:.4f}")
    assert abs(a - a_ols) < 1e-2 and abs(b - b_ols) < 1e-2
    print(
        "\nOK: a single recursive-CTE query fit the line to the OLS solution."
    )


if __name__ == "__main__":
    main()
