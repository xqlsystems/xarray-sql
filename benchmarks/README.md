# Benchmarks & demos

Standalone scripts that exercise xarray-sql against real data. Each declares its
own dependencies inline (PEP 723) and points `xarray_sql` at this checkout, so
they run with no setup:

```bash
uv run benchmarks/grad_era5.py
```

## `grad_era5.py` — differentiable SQL over ARCO-ERA5

Demonstrates the autograd feature on a real climate archive
([ARCO-ERA5](https://github.com/google-research/arco-era5), read anonymously
from GCS — needs `gcsfs` and network access).

The key idea: a physical quantity is written as an **analytic SQL formula** over
ERA5 variables, and `grad(...)` differentiates that formula **symbolically**,
evaluated at every grid cell. Because each row is an independent point, this is
the relational equivalent of `jax.vmap(jax.grad(f))`. It is *not* a finite-
difference spatial gradient — `grad(f(u, v), u)` is the exact partial derivative
of `f`.

Two worked cases, each checked against an analytic reference:

| Quantity | SQL | Derivative | Check |
| --- | --- | --- | --- |
| Wind speed | `sqrt(power(u,2) + power(v,2))` | `grad(speed, u) = u/speed` | exact |
| Saturation vapour pressure | `A*exp(B*tc/(tc+C))` | `grad(e_s, T)` | closed-form Clausius-Clapeyron slope |

Each query round-trips back to an `xarray.Dataset` via `.to_dataset(...)`.

## `grad_descent.py` — gradient descent as one declarative SQL query

Fits a line `y ~= a*x + b` by minimising the mean squared error, with the
**entire training loop expressed as a single recursive CTE** — no Python
iteration. Two pieces:

- **`grad` compiles the update rule.** `xql.differentiate_sql(loss, "a", cols)`
  turns the per-row loss into its symbolic derivative *as SQL text* — the
  autograd engine as a calculus compiler.
- **A recursive CTE is the optimiser.** `params(step, a, b)` starts at one row
  and each recursion appends the next generation, descending along the gradient
  (`AVG` of the compiled rule over the data):

  ```sql
  WITH RECURSIVE params(step, a, b) AS (
    SELECT 0, 0.0, 0.0
    UNION ALL
    SELECT params.step + 1, params.a - lr*AVG(da), params.b - lr*AVG(db)
    FROM params CROSS JOIN d WHERE params.step < STEPS
    GROUP BY params.step, params.a, params.b)
  SELECT * FROM params ORDER BY step
  ```

So gradient, update, and iteration are all declarative SQL; the trajectory is
the rows of one query. The fit matches numpy's least-squares solution.
Self-contained (no network).

(Why differentiate to text instead of `grad(...)` inside the recursion? `grad`
needs the Substrait round-trip, and Substrait has no recursion — so a `grad`
marker can't live inside a recursive CTE. Differentiating once to plain SQL
sidesteps that.)

