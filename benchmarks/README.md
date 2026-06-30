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

## `mnist_mlp.py` — train an MNIST MLP classifier in SQL

A one-hidden-layer neural network (196 -> 32 tanh -> 10 softmax, on 2x2-pooled
14x14 MNIST) where **every gradient is computed in SQL** and the whole model —
with its entire training history — lives in a single table.

The model is one append-only table `model(step, layer, i, j, val)`: every
parameter is a row, tagged by which generation (`step`) it belongs to. **A
training step never mutates anything; it appends the next generation's rows.**
`WHERE step = N` is the model at iteration N, and the full trajectory is the
table. Each step is a *single* SQL statement that reads the current generation
and writes the next — reverse-mode autodiff as relational algebra:

- **matmul = join + `GROUP BY SUM`** — a layer's pre-activation is
  `SUM(input * weight)` grouped by (sample, unit).
- **local derivatives = `grad()`** — the hidden activation's Jacobian is
  `grad(tanh(z), z)`, the autograd feature doing the calculus per (sample, unit).
- **cotangent propagation = join**, **parameter gradients = join + `GROUP BY
  AVG`**, and the update `w - lr*g` is emitted as the next generation's rows.

The images are registered as xarray (the library's core); evaluation is SQL too
(a forward pass with `ROW_NUMBER()` for the argmax). The only hand-written
gradient is softmax + cross-entropy's `delta = softmax - onehot` (softmax couples
classes through a per-sample normaliser, which an aggregate `grad` does not
cross). Reaches ~83% test accuracy over 60 steps (~140s on a laptop — the
parameter updates run in SQL and every generation is kept as rows, so it trades
speed for a fully relational, fully inspectable training history). Downloads
MNIST on first run.

Why is the *outer* loop still Python rather than one recursive query (like
`grad_descent.py`)? A recursive CTE may reference the recursive relation only
once, but a 2-layer net uses the current weights several times per step (W1 and
W2 forward, W2 again in backprop), so it can't be a single recursive statement.
Training is also sequential and reuses each step's result, so steps must be
*materialised* between iterations — which is exactly what the thin loop does
(append a generation, then query it). All the maths stays in SQL; Python only
sequences the steps.
