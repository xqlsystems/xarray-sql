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

## `mnist_mlp.py` — an MNIST MLP as relational tensor algebra

An MLP (196 -> 32 tanh -> 10 softmax on 2x2-pooled 14x14 MNIST) built on one
idea: **a neural net is a chain of tensor contractions (einsums), and an einsum
over coordinate-indexed arrays *is* relational algebra.**

```
C[i,k] = sum_j A[i,j] * B[j,k]   <=>   JOIN A, B ON A.j = B.j
                                       GROUP BY i, k -> SUM(A.val * B.val)
```

Contracting a shared index is a join on it followed by a grouped `SUM` over the
indices that survive. In xarray-sql an array indexed by named dims is a table
keyed by those dims, so **the dimension names are the join keys**.

**The whole network is one relation.** Two moves get there:

- **Bias folded into the weights (an `nn.Linear`).** Each layer's bias is the
  weight of a constant-`1` input, kept as the extra row `inp = width` of the same
  weight array — so a layer is a single matrix.
- **A `layer` dimension.** Every layer's weight lives in one
  `weight(layer, inp, out)` array, so the forward/backward filter on the `layer`
  *column* instead of referencing a table per layer.

So **the architecture is data**: the whole model is one `xr.Dataset` with a
`layer` dim, registered via `from_dataset`. The dim sizes are the layer widths
and the number of layers is the depth — differing neuron counts are just
differing sizes, NaN-padded in the dense array and dropped on the way in (the
relational form is naturally ragged). Change `WIDTHS` (e.g. `196, 64, 32, 10`)
and the same code trains the deeper net.

A small `contract()` helper turns an einsum spec into one query, and a single
generic loop trains a net of any shape:

- **forward** contracts the activation with `weight WHERE layer = L`, adds the
  bias row, `tanh` (softmax on the last layer).
- **backward is the *same* operator with indices transposed** — the VJP of a
  contraction is a contraction — accumulated into one `gweight` relation, with
  `grad(tanh(z), z)` for the only genuinely-calculus part. Even the update is one
  query over the whole `weight` relation. Linear algebra is joins; the
  derivatives of the nonlinearities are `grad`.

Everything stays relational: every stage is an inspectable table (`a1`, `delta2`,
`gweight`, …); the only hand-written gradient is softmax + cross-entropy's
`delta = softmax - onehot`. Even the training metrics are a table — each logged
step appends a `(step, loss, train_acc, test_acc)` row to a `metrics` relation
rather than a Python list (NN training produces a lot of such data; it belongs in
rows). Evaluation is SQL too (a forward pass + `ROW_NUMBER()` argmax), and the
trained model, predictions, and metrics all come **back out as xarray** via
`to_dataset`. Reaches ~83% test accuracy over 60 steps. Downloads MNIST on first
run.

This is not a numpy replacement — relational matmul carries join overhead a BLAS
inner product doesn't. What it buys is a fully declarative, inspectable pipeline
whose data side is chunked xarray (parallel over the batch, larger-than-memory).
The *outer* training loop stays in Python because the steps must be materialised
between iterations: a multi-layer net can't be one recursive CTE (the recursive
relation may be referenced only once, but the weights are used several times per
step), and unrolling the steps as non-recursive CTEs blows up exponentially
(DataFusion inlines CTEs). The thin loop does exactly that materialisation; all
the maths stays in SQL.
