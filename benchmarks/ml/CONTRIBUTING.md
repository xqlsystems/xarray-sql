# Contributing an ML demo

A demo here is a machine-learning model written as SQL over tables that
xarray-sql registers from xarray Datasets. It is trained or differentiated
with [ddx](https://github.com/xqlsystems/ddx), which differentiates SQL
queries, and checked against the same model in
[JAX](https://github.com/jax-ml/jax), Google's library for differentiable
array programs. Each demo is also a benchmark: the same SQL runs on every
engine in [`_hosts.py`](_hosts.py), and its times are what ddx, xarray-sql
and [einfold](https://github.com/xqlsystems/einfold) (a query optimizer
that rewrites SQL plans into array algorithms) work to bring down. The
performance work is tracked in
[#285](https://github.com/xqlsystems/xarray-sql/issues/285).

Good demos are models whose SQL teaches something: a model people know
from another library, written so a reader can follow it line by line.
Some that would fit well:

- linear or logistic regression on a gridded dataset, such as ERA5
  temperatures;
- a small convolutional network over a grid, where a convolution is a
  join on shifted coordinates;
- an MLP on MNIST;
- matrix factorization, or word2vec;
- fitting a physical model's parameters to observations, such as a
  diffusion coefficient in the heat equation.

## What a demo is made of

Read [`01_attention.py`](01_attention.py) (short) and
[`02_nanogpt.py`](02_nanogpt.py) (a whole training loop) first. A demo
is one script, `NN_name.py`, numbered after the last one, with:

1. **Inline script metadata**, copied from the other demos, so that
   `uv run benchmarks/ml/NN_name.py` runs it against this checkout. Add
   any dependency the demo needs beyond `xarray-sql[ml,duckdb]`,
   `ddxdb[duckdb]` and `jax`, both there and in the `dev` dependency
   group in [`pyproject.toml`](../../pyproject.toml). CI runs the demos
   in the project's environment, which reads no script's metadata.
2. **A module docstring** that says:
   - what the model is, and links its reference implementation or paper;
   - how each piece of the model maps to SQL (`02_nanogpt.py` has a
     table for this);
   - every way the demo differs from the reference, and why.
3. **The model as SQL**, built by small Python functions that each return
   one piece of SQL, named after the pieces of the reference. In
   `02_nanogpt.py`, `block()` is `layer_norm()`, `attention()`, `mlp()`
   and `residual()`, as nanoGPT's `Block` is. Write the model as a view
   (`CREATE OR REPLACE VIEW … AS`), and the loss as a separate query
   that reads it, as a Python loss calls its model. Write the SQL a reader
   would write, even where an engine runs it slowly: making it fast is
   the engines' work, and the demos measure it.
4. **The same model in JAX**, from the same inputs and parameters, as the
   reference. Turn on 64-bit floats
   (`jax.config.update("jax_enable_x64", True)`) so the two agree to
   about 1e-15.
5. **A `main()`** that:
   - parses the shared command line from `_bench.cli` (`--hosts`,
     `--sizes`, `--reps`, `--csv`), plus the demo's own options;
   - loads the tables on each host;
   - times each step with `Report.time`, or `Report.attempt` for a
     gradient that a host may refuse, naming the refusals it expects
     (`known=`);
   - checks each result with `assert_close` and `Report.check`;
   - ends with `Report.finish`.

   A mismatch with JAX must fail the run, which `assert_close` does.
6. **A line in `DEMOS`** in [`tests/test_ml_demos.py`](../../tests/test_ml_demos.py),
   with arguments that make the demo run in a few seconds. CI's `ml demos`
   job runs every demo there, and fails if one is missing.
7. **A row in the table in [`README.md`](README.md)**, and anything it
   found under "Findings".

## Writing the SQL

- **Legibility over speed.** Write the model as its definition reads,
  even where a faster spelling exists. The slow parts are what the
  engines and optimizers need to make fast, and a demo that hides them
  measures nothing.
- **Introduce every reference before relying on it**, in docstrings,
  comments and the README: say in a few words what a project, paper,
  algorithm or term is, and link it. This rule, and the others in
  [`AGENTS.md`](../../AGENTS.md), apply to demos too.
- **Name issues only in TODOs.** Docstrings and comments name no issue
  numbers, except a workaround's `TODO(#NN): <action item>`, or
  `TODO(owner/repo#NN)` for another repository's issue.
- **Make results deterministic** with `ORDER BY`, so they can be checked
  against JAX.
- **Use no NaN as a value.** xarray-sql reads a NaN as SQL `NULL`, which
  aggregates skip, so it can't show where a value went. To test that a
  value never reaches an output, plant an infinity instead. Demo 1 does
  this to check that its causal mask leaves the masked pairs out, as rows
  the query drops (`WHERE q.t >= k.t`): an infinity at a masked position
  must not reach that output. A plan that multiplied it by a zero weight
  instead would give NaN (0 · ∞).
- **Run the same SQL on every host.** If an engine needs different
  SQL, that difference is a finding: write it up rather than fork the
  query.

## Running

```shell
uv run benchmarks/ml/NN_name.py --sizes 16 --hosts datafusion
uv run pytest tests/test_ml_demos.py   # every demo, small
```

`uv run pytest` syncs the project's environment, with the `dev` group,
which holds JAX and ddx's DuckDB backend.

Against a ddx checkout rather than the release:

1. Build ddxdb into this checkout's `.venv`, with `maturin develop
   --release` in ddx's `python/ddxdb`.
2. Run the demos with the venv's `python`:
   `.venv/bin/python benchmarks/ml/NN_name.py`. `uv run` runs a script
   with inline metadata in an environment of its own, with ddxdb from
   PyPI, and `--no-sync` doesn't change that. A `uv sync` reinstalls the
   release in `.venv` too.

## When ddx or an engine can't run your model

That's a useful result. Each of these is a finding, not a failure:

- ddx raises a typed error (a `ddxdb.DdxError`) for anything it can't
  differentiate;
- an engine may refuse a plan;
- an engine may compute the wrong number.

`Report.attempt` records a refusal in the results table as `refused:`
with the error. Unless the demo passes it in `known=`, the run then
fails, so that CI notices a refusal that's new. Then:

1. Cut the query down to a minimal repro.
2. File it on [ddx](https://github.com/xqlsystems/ddx/issues), or on
   xarray-sql if it's about how a table is read.
3. List the limit in the README's findings.
4. Pass it in `known=`, the host and the start of its error, with a
   `TODO(xqlsystems/ddx#NN)` to drop it once the issue is fixed.

A demo may land with a refused step.

## Measuring

The output ends with a table: one row per step, host and size, with the
median time, the error against JAX, and the number of operators in the
host's plan. To compare runs, pass `--reps 3 --csv results.csv`, and post
the table on [#285](https://github.com/xqlsystems/xarray-sql/issues/285),
with your CPU and the versions of DataFusion, DuckDB and ddxdb.
