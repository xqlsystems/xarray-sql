# Machine learning in SQL

These scripts write three machine-learning models as SQL over tables that
[xarray-sql](../../README.md) registers from xarray Datasets, and time the
same SQL on several engines. Two related projects shape them:

- [ddx](https://github.com/xqlsystems/ddx) differentiates SQL queries: it
  turns the gradient of a query into more SQL, which trains the models.
- [einfold](https://github.com/xqlsystems/einfold) is a query optimizer
  that rewrites SQL plans into array algorithms. These models are its
  three [target demos](https://github.com/xqlsystems/einfold/blob/main/docs/demos.md),
  and this SQL is the baseline its plans must beat.

Each script checks its results against the same model in
[JAX](https://github.com/jax-ml/jax), Google's library for differentiable
array programs.

| # | Script | Model | What it measures |
|---|---|---|---|
| 01 | [`01_attention.py`](01_attention.py) | causal softmax attention, the core of a transformer, as a view (the model) in einfold's window and `GROUP BY` spellings, and a loss that reads it | the naive plan, which stores every score, for [FlashAttention](https://arxiv.org/abs/2205.14135)'s tiled plan to beat; gradients with respect to Q, K and V; masked pairs as absent rows; a control that isn't attention, the Nadaraya–Watson kernel average |
| 02 | [`02_nanogpt.py`](02_nanogpt.py) | [nanoGPT](https://github.com/karpathy/nanoGPT)'s GPT-2 model, at 1 layer and 4,416 parameters, as a view, and a cross-entropy loss that reads it, on Tiny Shakespeare | the loss on each host; training steps with SGD or, as nanoGPT trains, AdamW, every parameter table checked after every step |
| 03 | [`03_graphcast.py`](03_graphcast.py) | a miniature [GraphCast](https://arxiv.org/abs/2212.12794), DeepMind's graph neural network for weather forecasting, over [ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) reanalysis in Zarr | a global against a regional forecast; the gradient of the forecast error; the receptive field of a region |

The SQL favors legibility over speed. Each model reads like its definition,
and the slow parts are left slow on purpose: they are what ddx, xarray-sql
and einfold need to make fast.

Every result is checked before it is timed into the table. Each script
computes the same model in JAX from the same inputs and parameters, and a
mismatch fails the run.

## Running

```shell
uv run benchmarks/ml/01_attention.py
uv run benchmarks/ml/02_nanogpt.py --steps 5
uv run benchmarks/ml/03_graphcast.py --rounds 1
```

Each script carries [inline `uv` metadata](https://docs.astral.sh/uv/guides/scripts/)
and runs against the xarray-sql in this checkout, with the `ml` extra. The
options are the same in all three:

```shell
--hosts datafusion duckdb   # the engines to run on (default: all)
--sizes 64 256              # problem sizes: tokens, batch size, or mesh refinements
--reps 3                    # timed repetitions; the median is reported
--csv results.csv           # also write the results table
```

Demo 2 adds `--steps`, the number of training steps, and `--optimizer`:
`sgd` (the default), or `adamw` to train as nanoGPT does. Demo 3 adds
`--rounds`, the number of processor rounds.

The output ends with one table: a row per step, host and size, with its
median time, its error against JAX, and the number of operators in the
host's physical plan.

## The `ml` extra

```shell
pip install "xarray-sql[ml]"
```

The extra installs [ddxdb](https://pypi.org/project/ddxdb/). With it, an
`xarray_sql.Context` understands `grad`, `vjp` and `jvp` in a select list,
and `ddx_stop_gradient` as a function. A gradient-descent step on a
parameter table is one statement:

```sql
WITH pred AS (
  SELECT x.sample, SUM(x.value * w.val) AS y
  FROM x JOIN w ON x.feature = w.feature GROUP BY x.sample),
loss AS (
  SELECT AVG(power(pred.y - t.value, 2)) AS l
  FROM pred JOIN t ON pred.sample = t.sample)
SELECT feature, val - 0.1 * grad(loss, val) AS val FROM w
```

`xarray_sql.ad.sql_all(ctx, statements)` runs several such statements for
one backward pass, which is how Demo 2 updates its eleven parameter tables.

## Hosts

einfold calls the engine that runs a plan its *host*.
[`_hosts.py`](_hosts.py) defines each one in a few lines: how it registers
an `xr.Dataset` (through xarray-sql), runs SQL, and counts the operators
in its plan.

| Host | Forward pass | Gradients |
|---|---|---|
| `datafusion` | `xarray_sql.Context` | ddx |
| `duckdb` | DuckDB, through `xql.register` | not yet: ddx runs its programs on DataFusion only |

A host that cannot differentiate still runs every forward query, and its
gradient rows say `no ddx backend`. To add a host, such as DataFusion with
einfold's optimizer or DuckDB with a GPU extension, write one class in
`_hosts.py` and add it to `HOSTS`.

## Findings

Measured on 2026-10-17 on an Intel Core i7-8700 (6 cores, 12 threads)
with DataFusion 54.0.0, DuckDB 1.5.5 and ddxdb 0.3.0, one repetition each.
The demos' gradients were timed again on 2026-10-18 with ddxdb 0.3.2,
taking the median of three repetitions.

**ddx's gradients agree with JAX.** The gradients of attention with
respect to Q, K and V, in both spellings, match `jax.grad` to 3e-15 of the
largest value at every size. After each of five SGD steps, all eleven of
the GPT's parameter tables match JAX's to 1e-15, and the loss falls from
4.18 to 3.84 (each step on a different batch). With AdamW, the weights
and both of Adam's moments match a JAX transcription of
`torch.optim.AdamW` to 6e-15 after each of five steps, and that
transcription matches PyTorch exactly.

**DataFusion computes a CTE once for every place that reads it.** A
model reads its activations more than once: a layer norm reads its input
three times, a residual connection twice, a message-passing round reads
the mesh as sender, receiver and residual. DataFusion inlines each read,
so the plan grows with the model's depth; DuckDB computes a CTE that is
read more than once a single time.

| Query | DataFusion operators | DuckDB operators | DataFusion | DuckDB |
|---|---:|---:|---:|---:|
| GPT loss, 1 layer, batch 4 | 12,794 | 126 | 9.4 s | 0.058 s |
| GraphCast forecast, 1 round | 1,816 | 243 | 43 s | 17 s |
| GraphCast forecast, 2 rounds | 5,948 | 282 | 159 s | 13 s |

A GPT training step, forward and backward through ddx, takes 2.7 s on
DataFusion with SGD and 3.1 s with AdamW (3.0 s and 3.4 s with ddxdb
0.3.1, which planned the loss again for every parameter table): less than its forward query alone, because ddx stores the
result of each step of its program as a table. Computing shared
subexpressions once is the first thing xarray-sql's DataFusion path needs
for these models, and einfold should plan them as graphs rather than
trees.

**Attention stores every score.** Causal attention's forward pass at 512
tokens takes 0.68 s on DataFusion in the window spelling and 1.3 s in the
`GROUP BY` spelling; DuckDB takes 0.52 s and 0.26 s. Both grow with the
square of the tokens. The gradients take 2 to 4 times the forward pass: 2.7 s and 2.6 s on
DataFusion at 512 tokens, the same with ddxdb 0.3.1 and 0.3.2.

**A regional forecast costs what a global one does.** For the contiguous
United States, the forecast covers 55 of 2,048 grid cells. Walking the
graph back from them, the decoder needs 15 mesh nodes, the processor's
two rounds need 36 and then 82, and the encoder reads 1,103 grid cells
(54%); with one round, 576 (28%). Neither host uses any of this: the
regional forecast takes as long as the global one on both. Pushing the
region's filter back through the graph is einfold's Demo 3.

**What ddx cannot differentiate yet.** Each of these raised a typed
error naming the construct, never a wrong number:

- A `GROUP BY` key that depends on a differentiated column. einfold's
  window spelling groups by the normalizer `z`; the demos divide inside
  the sum instead, `SUM(w / z * v)`, which einfold also lists.
- `UNION ALL` between a parameter table and the loss. GraphCast
  concatenates features with `UNION ALL`, so its gradient row records the
  error. It passes once ddx differentiates a union.
- Any engine but DataFusion: ddx's DuckDB support is planned.

**xarray-sql reads NaN as missing.** A NaN in a Dataset becomes SQL
`NULL`, which `SUM` skips, so Demo 1 checks masks with an infinity: a
plan that multiplies a masked pair's zero weight by it gives NaN where
SQL gives a number.

## Next

- Part B of the NanoGPT demo: the four tricks behind a record of the
  [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt), a
  public race to train GPT-2 fastest, as a small diff to `02_nanogpt.py`.
  Its sparse optimizer updates build on `--optimizer adamw`.
- Peak memory per host, from a fresh process per measurement.
- An einfold host, once einfold produces plans.

[`_bench.py`](_bench.py) holds the shared command line, timing, checks and
results table.
