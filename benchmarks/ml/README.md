# Machine learning in SQL

These scripts write machine-learning models as SQL over tables that
[xarray-sql](../../README.md) registers from xarray Datasets, and time the
same SQL on several engines. Two related projects shape them:

- [ddx](https://github.com/xqlsystems/ddx) differentiates SQL queries: it
  turns the gradient of a query into more SQL, which trains the models.
- [einfold](https://github.com/xqlsystems/einfold) is a query optimizer
  that rewrites SQL plans into array algorithms. Two of these models
  are its [target demos](https://github.com/xqlsystems/einfold/blob/main/docs/demos.md),
  and the demos run on DataFusion with einfold as one of their engines.

Each script checks its results against the same model in
[JAX](https://github.com/jax-ml/jax), Google's library for differentiable
array programs.

| # | Script | Model | What it measures |
|---|---|---|---|
| 01 | [`01_attention.py`](01_attention.py) | causal softmax attention, the core of a transformer, as a view (the model) in einfold's window and `GROUP BY` spellings, and a loss that reads it | the naive plan, which stores every score, against [FlashAttention](https://arxiv.org/abs/2205.14135)'s tiled plan, which einfold finds from the same SQL; gradients with respect to Q, K and V; masked pairs as absent rows; a control that isn't attention: the [Nadaraya–Watson estimator](https://en.wikipedia.org/wiki/Kernel_regression#Nadaraya%E2%80%93Watson_kernel_regression), an average of nearby points weighted by a Gaussian kernel, which einfold must fuse with the same rules |
| 02 | [`02_nanogpt.py`](02_nanogpt.py) | [nanoGPT](https://github.com/karpathy/nanoGPT)'s GPT-2 model, at 1 layer and 4,416 parameters, as CTEs, and a cross-entropy loss that reads them, on Tiny Shakespeare | the loss on each host; training steps with SGD or, as nanoGPT trains, AdamW, every parameter table checked after every step |

The SQL favors legibility over speed. Each model reads like its definition,
and the slow parts are left slow on purpose: they are what ddx, xarray-sql
and einfold need to make fast.

To add a demo, see [CONTRIBUTING.md](CONTRIBUTING.md). The work to make
these models fast is tracked in
[#285](https://github.com/xqlsystems/xarray-sql/issues/285), which lists
places to start.

Every result is checked before it is timed into the table. Each script
computes the same model in JAX from the same inputs and parameters, and a
mismatch fails the run.

## Running

```shell
uv run benchmarks/ml/01_attention.py
uv run benchmarks/ml/02_nanogpt.py --steps 5
```

Each script carries [inline `uv` metadata](https://docs.astral.sh/uv/guides/scripts/)
and runs against the xarray-sql in this checkout, with the `ml` extra. The
options are the same in each:

```shell
--hosts datafusion duckdb   # the engines to run on (default: all)
--sizes 64 256              # problem sizes: tokens, or batch size
--reps 3                    # timed repetitions; the median is reported
--csv results.csv           # also write the results table
```

Demo 2 adds `--steps`, the number of training steps, and `--optimizer`:
`sgd` (the default), or `adamw` to train as nanoGPT does, and the
model's size: `--layers`, `--embed`, `--heads` and `--context`, nanoGPT's
`n_layer`, `n_embd`, `n_head` and `block_size` (1, 16, 2 and 16 by
default; nanoGPT's Shakespeare model is 6, 384, 6 and 256).

The output ends with one table: a row per step, host and size, with its
median time, its error against JAX, and the number of operators in the
host's physical plan.

To run against a ddx checkout rather than the release, build ddxdb into
this checkout's `.venv` (`maturin develop --release` in ddx's
`python/ddxdb`) and run the scripts with the venv's `python`, as in
`.venv/bin/python benchmarks/ml/01_attention.py`. `uv run` runs a script
with inline metadata in an environment of its own, with ddxdb from PyPI,
and `--no-sync` doesn't change that.

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
| `einfold` | `xarray_sql.Context`, with einfold as its query planner | ddx, whose steps einfold plans too |
| `duckdb` | DuckDB, through `xql.register` | ddx, with the `duckdb` extra of ddxdb, which pins the DuckDB it supports |

A host that cannot differentiate still runs every forward query, and its
gradient rows say `no ddx backend`. A gradient that ddx, or the engine
it plans with, refuses is reported as `refused:` with the error, and a
refused training step ends that host's training run. Each demo lists the
refusals it expects, by host and error. Any other refusal fails the run,
so a refusal that's new is noticed. To add a host, such as DuckDB with a
GPU extension, write one class in `_hosts.py` and add it to `HOSTS`.

## Findings

Measured on 2026-10-17 on an Intel Core i7-8700 (6 cores, 12 threads)
with DataFusion 54.0.0, DuckDB 1.5.5 and ddxdb 0.3.0, one repetition each.
The GPT and attention demos were timed again on 2026-10-21 with
DataFusion 55.0.0 and ddxdb 0.4.0, after xarray-sql began registering
small in-memory Datasets as DataFusion in-memory tables. Both demos were
timed again on 2026-10-25 with DataFusion 55.0.0, DuckDB 1.5.5 and
ddxdb 0.4.2, and on DuckDB on 2026-10-27 with DuckDB 1.5.6 and ddxdb
0.4.3; where those runs give new numbers, they are given here.

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
three times, a residual connection twice. DataFusion inlines each read,
so the plan grows with the model's depth; DuckDB computes a CTE that is
read more than once a single time.

| Query | DataFusion operators | DuckDB operators | DataFusion | DuckDB |
|---|---:|---:|---:|---:|
| GPT loss, 1 layer, batch 4 | 13,756 | 126 | 3.0 s | 0.063 s |

Setting up a lazy read of a Dataset takes xarray-sql about 1.5 ms on its
own (1.7 ms for a `SELECT *` of 1,040 rows, against 0.26 ms for an
in-memory table). In the GPT's forward pass, which reads its Datasets
2,989 times, the lazy reads cost 6.7 s of its 9.2 s, about 2.2 ms each. A small
in-memory Dataset is now an in-memory table, which takes a quarter of a
millisecond per read. A GPT training step, forward and backward through
ddx, takes 0.65 s on DataFusion with SGD and 0.74 s with AdamW, a quarter
of its forward query alone (1.2 s and 1.55 s with ddxdb 0.3.3 on
DataFusion 54). ddx builds the gradient program on the first step, which
takes 1.2 s, and runs it again on later ones. It plans each CTE the loss
reads twice on its own, and stores the result of each step of its program
as a table, so it computes each shared subexpression once. Doing the same
for forward queries is the first thing xarray-sql's DataFusion path needs
for these models, and einfold should plan them as graphs rather than
trees.

**DataFusion spends the GPT's time planning, not running.** Measured on
2026-10-24 with DataFusion 55.0.0. Of the 1-layer loss's 2.5 s, turning
its SQL into a physical plan takes nearly all; the same physical plan,
built once and run again, takes 1.8 ms (with one partition, and the same
tables). A second layer multiplies the plan by about 100, to 1.74
million logical nodes and 313,789 table scans: the SQL planner takes 8.6
s to build it, and planning it runs out of 8 GB. DuckDB plans the 2-layer
loss in 211 operators and runs it in 0.12 s. So, on DataFusion, the demo
now runs each of the loss's CTEs on its own and registers its result as
an in-memory table, and writes the model as CTEs in each statement
rather than as a view, whose CTEs DataFusion writes out when it creates
it. With ddxdb 0.4.4, at batch 4, the 2-layer loss then takes 0.07 s on
DataFusion, and a 6-layer SGD step 2.4 s, against 0.30 s and 5.5 s on
DuckDB. A ddx training step runs 76 small queries, of
5.4 ms each, and about 4 ms of each is planning; with one partition
instead of twelve, and one optimizer pass, a step takes 0.40 s instead
of 0.66 s.

**JAX is the target.** JAX, compiled with `jax.jit` on the same CPU,
computes the GPT's loss in 0.15 ms and an SGD step in 0.37 ms, and
attention at 512 tokens in 0.92 ms forward and 2.4 ms with its
gradients: 300 to 400 times faster than DataFusion on attention, and up
to 1,000 times faster than DuckDB on its gradients, whose time is in
running their joins; and thousands of times faster on the GPT, whose time
is in planning.

**DataFusion and DuckDB store every score.** Causal attention's forward
pass at 512 tokens takes 0.27 s on DataFusion in the window spelling and
0.39 s in the `GROUP BY` spelling; DuckDB takes 0.47 s and 0.26 s. Both
grow with the square of the tokens. The gradients take 2.5 to 3.5 times
the forward pass: 0.84 s and 0.95 s on DataFusion at 512 tokens, against
1.1 s and 1.3 s with ddxdb 0.3.3 on DataFusion 54, and 2.8 s and 2.7 s
with ddxdb 0.3.2, before ddx differentiated the masked product as
`jax.grad` does.

**einfold finds FlashAttention's plan.** Measured on 2026-10-29 with
einfold 0.1.0rc2, ddxdb 0.4.6, DataFusion 55.0.0 and DuckDB 1.5.6. Both
spellings of attention, and the Nadaraya–Watson control, run as one of
einfold's fused pipelines. It computes 64 × 64 tiles of scores one at a
time and keeps a running softmax, and it never visits a pair of tiles the
causal mask rules out. That is FlashAttention's plan, found from this SQL
by a rule that mentions no attention. Demo 1 checks that each runs fused.

| Tokens | Forward pass, einfold | DataFusion | DuckDB | Gradients, einfold | DataFusion | DuckDB |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 13–15 ms | 0.28–0.41 s | 0.27–0.48 s | 0.61–0.68 s | 0.85–0.99 s | 1.7–2.2 s |
| 1,024 | 17–19 ms | 1.9–3.0 s | 0.91–1.9 s | 1.8–2.2 s | 4.0–4.4 s | 5.3–7.1 s |

(Each range covers the two spellings.)

- **Memory:** at 32,768 tokens the scores would take 4.3 GB. The forward
  pass takes 1.8 s, and the process's peak memory grows by 0.42 GB.
- **Determinism:** its results are the same bits on 1 and on 12 threads.
- **The gradient isn't fused yet.** einfold runs ddx's backward
  contractions as array algorithms, which about halves their time, but the
  backward pass still stores the scores.
- **Fixed costs:** einfold costs a few milliseconds per query. At the
  GPT demo's own size, a training step, dozens of small queries, takes
  0.85 s with einfold against 0.37 s without. At nanoGPT's CPU size (1
  layer, 4 heads, 128 channels, context 64, batch 12), its gradients
  take 2.7 s against 9.5 s on DataFusion and 16.4 s on DuckDB.

**ddx differentiates on DuckDB.** Measured with ddxdb 0.4.2. The
gradients of attention in the `GROUP BY` spelling match `jax.grad` to
2e-15 on DuckDB, and take 0.25 s, 0.78 s and 2.4 s at 64, 256 and 512
tokens, against 0.095 s, 0.31 s and 0.95 s on DataFusion. Nearly all of DuckDB's
time is in running ddx's program, not planning it: two contractions over
about 8 and 17 million joined rows take half of it. DuckDB orders those
joins worse when it reads a table through an Arrow scan, as
`xql.register` gives it, than when it reads a native table (0.64 s
against 0.20 s for one step), but copying Q, K and V into native tables
only brings the gradient to 1.7 s, and slows the forward pass (measured
on 2026-10-24, with ddx's development branch and DuckDB 1.5.6).

**The GPT trains on DuckDB.** Measured on 2026-10-27 with ddxdb 0.4.3,
the first to differentiate a view whose CTE computes a window on DuckDB,
as the softmax's weights are in the GPT and in attention's window
spelling. After each of four SGD steps and three AdamW steps, the GPT's
weights match JAX's to 2e-15 on DuckDB, and attention's window spelling
matches `jax.grad` to 2e-15, taking 0.68 s and 1.9 s at 256 and 512
tokens. DuckDB computes the GPT's loss in 0.062 s, against 2.7 s on
DataFusion in the same run, but an SGD step takes 0.87 s, against 0.41
s (1.15 to 1.3 s with AdamW). ddx runs its program's 77 steps on DuckDB
as `CREATE TEMP TABLE … AS` statements, which take 0.56 s of the 0.87 s:
DuckDB plans each one afresh, which takes about 0.3 s (an `EXPLAIN` of
each takes 0.31 s in all), and xarray-sql's adapter reads the parameter
Datasets in Python for about 0.19 s of the running. Most of the rest is ddx
planning on DuckDB, against an empty copy of the connection's catalog
that it keeps current.

**What ddx cannot differentiate yet.** A `GROUP BY` key that depends on
a differentiated column raises a typed error naming the construct, never
a wrong number. einfold's window spelling groups by the normalizer `z`;
the demos divide inside the sum instead, `SUM(w / z * v)`, which einfold
also lists.

**xarray-sql reads NaN as missing.** A NaN in a Dataset becomes SQL
`NULL`, which `SUM` skips, so Demo 1 checks masks with an infinity: a
plan that multiplies a masked pair's zero weight by it gives NaN where
SQL gives a number.

## Next

The open work, and how to measure it, is in
[#285](https://github.com/xqlsystems/xarray-sql/issues/285). Part B of
the NanoGPT demo is also still to come: the four tricks behind a record
of the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt),
a public race to train GPT-2 fastest, as a small diff to `02_nanogpt.py`.

[`_bench.py`](_bench.py) holds the shared command line, timing, checks and
results table.
