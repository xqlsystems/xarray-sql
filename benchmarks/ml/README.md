# Machine learning in SQL

These scripts write machine-learning models as SQL over tables that
[xarray-sql](../../README.md) registers from xarray Datasets, and time the
same SQL on several engines. Two related projects shape them:

- [ddx](https://github.com/xqlsystems/ddx) differentiates SQL queries: it
  turns the gradient of a query into more SQL, which trains the models.
- [einfold](https://github.com/xqlsystems/einfold) is a query optimizer
  that rewrites SQL plans into array algorithms. Two of these models
  are its [target demos](https://github.com/xqlsystems/einfold/blob/main/docs/demos.md),
  and this SQL is the baseline its plans must beat.

Each script checks its results against the same model in
[JAX](https://github.com/jax-ml/jax), Google's library for differentiable
array programs.

| # | Script | Model | What it measures |
|---|---|---|---|
| 01 | [`01_attention.py`](01_attention.py) | causal softmax attention, the core of a transformer, as a view (the model) in einfold's window and `GROUP BY` spellings, and a loss that reads it | the naive plan, which stores every score, for [FlashAttention](https://arxiv.org/abs/2205.14135)'s tiled plan to beat; gradients with respect to Q, K and V; masked pairs as absent rows; a control that isn't attention, the Nadaraya–Watson kernel average |
| 02 | [`02_nanogpt.py`](02_nanogpt.py) | [nanoGPT](https://github.com/karpathy/nanoGPT)'s GPT-2 model, at 1 layer and 4,416 parameters, as a view, and a cross-entropy loss that reads it, on Tiny Shakespeare | the loss on each host; training steps with SGD or, as nanoGPT trains, AdamW, every parameter table checked after every step |

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
`python/ddxdb`) and run the scripts with `uv run --no-sync`, or the
venv's `python`: a plain `uv run` or `uv sync` reinstalls the release.

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
| `duckdb` | DuckDB, through `xql.register` | ddx, with the `duckdb` extra of ddxdb |

A host that cannot differentiate still runs every forward query, and its
gradient rows say `no ddx backend`. A gradient that ddx, or the engine
it plans with, refuses is reported as `refused:` with the error, and a
refused training step ends that host's training run. To add a host, such as DataFusion with
einfold's optimizer or DuckDB with a GPU extension, write one class in
`_hosts.py` and add it to `HOSTS`.

## Findings

Measured on 2026-10-17 on an Intel Core i7-8700 (6 cores, 12 threads)
with DataFusion 54.0.0, DuckDB 1.5.5 and ddxdb 0.3.0, one repetition each.
The GPT and attention demos were timed again on 2026-10-21 with
DataFusion 55.0.0 and ddxdb 0.4.0, after xarray-sql began registering
small in-memory Datasets as DataFusion in-memory tables.

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
| GPT loss, 1 layer, batch 4 | 13,756 | 126 | 3.0 s | 0.057 s |

Setting up each lazy read of a Dataset takes xarray-sql about 1.5 ms, so
the GPT's 2,989 reads in one forward pass took 6.7 s of its 9.4 s. A small
in-memory Dataset is now an in-memory table, which takes a quarter of a
millisecond per read. A GPT training step, forward and backward through
ddx, takes 0.75 s on DataFusion with SGD and 0.85 s with AdamW, a quarter
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
loss in 211 operators and runs it in 0.12 s. DataFusion 56 plans a CTE
written `WITH … AS MATERIALIZED` once, which the GPT needs to grow past
one layer on DataFusion. A ddx training step runs 76 small queries, of
5.4 ms each, and about 4 ms of each is planning; with one partition
instead of twelve, and one optimizer pass, a step takes 0.40 s instead
of 0.66 s.

**JAX is the target.** JAX, compiled with `jax.jit` on the same CPU,
computes the GPT's loss in 0.15 ms and an SGD step in 0.37 ms, and
attention at 512 tokens in 0.92 ms forward and 2.4 ms with its
gradients: 300 to 400 times faster than either engine on attention,
whose time is in running its joins, and thousands of times on the GPT,
whose time is in planning.

**Attention stores every score.** Causal attention's forward pass at 512
tokens takes 0.28 s on DataFusion in the window spelling and 0.37 s in the
`GROUP BY` spelling; DuckDB takes 0.48 s and 0.27 s. Both grow with the
square of the tokens. The gradients take 2.5 to 3.5 times the forward
pass: 0.92 s and 0.93 s on DataFusion at 512 tokens, against 1.1 s and 1.3
s with ddxdb 0.3.3 on DataFusion 54, and 2.8 s and 2.7 s with ddxdb 0.3.2,
before ddx differentiated the masked product as `jax.grad` does.

**ddx differentiates on DuckDB.** Measured on 2026-10-24 with ddx's
development branch (after ddxdb 0.4.1) and DuckDB 1.5.6. The gradients
of attention in the `GROUP BY` spelling match `jax.grad` to 2e-15 on
DuckDB, and take 0.26 s, 0.78 s and 2.4 s at 64, 256 and 512 tokens,
against 0.10 s, 0.31 s and 0.94 s on DataFusion. Nearly all of DuckDB's
time is in running ddx's program, not planning it: two contractions over
about 8 and 17 million joined rows take half of it. DuckDB orders those
joins worse when it reads a table through an Arrow scan, as
`xql.register` gives it, than when it reads a native table (0.64 s
against 0.20 s for one step), but copying Q, K and V into native tables
only brings the gradient to 1.7 s, and slows the forward pass. With the
same ddx, a GPT training step on DataFusion takes 0.67 s after the
first.

**What ddx cannot differentiate yet.** Each of these raised a typed
error naming the construct, never a wrong number:

- A `GROUP BY` key that depends on a differentiated column. einfold's
  window spelling groups by the normalizer `z`; the demos divide inside
  the sum instead, `SUM(w / z * v)`, which einfold also lists.
- On DuckDB, two views whose CTEs are read more than once: one whose
  CTE computes a window, as the softmax does in attention's window
  spelling and in the GPT; and any such view, when the loss also takes a
  gradient with respect to a table the view doesn't read. ddx reads
  DuckDB's plans as [Substrait](https://substrait.io/), a portable format
  for query plans, and DuckDB's Substrait producer has no plan for a CTE
  it computes once for several reads. These rows say
  `refused: Not implemented Error: CTE`.

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
