# Machine learning in SQL

The optional `dense_cpu` and `dense_gpu` hosts retain compact arrays across
SQL and gradient stages using the local
[EinFold integration](../../accelerators/einfold_mlx/README.md).
`dense_gpu` explicitly permits float32 arithmetic and requires
`MLX_ENABLE_TF32=0`. The hosts use identical model SQL, ddx derivatives and
existing reference checks. See the
[dense execution measurements](../apple/dense-execution.md).

These scripts write machine-learning models as SQL over tables that
[xarray-sql](../../README.md) registers from xarray Datasets, and time the
same SQL on several engines. Two related projects shape them:

- [ddx](https://github.com/xqlsystems/ddx) differentiates SQL queries: it
  turns the gradient of a query into more SQL, which trains the models.
- [einfold](https://github.com/xqlsystems/einfold) is a query optimizer
  that rewrites SQL plans into array algorithms. Two of these models
  are its [target demos](https://github.com/xqlsystems/einfold/blob/main/docs/demos.md),
  and the demos run on DataFusion with einfold as one of their engines.

The first two scripts check their results against the same model in
[JAX](https://github.com/jax-ml/jax), Google's library for differentiable
array programs.

| # | Script | Model | What it measures |
|---|---|---|---|
| 01 | [`01_attention.py`](01_attention.py) | causal softmax attention, the core of a transformer, as a view (the model) in einfold's window and `GROUP BY` spellings, and a loss that reads it | the naive plan, which stores every score, against [FlashAttention](https://arxiv.org/abs/2205.14135)'s tiled plan, which einfold finds from the same SQL; gradients with respect to Q, K and V; masked pairs as absent rows; a control that isn't attention: the [Nadaraya–Watson estimator](https://en.wikipedia.org/wiki/Kernel_regression#Nadaraya%E2%80%93Watson_kernel_regression), an average of nearby points weighted by a Gaussian kernel, which einfold must fuse with the same rules |
| 02 | [`02_nanogpt.py`](02_nanogpt.py) | [nanoGPT](https://github.com/karpathy/nanoGPT)'s GPT-2 model, at 1 layer and 4,416 parameters, as a view, and a cross-entropy loss that reads it, on Tiny Shakespeare | the loss on each host; training steps with SGD or, as nanoGPT trains, AdamW, every parameter table checked after every step |
| 03 | [`03_shakespeare.py`](03_shakespeare.py) | nanoGPT's 6-layer, 384-channel, 6-head Shakespeare character model with context 256 and batch 12 | training from scratch with exact GELU, dropout, SQL AdamW, train/validation evaluation and resumable checkpoints; optional checks of every gradient, weight and optimizer moment against the original PyTorch model |

The SQL favors legibility over speed. Each model reads like its definition,
and the slow parts are left slow on purpose: they are what ddx, xarray-sql
and einfold need to make fast.

To add a demo, see [CONTRIBUTING.md](CONTRIBUTING.md). The work to make
these models fast is tracked in
[#285](https://github.com/xqlsystems/xarray-sql/issues/285), which lists
places to start.

Every result in the original demos is checked before it is timed into the table. Each demo
computes the same model in JAX from the same inputs and parameters, and a
mismatch fails the run.

## Running

```shell
uv run benchmarks/ml/01_attention.py
uv run benchmarks/ml/02_nanogpt.py --steps 5
```

### Shakespeare training matching nanoGPT's train.py

`03_shakespeare.py` is a training loop with checkpoints and held-out
validation. Its defaults match `train_shakespeare_char.py` with these
overrides: batch 12, context 256, layers/heads/channels 6/6/384, gradient
accumulation 1, `max_iters=2000`, `lr_decay_iters=2000`, warmup 100,
evaluation every 200 iterations over 20 batches per split. Inherited
settings also match: dropout 0.2, exact GELU, no biases, tied embeddings,
AdamW with learning rate 1e-3 to 1e-4, betas 0.9/0.99, epsilon 1e-8,
weight decay 0.1 excluding layer norm gains, and global gradient clipping
at 1. The learning rate uses `lr*(iteration+1)/(warmup_iters+1)` during
warmup, as in the local nanoGPT source. There are **10,745,088 parameters**
with the 65-character vocabulary; nanoGPT prints 10.65M when excluding
position embeddings.

Prepare the dataset once in a nanoGPT checkout with
`python data/shakespeare_char/prepare.py`. The SQL script reads its
`input.txt`, `train.bin` and `val.bin`, verifies the same sorted vocabulary
and 90/10 split, and samples shifted next-character targets without
crossing split boundaries. It initializes weights once per invocation
and keeps updating them across iterations.

Use this checkout's existing virtual environment with the optional
[EinFold build](../../accelerators/einfold_mlx/README.md) and
[ddx erf build](../../accelerators/ddx_math/README.md). Avoid `uv run`
for this script, since environment synchronization can replace those local
wheels. From the xarray-sql directory:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --backend dense_gpu \
  --data-dir /Users/cromwellian/nanoGPT/data/shakespeare_char \
  --out-dir out-shakespeare-sql
```

Use `--backend dense_cpu --cache-gib 32` for the measured compact float64 CPU path. GPU uses MLX
float32. Both differentiate the original SQL with ddx and execute the
model, global gradient clipping and AdamW arithmetic as SQL. They cache
query programs across updates. AdamW moments live in separate SQL tables
so the model's differentiated parameter column remains `w`. The maximum
used for softmax stabilization is spelled with `GROUP BY` and joins to
avoid repeated window-lowering traversal through the shared model graph.
Its maximum is held constant with `ddx_stop_gradient`: subtracting any
shared constant leaves softmax and cross entropy unchanged, so this
removes a cancelling derivative through the maximum without changing the
model's loss or parameter gradients.

All size and recipe options accept nanoGPT's underscore spelling and
hyphens, for example `--batch_size=12` and `--eval-iters=20`. The loop
matches `train.py`'s termination condition: `max_iters=2000` includes
iterations 0 through 2000, or 2001 optimizer updates. Add `--steps 5` to
run exactly five updates while retaining the 2000-iteration LR schedule.

Evaluation runs before the update at iteration 0, then every 200
iterations, with dropout disabled on 20 train and 20 validation batches.
`metrics.jsonl` records losses, learning rates and synchronized SQL
training times. Times include forward loss, backward gradients, clipping,
AdamW and result registration; sampling masks/batches, evaluation,
checkpoint I/O and optional reference checks are outside the timer.

`last.npz` contains all weights, optimizer moments, next iteration, both
RNG states, vocabulary and configuration. `best.npz` saves an improved
validation result after iteration 0. Resume with the same configuration:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --resume out-shakespeare-sql/last.npz
```

For an independent correctness check, install PyTorch in the execution
environment and add `--steps 5 --verify-nanogpt /Users/cromwellian/nanoGPT`.
This imports the original `model.py`, retains exact GELU and PyTorch
AdamW, and compares evaluation/training loss, every raw gradient, every
updated weight and both moments on every step. The oracle uses explicit
attention and the same externally sampled Bernoulli dropout masks;
reference checks do not replace SQL computation. GPU comparisons use
the existing 3e-5 absolute/relative tolerance and CPU comparisons 1e-8.

Random initialization and sampling use NumPy's RNG rather than PyTorch's,
so the same integer seed alone does not reproduce a separate stock
`train.py` trajectory. The oracle loads identical initial arrays and
masks to test the arithmetic. Checkpoints use this script's safe NPZ
format, rather than nanoGPT's PyTorch checkpoint format. This script
currently supports gradient accumulation 1 and training without biases.
The [measured comparison](../apple/shakespeare-training.md) records the
verified five-step CPU/GPU runs and validation results.

The first two scripts carry [inline `uv` metadata](https://docs.astral.sh/uv/guides/scripts/)
and runs against the xarray-sql in this checkout, with the `ml` extra. The
options are the same in each:

```shell
--hosts datafusion duckdb   # default: datafusion einfold duckdb
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

The optional Apple silicon comparison adds explicit hosts (the three
defaults above stay the same):

| Host | Execution |
|---|---|
| `datafusion_prepared` | Retain forward query plans, ddx programs, model views and parameter buffers |
| `cpu_optimized` | Prepared path with EinFold and the float64 CPU matrix backend, using its normal thresholds |
| `cpu_forced` | Same, with the backend threshold set to zero for a direct small-kernel comparison |
| `gpu_optimized` | Prepared path with [MLX](https://github.com/ml-explore/mlx), Apple's array framework, and measured automatic GPU dispatch thresholds |
| `gpu_forced` | Same, forcing eligible dense operations to GPU to measure dispatch overhead |
| `shared_einfold` | Shared forward CTEs and reusable ddx stages, with EinFold's CPU kernels |
| `shared_cpu` | Shared execution with the float64 CPU matrix backend |
| `shared_gpu` | Shared execution with explicitly enabled float32 MLX kernels |

These hosts need the [optional native bridge](../../accelerators/einfold_mlx/README.md).
GPU hosts explicitly permit float32 arithmetic and check JAX with
`rtol=atol=3e-5`; float64 hosts retain `1e-9`. Completed CPU/GPU kernel
counts are recorded, so an automatic-dispatch run that stays on CPU is
visible. Dataset loading and model-view creation times are also recorded
cumulatively on training rows. The loss/training times keep their original
scope and exclude JAX reference work and those loading/view times.

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/02_nanogpt.py \
  --steps 5 --hosts datafusion einfold duckdb datafusion_prepared \
  cpu_optimized cpu_forced gpu_optimized gpu_forced --csv comparison.csv
```

The [M5 Ultra five-step comparison](../apple/nanogpt-5-steps.md) reports
cold and warm costs separately, checks every loss and weight update, and
includes a forced GPU run with actual kernel counts.

The [shared-execution report](../apple/nanogpt-shared.md) adds larger models.
`--preset demo` uses one layer, 128 channels, four heads and context 64;
`--preset shakespeare` uses six layers, 384 channels, six heads and context
256, following nanoGPT's character-model dimensions. `--preset gpt2` uses
12 layers, 768 channels, 12 heads, context 1024 and vocabulary 50257.
Explicit dimension flags override presets. Inputs remain Shakespeare
character IDs: padding the vocabulary does not add GPT-2 tokenization,
pretrained weights, biases or dropout. `--forward-only` checks losses
without running gradients or updates.

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/apple/run_nanogpt_sizes.py \
  --preset shakespeare --steps 5 --batch 1 --learning-rate 0.01 \
  --hosts shared_cpu shared_gpu --prefix local-shakespeare
```

This runner starts hosts serially, checkpoints checked CSV rows and records
commands, exit status and sampled resident memory. A failed or bounded run
is not reported as a completed training benchmark.

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
loss in 211 operators and runs it in 0.12 s, and trains the 6-layer
model at 5.5 s per SGD step (ddxdb 0.4.4, batch 4). So on DataFusion the
GPT doesn't grow past one layer yet. The demo keeps the SQL as a reader
would write it, and the fix belongs in DataFusion, which plans a CTE again
at every read
([apache/datafusion#8777](https://github.com/apache/datafusion/issues/8777);
[apache/datafusion#26237](https://github.com/apache/datafusion/issues/26237)
asks it to decide when to compute a CTE once). Run separately, one
in-memory table per CTE, the same SQL takes 0.07 s for the 2-layer loss
and 2.4 s for a 6-layer SGD step on DataFusion, so that is what the fix
is worth. A ddx training step runs 76 small queries, of
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
  0.92 s with einfold against 0.41 s without. At nanoGPT's CPU size (1
  layer, 4 heads, 128 channels, context 64, batch 12), its gradients
  take 3.3 s against 9.6 s on DataFusion and 16.4 s on DuckDB.

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
