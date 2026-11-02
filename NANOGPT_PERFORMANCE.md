# NanoGPT SQL execution and training performance

Measured November 1, 2026, on Apple M5 Ultra (30 CPU cores, 64 GPU cores,
256 GiB unified memory), macOS 27.0.1. GPU game contention was off for the
final runs. This report covers the implementation and the measurements;
ratios from different workloads must not be multiplied together.

[nanoGPT](https://github.com/karpathy/nanoGPT) is a small GPT training
implementation. This work keeps the model and optimizer expressed in SQL:
[DataFusion](https://datafusion.apache.org/) plans the SQL,
[ddx](https://github.com/xqlsystems/ddx) differentiates it,
[EinFold](https://github.com/xqlsystems/einfold) lowers numerical operations
to compact array programs, and [MLX](https://github.com/ml-explore/mlx)
executes float32 arithmetic on Apple's GPU. CPU execution uses NumPy
float64. There is no replacement model or handwritten backward pass.

## Setup, plan reuse and CPU overhead

| Change | Reason and observed improvement |
|---|---|
| Count Dataset chunk partitions arithmetically | Avoid constructing a Cartesian grid merely to count it. Registration of a 4,096×16 Dataset with one-cell chunks fell from 31.372 to 0.405 ms, **77.44×**. This applies to that fine-chunk case, not all registration. |
| Register ordinary in-memory Arrow batches directly | Remove the temporary catalog table used during registration. Included in the preceding combined measurement; no isolated ratio. |
| Retain prepared input buffers and update their contents | Repeated update-and-sum of 1,048,576 values fell from 2.769 to 1.027 ms, **2.70×**, versus re-registering every call. |
| Cache forward logical plans and ddx derivative programs | Avoid parsing and planning identical SQL on each step. A repeated-CTE DataFusion product fell from 1.709 to 1.307 ms, **1.31×**. Prepared grad/VJP/JVP calls reuse derivative programs and invalidate them when the catalog/settings change. |
| Preserve Context metadata when installing extensions | Keep Dataset reconstruction and prepared invalidation correct when adding execution extensions. Correctness improvement; no isolated speed claim. |
| Expose `Context(threads=...)` and benchmark `--threads` | Target-partition sweep from 1 to 30 reduced DataFusion attention from 6,797 to 795 ms, **8.55×**, and EinFold attention from 25.1 to 9.3 ms, **2.70×**. The default already used the 30 physical cores. |

An independent 30→60 partition comparison gave DataFusion 790.149→747.087 ms
(1.06×), but EinFold 8.737→9.076 ms (3.9% slower). Higher partition counts
are explicit workload choices, not a universal improvement or a promise of
that many OS threads. BLAS has separate thread controls.

Preparing the tiny original training example improved warm DataFusion time
from 1,227.4 to 1,103.2 ms (1.11×), but its five-step total increased from
6.504 to 7.036 s because of setup/first-use costs. Warm performance alone
does not imply faster startup. See the [CPU/kernel report](benchmarks/apple/README.md)
and [five-step report](benchmarks/apple/nanogpt-5-steps.md).

## Shared CTE execution and kernel dispatch

The explicit shared executor evaluates each reachable common table
expression (CTE) once per invocation, retains reusable stage programs, and
gives ddx the original view definitions. Fresh inputs and physical join
state prevent reuse of old numerical values. Native SQL parsing and
verified ordered Arrow providers remove Python parsing and unnecessary
sorts. Function overrides invalidate permission to use equivalent array
primitives.

The session-local EinFold kernel hook dispatches validated contractions
and fused causal attention to NumPy or MLX. SQL checks of keys, duplicates,
NULLs and row domains precede dispatch; unsupported cases keep their SQL
execution. GPU precision is explicitly float32. Automatic thresholds keep
tiny work on CPU.

| Five-step original NanoGPT workload, warm loss + update | Before sharing | After sharing | Improvement |
|---|---:|---:|---:|
| Tiny 4,416 parameters, CPU kernel host | 314 ms | 206 ms | 1.53× |
| Tiny 4,416 parameters, GPU kernel host | 371 ms | 224 ms | 1.66× |
| 213,504 parameters, CPU | 551 ms | 431 ms | 1.28× |
| 213,504 parameters, GPU | 601 ms | 482 ms | 1.25× |

The tiny shared GPU run used 160 matrix and 5 attention kernels rather
than 1,015 and 200. At this stage, shared CPU took 206 ms versus DuckDB
306 ms and plain DataFusion 1,260 ms. These are tiny-model controls.
Shared GPU remained about as slow as shared CPU for the 123.6M model:
163.65 versus 164.03 s per update pair. GPU dispatch alone left the
backward SQL/Arrow bottlenecks intact.

Isolated kernel results include 256×512×256 CPU matmul 8.698→4.422 ms
(1.97×), and 1024³ original CPU→GPU matmul 64.131→35.174 ms (1.82×).
The new NumPy CPU implementation took 33.745 ms on the latter, slightly
faster than GPU. Attention at 16,384 tokens/context 64 improved
95.044→71.292 ms (1.33×). Tiny forced-GPU training regressed; it is not
presented as an acceleration. See [shared execution measurements](benchmarks/apple/nanogpt-shared.md).

## Compact execution of forward and backward SQL

Profiling identified coordinate expansion, sorting, Python/Arrow
registration, repeated backward stages, and device readbacks. In one
earlier embedding-gradient stage, materializing about 926 MB of scalar
rows took 18.58 s. A coordinate-expansion/sort profile consumed about
3.3 s. These profiles diagnose bottlenecks, not isolated before/after
speedups.

`DenseExecutor` retains each axis domain once and stores numeric values
separately. Joins lower to gathers or contractions; window reductions
broadcast; UNION branches retain bag semantics and accumulate gradients.
Row presence and SQL NULL validity are distinct masks. Ordering compact
axes before Arrow expansion avoids expanded-coordinate sorting when the
domains prove that order. A bounded computed-value LRU limits retained
intermediates. Sources are imported afresh for ordinary query calls;
public results remain independent Arrow snapshots.

MLX provides matmul, reductions, gather/scatter and compiled elementwise
expressions with explicit numeric arguments. Integer/domain guards refuse
unsupported behavior. Unsupported SQL falls back before caller-visible
mutation; strict benchmarks disable fallback to expose omissions.

| Parameters / context / batch | Shared CPU → compact CPU | CPU improvement | Shared GPU → compact GPU | GPU improvement |
|---|---:|---:|---:|---:|
| 4,416 / 16 / 4 | 193 → 52 ms | 3.7× | 219 → 78 ms | 2.8× |
| 213,504 / 64 / 1 | 431 → 61 ms | 7.1× | 482 → 83 ms | 5.8× |
| 10,745,088 / 256 / 1 | 29.257 → 1.406 s | 20.8× | 29.937 → 0.589 s | 50.8× |
| 123,563,520 / 16 / 1 | 164.034 → 4.037 s | 40.6× | 163.646 → 2.434 s | 67.2× |

All rows are **training from scratch**, five sequential loss/backward/SGD
updates. Warm means exclude the first update pair. Every loss and weight
update was checked against a float64 JAX reference outside the timer.
These original examples use tanh GELU, SGD, Shakespeare character IDs
padded to a vocabulary of 50,257 for GPT-2 sizes, and no dropout. They are
not the Shakespeare AdamW recipe below or a GPT-2 language-model quality
benchmark. The shared tiny baseline was rerun, so its value differs from
the earlier sharing experiment.

The 123.6M/context-16 compact GPU first pair took 3.750 s; warm pairs
took 2.434 s, with peak sampled RSS 17.91 GiB versus earlier shared GPU
40.0 GiB. CPU first/warm pairs took 5.193/4.037 s, with RSS 22.90 GiB
versus 38.7 GiB. At context 1,024, the 124,337,664-parameter GPU model
completed five training updates: first pair 7.496 s, warm 5.575 s, peak
RSS 28.72 GiB. CPU full-context trials did not complete and give no speedup.

An 8→32 GiB computed cache experiment reduced full-context GPU warm time
14.242→5.835 s (2.44×); a subsequent final run measured 5.575 s. A cache
budget is not process RSS. The tiny compact CPU/GPU controls took
52/78 ms versus DuckDB 274 ms; there is no matching full-size DuckDB
training result. See [compact execution evidence](benchmarks/apple/dense-execution.md)
and the [framework comparison](benchmarks/apple/nanogpt-frameworks.md).

## Complete GPU-resident Shakespeare training

`benchmarks/ml/03_shakespeare.py` matches the character-training recipe
from nanoGPT: batch 12, context 256, 6 layers/heads, embedding width 384,
65-character vocabulary, **10,745,088 parameters**, exact GELU, dropout
0.2, tied embeddings, no biases, global clipping at 1, and SQL AdamW
(betas 0.9/0.99, epsilon 1e-8, weight decay 0.1 excluding layer norms).
Learning rate is 1e-3→1e-4 with warmup 100 and decay through iteration
2,000; evaluation every 200 uses 20 batches each for train/validation.
The local native warmup formula is `lr * (iteration + 1)/(warmup + 1)`.

The recipe uses grouped MAX/join softmax normalization. Its common shift
is marked `ddx_stop_gradient`: shifting all logits by the same scalar
cancels analytically in softmax, avoiding an unnecessary max/backward
window subgraph. Optimizer moments occupy separate SQL tables with complete
coordinate domains. These choices preserve the checked gradients; their
individual timing contributions were not isolated. Checkpoints store
weights, moments, next iteration, RNG state and recipe metadata in NPZ,
and resume tests check the subsequent updates.

`TrainingSession` owns private rectangular, finite, nonnullable state
arrays, including weights and AdamW moments. A prepared step evaluates
ddx gradients, clipping and all SQL updates in one program. The new ddx
`Prepared.values` metadata identifies the primal loss already computed
by differentiation, eliminating the extra forward query. State remains
on GPU between steps and is published atomically after validation.
Failed updates retain the old state and dropout counter.

Whole-step `mx.compile` accepts weights, moments, batch values and optimizer
scalars as explicit arguments. Coordinate layouts, schemas, views and
session settings guard reuse. Bounded token domains permit GPU index joins
and embedding-gradient scatter; actual bound values are checked before
upload. Structural uniqueness checks are retained or proven from exact
coordinate/count bounds; unsupported dynamic host decisions decline
compilation. GPU dropout uses explicit seed/step keys and resumes from
checkpointed RNG state. Numerical host conversion during tracing is refused.

| Same 10.7M-character model, batch 12 | First step | Warm mean | Warm sample |
|---|---:|---:|---|
| Arrow compact CPU, float64, 32 GiB cache | 10.362 s | 9.747 s | Updates 2–5; all checked |
| Arrow compact MLX GPU, float32 | 2.523 s | 2.024 s | Updates 2–5; all checked |
| Resident GPU, uncompiled, CPU masks | 1.840 s | 1.356 s | Updates 2–5; all checked |
| Compiled resident GPU, CPU masks | 5.618 s | 120.7 ms | Updates 2–5; all checked |
| Compiled resident GPU, GPU dropout | 1.380 s | 145.9 ms | Updates 2–5; all checked |
| Compiled resident GPU, GPU dropout | 1.468 s | **101.3 ms** | Updates 2–100; no per-step oracle |
| Native nanoGPT / PyTorch MPS, float32, compile=False | See raw record | **27.0 ms** | Updates 2–100 |

The delivered 100-step path is **19.98× faster** than the earlier Arrow
GPU path and **3.76× slower** than native PyTorch MPS on this recipe.
The fully checked five-step run improved **13.87×** over the Arrow path.
The 100-step comparison uses a longer warm window than the five-step
baseline and includes GPU dropout inside the step. These are observations
from individual serial runs, not confidence intervals or isolated gains
attributable to each component. CPU-mask and GPU-dropout trajectories differ.

First-step times include preparation/compilation within that API call,
but exclude process startup and the preceding 40 evaluation calls. Shader
caches may be warm from earlier trials; the 1.380–5.618 s variation must
not be described as a cold-start improvement.

Timed SQL steps include the complete forward/backward/clipping/AdamW
update, state validation/publication and reported loss scalar. Dataset
loading, token sampling/upload, evaluation, correctness exports and
checkpoint I/O are outside the timer. Native synchronized timing includes
next-batch prefetch. Native uses its own PyTorch RNG trajectory; recipe
equivalence is separate from the identical-input oracle checks.

All 100 timed resident steps exported **zero numerical arrays** and read
back **zero array bytes**; each read one validation flag and one loss
scalar. There was one whole-step compilation. Batch input uploads occur
outside this timer. Scalar synchronization, CPU planning/setup and final
checkpoint exports remain; this is not a claim of zero host interaction.
Byte counters describe logical array transfers, not measured physical
bus traffic on Apple's unified-memory hardware.

Process RSS was 2,639,101,952 bytes after step 1 and 2,644,819,968 after
step 100, a 5.7 MB increase. Last-step MLX active/cache/peak counters were
532,354,042 / 5,081,660,545 / 4,428,988,178 bytes. These counters overlap
with unified-memory/RSS accounting and must not be added together. The
100-step observation supports stable retention over that run, not an
unbounded-memory guarantee. Earlier Arrow runs did not record comparable
RSS, so no exact memory-reduction ratio is claimed for this recipe.

## Correctness, reproduction and limitations

The final xarray-sql non-integration suite passed **714 tests**, with
277 optional skips and 102 integration tests deselected. The suite covers
SQL values/coordinates, NULL/duplicate behavior, live inputs, cache
invalidation, numerical precision, state ownership, failed-step atomicity,
changing compiled token/LR inputs, snapshots and checkpoint/resume.

This was repeated against wheels rebuilt from the persistent ddx/EinFold
checkouts, then again after integrating upstream's newer EinFold runtime:
[final publication test log](benchmarks/apple/resident-final-merged-correctness.log).
ddx passed its Rust workspace suite (535 passed, 10 ignored). EinFold's
Rust suite had 72 passes, 4 ignored and one NaN-sign behavior assertion
failure, reproduced unchanged on its clean pinned base on this macOS host.
Both passed Clippy, formatting and warning-free Rust docs; EinFold also
passed REUSE license checks. The [final combined Python binding suite](benchmarks/apple/publication-upstream-python.log)
passed 128 tests with 6 optional skips.

The five-step resident recipe checked every loss, raw gradient, weight and
both optimizer moments against original nanoGPT `model.py` and PyTorch
AdamW, using identical arrays, batches and explicit dropout masks outside
the timer (float32 tolerance 3e-5). The float64 Arrow recipe used 1e-8.
The 100-step performance run intentionally omitted per-step oracle exports;
all losses were finite. Neither run establishes convergence or general SQL
support for every compiled program.

Build the pinned optional dependencies using the
[EinFold instructions](accelerators/einfold_mlx/README.md) and
[ddx instructions](accelerators/ddx_math/README.md), then from this checkout:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --compile-training --gpu-dropout --steps 5 \
  --verify-nanogpt /path/to/nanoGPT --out-dir out-shakespeare-verified

MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --compile-training --gpu-dropout --steps 100 --profile-memory \
  --out-dir out-shakespeare-resident

MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/pytest -q -m 'not integration'
```

Use `--data-dir /path/to/nanoGPT/data/shakespeare_char` if needed, after
running nanoGPT's data preparation. Omit `--steps` to follow the full native
iteration-0-through-2000 recipe (2,001 updates). Each ordinary launch
initializes once from scratch; `--resume` loads a checkpoint.

The [ML benchmark README](benchmarks/ml/README.md) documents model-size
flags, all execution modes, evaluation and resume. Native comparison is
reproducible with [compare_shakespeare_mps.py](benchmarks/apple/compare_shakespeare_mps.py).
Recorded results are in [resident-summary.json](benchmarks/apple/resident-summary.json),
the [five-step checked metrics](benchmarks/apple/shakespeare-resident-verified-gpu/metrics.jsonl),
the [100-step metrics](benchmarks/apple/shakespeare-resident-steady-gpu/metrics.jsonl),
and the [native 100-step record](benchmarks/apple/shakespeare-resident-torch-mps-100.json).
Run `python benchmarks/apple/summarize_resident.py` to recompute ratios.

The pinned local integration builds used ddxdb `0.4.6+xqlmath.3`, EinFold
`0.2.0+xqlmlx.6`, DataFusion Python 55.0.0, MLX 0.32.4, NumPy 2.5.4,
PyArrow 26.0.0, Python 3.12.14 and PyTorch 2.15.0. Upstream contributions
retain normal package versions; local suffixes identify benchmark builds.
The ddx and EinFold repositories contain their own `OPTIMIZATION.md` reports.

## Review and dependency order

| Repository / change | PR | Depends on |
|---|---|---|
| ddx: exact error-function derivatives | [256](https://github.com/xqlsystems/ddx/pull/256) | Independent |
| ddx: prepared primal-output metadata | [257](https://github.com/xqlsystems/ddx/pull/257) | ddx 256 |
| EinFold: session-local numerical backends | [156](https://github.com/xqlsystems/einfold/pull/156) | Independent |
| EinFold: compact logical-plan arrays | [157](https://github.com/xqlsystems/einfold/pull/157) | EinFold 156 |
| xarray-sql: setup and plan reuse | [301](https://github.com/xqlsystems/xarray-sql/pull/301) | Independent |
| xarray-sql: shared stages and numerical kernels | [302](https://github.com/xqlsystems/xarray-sql/pull/302) | xarray-sql 301, EinFold 156 |
| xarray-sql: compact array integration | [303](https://github.com/xqlsystems/xarray-sql/pull/303) | xarray-sql 302, EinFold 157 |
| xarray-sql: resident SQL training | [305](https://github.com/xqlsystems/xarray-sql/pull/305) | xarray-sql 303, ddx 257 |
| xarray-sql: reports, harnesses and evidence | [304](https://github.com/xqlsystems/xarray-sql/pull/304) | Independent documentation branch; describes the complete stack |

Branches are published through the `cromwellian` forks because that account
cannot push upstream branches. GitHub requires those PRs to target an
upstream branch, so dependent PR default diffs include their predecessors
until those merge. Each description links the incremental comparison for
its own change. No PR is merged by this work. The reports retain benchmark
provenance separately from later upstream runtime integration and packaging
changes; see [publication sources](benchmarks/apple/publication-sources.json).
