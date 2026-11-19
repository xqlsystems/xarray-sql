# Compact SQL execution on Apple M5 Ultra

November 1, 2026. [EinFold](https://github.com/xqlsystems/einfold) lowers
SQL numerical operations to arrays. [ddx](https://github.com/xqlsystems/ddx)
still differentiates the original SQL. The new optional executor retains
compact arrays across shared forward and backward stages, using the
[NumPy](https://numpy.org/) array library with float64 on CPU or
[MLX](https://github.com/ml-explore/mlx) float32 on GPU.
It does not substitute framework model code or MLX automatic differentiation.

See the [research and execution plan](../../accelerators/einfold_mlx/dense-execution-plan.md)
and [installation/API instructions](../../accelerators/einfold_mlx/README.md).
The implementation is local and experimental, packaged in
`einfold==0.2.0+xqlmlx.4`; it is not in the ordinary upstream wheel.

## Method

Machine: Apple M5 Ultra, 30 CPU cores, 64 GPU cores, 256 GiB RAM,
macOS 27.0.1. GPU game contention was off. Hosts ran serially. Every
reported training run passed **five steps**, with each loss and every
parameter update checked against the same float64
[JAX](https://docs.jax.dev/) array-framework reference outside
the API timer. CPU tolerances remain `rtol=atol=1e-9`; GPU tolerances
remain `3e-5`. GPU runs launched with `MLX_ENABLE_TF32=0`.

Warm time is the mean loss-plus-training API time at steps 1–4, excluding
step 0. The SQL interface makes separate loss and update calls, including
full [Apache Arrow](https://arrow.apache.org/) columnar-table output
materialization and synchronization. Dataset loading,
model definition, cold first-use time, and total API time are retained in
the summary. Reference computation, parameter validation and conversion
of returned rows into Datasets are outside these API timers. These are
four warm observations, not a confidence interval.

All SGD runs use the same model SQL, dimensions, tied embeddings, tanh
GELU, inputs and seeds as their controls. The tiny run uses learning rate
0.5; larger runs use 0.01. GPT-2-size runs use vocabulary 50257 and batch 1.
Inputs remain Shakespeare character IDs with a padded vocabulary, rather
than GPT-2 byte-pair tokenization. Context 16 has 123,563,520 parameters;
context 128 has 123,649,536; context 1024 has 124,337,664. All vocabulary
classes and declared parameters participate.

Sampled RSS is process memory measured once per second. It is not a hard
allocator limit or a complete accounting of Metal driver/unified memory.
The runner kills only its own benchmark process group when a configured
RSS/time bound is exceeded. Incomplete runs contribute no speedup.

## Measurements

<!-- RESULTS -->

All rows below are completed five-step training runs. Times are seconds.

| Parameters / context / batch | CPU before | CPU now | CPU speedup | GPU before | GPU now | GPU speedup |
|---|---:|---:|---:|---:|---:|---:|
| 4,416 / 16 / 4 | 0.193 | 0.052 | 3.7× | 0.219 | 0.078 | 2.8× |
| 213,504 / 64 / 1 | 0.431 | 0.061 | 7.1× | 0.482 | 0.083 | 5.8× |
| 10,745,088 / 256 / 1 | 29.257 | 1.406 | 20.8× | 29.937 | 0.589 | 50.8× |
| 123,563,520 / 16 / 1 | 164.034 | 4.037 | 40.6× | 163.646 | 2.434 | 67.2× |

The 123.6M/context-16 comparison is **40.6× faster on CPU and 67.2× on GPU** than the earlier shared SQL implementation. The new GPU path is 1.66× faster than the new CPU path on that case. CPU remains faster for the smallest models.

| Parameters / context | Host | Cache GiB | Cold first pair | Warm pair | Five API + loading/model | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|
| 4,416 / 16 | [dense_cpu](dense-measured-tiny-runs.json) | 8 | 0.128 | 0.052 | 0.348 | 0.53 |
| 4,416 / 16 | [dense_gpu](dense-measured-tiny-runs.json) | 8 | 0.356 | 0.078 | 0.679 | 0.57 |
| 213,504 / 64 | [dense_cpu](dense-measured-demo-runs.json) | 8 | 0.139 | 0.061 | 0.396 | 0.60 |
| 213,504 / 64 | [dense_gpu](dense-measured-demo-runs.json) | 8 | 0.188 | 0.083 | 0.534 | 0.58 |
| 10,745,088 / 256 | [dense_cpu](dense-measured-shakespeare-runs.json) | 8 | 1.870 | 1.406 | 7.548 | 7.97 |
| 10,745,088 / 256 | [dense_gpu](dense-measured-shakespeare-runs.json) | 8 | 0.993 | 0.589 | 3.401 | 2.78 |
| 123,563,520 / 16 | [dense_cpu](dense-measured-gpt2-c16-runs.json) | 8 | 5.193 | 4.037 | 21.627 | 22.90 |
| 123,563,520 / 16 | [dense_gpu](dense-measured-gpt2-c16-runs.json) | 8 | 3.750 | 2.434 | 13.774 | 17.91 |
| 123,649,536 / 128 | [dense_cpu](dense-measured-gpt2-c128-runs.json) | 32 | 7.194 | 5.787 | 30.638 | 35.78 |
| 123,649,536 / 128 | [dense_gpu](dense-measured-gpt2-c128-runs.json) | 32 | 3.882 | 2.540 | 14.330 | 18.46 |
| 124,337,664 / 1024 | [dense_gpu](dense-measured-gpt2-c1024-runs.json) | 32 | 7.496 | 5.575 | 30.085 | 28.72 |

Fresh tiny-model DuckDB time is 0.274 s. Compact CPU is 5.29× faster, and compact GPU is 3.53× faster. This is a bounded tiny-model DuckDB comparison, not a claim about full-size DuckDB training.

The compatible API includes cold ddx preparation and initial MLX compilation in the first pair. Loading/model time includes cumulative input/state writes over the run. Full-context CPU trials and strict dense AdamW refusals are excluded from these tables. The earlier compact cache study improved context-1024 GPU time from 14.242 s at 8 GiB to 5.835 s at 32 GiB; the final 32 GiB run measures 5.575 s. Cache-study records are in [the 8 GiB trial](dense-masks-gpt2-c1024-runs.json) and [32 GiB trial](dense-cache32-gpt2-c1024-runs.json).

[All final aggregate records](dense-summary.json). The fourteen completed runs contain 140 checked loss/update checkpoints.
<!-- END RESULTS -->

The [previous shared execution report](nanogpt-shared.md) supplies the
large-model baselines. Tiny DuckDB/shared CPU/shared GPU controls were
rerun during this work, with their commands in
[the control manifest](dense-final-tiny-runs.json). Development runs with
older intermediate implementations are retained separately and are not
used as final measurements.

## What changed

- Coordinate axes are represented once. Numerical values remain compact
  arrays between SQL stages. Scalar key columns expand at public output
  boundaries.
- Actual [Apache DataFusion](https://datafusion.apache.org/) SQL-engine
  logical joins, projections, CASE expressions,
  contractions, full-partition windows, gathers and grouped reductions
  lower to an EinFold array program. Shared CTEs and ddx named reads bind
  to current array slots. Numerical stages do not build hash joins or
  reconstruct intermediate Arrow coordinates.
- Embedding gradients use segmented/scatter reductions. UNION remains a
  bag union until a following grouped SUM justifies accumulating its
  contributions. Domain left joins still fill missing gradients correctly.
- Exact domains keep row-presence and NULL masks compact. Constant
  coordinate predicates slice axes before numerical work, including
  predicates within conjunctions. Coordinate ordering is established
  before row expansion when its exact domains prove the requested order.
- Structural logical programs and compiled pointwise templates are reused
  across steps. Every compiled MLX function receives its numerical leaves
  explicitly; it does not capture previous parameter values. Input buffers
  are imported afresh per call, including external in-place writes.
- Computed arrays have a bounded LRU cache; sources remain pinned for the
  call. Values may be recomputed after eviction. Graphs are released at
  call completion. Default cache capacity is 8 GiB; the full-context GPU
  experiment also measures 32 GiB. This bounds retained computed values,
  not all allocations or process memory.

Private schema-only providers let ddx serialize named read schemas. They
are not executed as numerical scans. Runtime guards and existence checks
evaluate actual data. Unsupported execution restarts the call through the
original shared SQL executor before any parameter update is committed.
Benchmarks set `fallback=False`, so a refusal cannot silently become an
accelerated result.

## Profiling and device crossings

The original instrumented backward profile identified these large stages:

| Stage | Time | Rows | Arrow bytes |
|---|---:|---:|---:|
| Combined token embedding gradient | 18.58 s | 38,597,376 | 926,337,024 |
| One token embedding gradient contribution | 16.81 s | 38,597,376 | 926,337,024 |
| One stacked projection gradient | 6.32 s | 28,311,552 | 905,969,664 |

One cotangent stage took 3.8 seconds to produce just 12,288 values. Its SQL
contained many independently expanded gradient branches. DataFusion's
outer FFI node count hid those internal operators; it was not a useful
cross-backend operator count. The full diagnostic is in
[the stage JSON](dense-before-stages-summary.json) and
[log](dense-before-stages.log).

An intermediate compact CPU profile then exposed about 3.3 seconds of
sorting fully expanded output coordinates. Ordering compact axes removed
that work. It also exposed repeated full-domain gradient validity arrays;
domain-aware masks now avoid those allocations where exact facts permit.
[That diagnostic profile](dense-cpu-profile.log) is not a final benchmark.

<!-- PROFILE -->
The final diagnostic warm pair takes 5.19 seconds under CPU profiling and
3.62 seconds under GPU profiling. These include profiler overhead and are
not substituted for the uninstrumented benchmark times.

The GPU profile records 0.278 seconds in input conversion, 0.257 seconds
waiting for pending GPU computation, and 0.00044 seconds creating NumPy
views after that wait. The rest includes Python graph construction,
domain/contribution processing, coordinate expansion, Arrow conversion and
profiler overhead. Every training call returns **3,871,488,000 Arrow bytes**
at context 16, even though its compact GPU parameter values occupy roughly
494 MB. The final CPU profile attributes 1.80 seconds cumulatively to dense
stage execution and 3.08 seconds to output collection, including deferred
numerical work; these nested timings should not be added to other entries.

[CPU profile](dense-final-cpu-profile.txt),
[CPU phase record](dense-final-cpu-profile.json),
[GPU profile](dense-final-gpu-profile.txt),
[GPU phase record](dense-final-gpu-profile.json).
<!-- END PROFILE -->

The raw counters distinguish source imports, numerical operations and
readback requests. Readback byte counts are logical array bytes made
available to NumPy; they are not measurements of PCIe traffic. Apple
silicon uses unified memory, and float64-to-float32 imports still copy and
convert. The returned Arrow values keep SQL dtypes, so GPU float32 values
also convert back to Arrow float64.

Across the five context-16 steps, counters record 140 reusable fused MLX
templates, 20,020 contractions/reductions and 170 readback requests. Source
imports total 9,885,144,280 bytes: approximately one import of each parameter
buffer per loss/update call, plus scalar constants. Requested readback
arrays total 2,664,259,240 bytes, including validity/presence data. The
context-1024 run also uses 170 readback requests, while larger arrays and
cache eviction increase its numerical work. These counts verify actual
GPU dispatch, not just an explained GPU node.

The 12-layer runs retain 707 logical programs after the first training
step. Their count stays fixed over five steps. The earlier shared backend
built 7,148 physical plans over five steps. Those are different categories
of plan, not equivalent operator counts: the new numerical path uses
logical programs without per-stage physical planning.

## Native framework comparison

<!-- NATIVE -->
The native GPU control uses
[PyTorch](https://pytorch.org/), a tensor framework, through its Metal
Performance Shaders (MPS) backend and the
[upstream nanoGPT model](https://github.com/karpathy/nanoGPT).
For **124,337,664 parameters, context 1024, batch 1 and five training steps**:

| Implementation | Warm step | Precision |
|---|---:|---|
| Compact SQL + MLX | 5.575 s | float32 arithmetic, Arrow outputs |
| JAX CPU, compiled | 2.445 s | float64, resident arrays |
| Upstream nanoGPT / PyTorch MPS | 0.099 s | float32, resident arrays |

The SQL API is approximately **2.28× slower than compiled JAX CPU** and
**56.6× slower than PyTorch MPS** in these measurements. Its substantial
improvement over the earlier SQL backend does not eliminate the remaining
gap to native array execution.

[JAX result](dense-native-gpt2-c1024-jax_jit.json),
[JAX log](dense-native-gpt2-c1024-jax_jit.log),
[nanoGPT/MPS result](dense-native-gpt2-c1024-torch_mps.json),
[nanoGPT/MPS log](dense-native-gpt2-c1024-torch_mps.log).
<!-- END NATIVE -->

These are API comparisons with important scope differences. SQL performs
a loss call and an update call and returns complete Arrow parameter tables.
The native frameworks keep array state resident and synchronize numerical
work inside their timer, with validation/readback outside. CPU reference
arithmetic is float64; GPU arithmetic is explicitly float32. The framework
harness uses the actual upstream nanoGPT `GPT` class, matching the SQL
math with tanh GELU and SGD; it is not the repository's untouched CUDA
AdamW speed benchmark. Source revisions, actual allocated parameter counts
and all per-step checks are saved in the JSON records. See also the
[original comparison and harness documentation](nanogpt-frameworks.md).

## Correctness and limitations

<!-- VALIDATION -->
The frozen implementation passes **695 tests**, with 277 skipped and 102
network integration tests deselected. The 83 existing warnings are retained
in [the full test log](dense-full-correctness.log). Focused CPU/Metal SQL
and backward tests pass 73 cases. Ruff and focused mypy checks pass;
`git diff --check` and clean application of the pinned native patch pass.
No Rust numerical kernel changed in this phase; the prior native kernel
validation remains in the shared-backend report.

[Environment and source hashes](dense-environment.json) identify the exact
checkout, dependencies and hardware. The installed EinFold dense module is
byte-identical to the adjacent source included by the wheel builder.
[Build log](dense-build.log), [focused test log](dense-final-contract.log),
[mypy log](dense-mypy.log).
<!-- END VALIDATION -->

Public contract checks cover live values and snapshots, changed coordinate
domains, non-positional/permuted labels, matrix gradients through normalized
operands, nullable and null-safe joins, bag unions, all-NULL/empty groups,
integer accumulators and averages, filtered-out rows, function overrides,
grad/VJP/JVP, stop-gradient and typed refusals. The existing suite continues
to check xarray dimensions, coordinates and attributes. Source arrays are
not mutated by numerical execution. No tolerance was relaxed.

Full-context CPU execution remains sensitive to recomputation and cache
capacity. An early unbounded trial crossed the 64 GiB RSS guard; bounded
trials stayed within the process bound but did not complete five steps
within their chosen time limits. The completed GPU context-1024 run is
training; the earlier context-1024 shared-backend comparison was only
forward execution. There is no successful old full-context training run
from which to calculate a training speedup.
The final CPU trial used a 32 GiB computed cache and a 90-second process
time bound; it did not finish its first update. An earlier five-minute
trial completed one checked update in 171 seconds but not five steps.
Both are retained as timeouts, not averaged into a training result.

The strict dense AdamW demo currently declines an aggregate with multiple
dependent or nullable keys. It is excluded from performance claims. The
public executor's default SQL fallback remains available and is tested
through a partial backward refusal and subsequent input updates. Existing
float32 AdamW sensitivity is not concealed by weakening its tolerance or
changing its update. For exact arithmetic, use the default float64 CPU
mode or the original SQL executor.

Remaining work is an explicit API that owns GPU training state across
calls, compilation of larger portions of the whole ddx program, and better
liveness/sparse handling of lifted backward contributions. They address
the remaining native-framework gap. The compatible API still imports and
exports full state, constructs numerical graphs in Python, and may
recompute after cache eviction. It is not native array-training parity.

## Reproduce

```sh
.venv/bin/python accelerators/einfold_mlx/build.py
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python \
  benchmarks/apple/run_nanogpt_sizes.py --preset gpt2 --context 16 \
  --steps 5 --batch 1 --hosts dense_cpu dense_gpu --prefix local-gpt2-16 \
  --rss-limit-gib 64 --max-minutes 5 --keep-going
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python \
  benchmarks/apple/run_nanogpt_sizes.py --preset gpt2 --context 1024 \
  --steps 5 --batch 1 --hosts dense_gpu --dense-cache-gib 32 \
  --prefix local-gpt2-1024 --rss-limit-gib 64 --max-minutes 5
```

Exact commands, exits, checks and sampled memory are in each linked run
manifest. `summarize_dense.py` regenerates the aggregate JSON from those
manifests and CSVs and excludes failed runs. `profile_dense_execution.py`
records a separate diagnostic warm pair, including import conversion,
device wait and NumPy view creation on GPU. Profiles are not used as
performance measurements.
