# Apple M5 Ultra: measured SQL tensor acceleration

Implemented directly in the local xarray-sql checkout. The optional
[EinFold/MLX bridge](../../accelerators/einfold_mlx/README.md) connects
[EinFold](https://github.com/xqlsystems/einfold), a tensor optimizer for SQL,
to [MLX](https://github.com/ml-explore/mlx), Apple's array framework.
[ddx](https://github.com/xqlsystems/ddx) continues to differentiate SQL;
MLX evaluates arithmetic inside the physical plan, including arithmetic
in ddx's forward and backward programs.

## Measurement conditions

Mac Studio, Apple M5 Ultra: 30 CPU cores, 64 GPU cores, 256 GiB memory;
macOS 27.0.1, Python 3.12.14. Python packages: DataFusion 55.0.0,
EinFold 0.2.0+xqlmlx.1, ddxdb 0.4.6, MLX 0.32.4, NumPy 2.5.4,
PyArrow 26.0.0. The EinFold native library uses DataFusion 55.2.
`MLX_ENABLE_TF32=0` throughout final GPU runs.

Each number is the median of **seven measured executions after one warmup**.
Registration and NumPy reference construction are excluded. SQL planning,
execution, native/Python copies, sorting and Arrow materialization are
included. GPU results synchronize before their Arrow buffers are returned.
The same SQL and seeded inputs run on each host within a run. No compilation
or separate benchmark process ran concurrently with the final measurements.
Raw files contain every sample, maximum absolute error, environment and
completed-kernel counts. Variation between repeated runs was visible at the
millisecond level; small differences should not be treated as robust gains.

The EinFold baseline uses the **same pinned native library with no backend
callback installed**. Its existing tensor detection is responsible for most
of the speedup over plain DataFusion. Incremental kernel speedups are reported
separately below. No results from another chip or published benchmark are
presented as local measurements.

## Default dispatch: end-to-end milliseconds

`MLX(precision="float32")` permits GPU arithmetic but uses float64 CPU
matrix multiplication or original CPU attention below the measured GPU
crossover. These particular workloads all select CPU arithmetic.

| Workload | Plain DataFusion | EinFold CPU | Option B default dispatch | vs DataFusion |
|---|---:|---:|---:|---:|
| (256×512) · (512×256) | 36.510 | 8.698 | 4.775 | 7.65× |
| (1024×512) · (512×256) | 108.929 | 14.434 | 10.593 | 10.28× |
| Causal attention, 1,024 tokens, 64 features | 181.283 | 6.126 | 6.058 | 29.93× |
| Causal attention, 4,096 tokens, 64 features | 3,985.345 | 17.381 | 16.593 | 240.18× |

The float64 CPU matrix backend alone reduced the 256-row product from
**8.698 ms to 4.422 ms (1.97×)**, without reducing precision.

## Incremental GPU measurements

This run explicitly sets `min_gpu_ops=0` to compare direct GPU arithmetic
with both CPU implementations. The default threshold still selects GPU
for the 8,192- and 16,384-token attention cases. It selects CPU for this
matrix product. Each GPU case below completed eight GPU kernels, including
the warmup, as verified by `statistics()`.

| Workload | Original EinFold CPU | Float64 CPU backend | MLX GPU float32 | GPU vs original EinFold |
|---|---:|---:|---:|---:|
| (1024×1024) · (1024×1024) | 64.131 | 33.745 | 35.174 | 1.82× |
| Causal attention, 8,192 tokens, 64 features | 36.204 | 38.158 | 33.864 | 1.07× |
| Causal attention, 16,384 tokens, 64 features | 95.044 | 97.978 | 71.292 | 1.33× |

Float64 CPU matrix multiplication is slightly faster than the GPU for this
matrix. GPU acceleration is useful for large attention, and is not a
universal replacement for CPU execution. The calibration run at the earlier
`2**28` GPU threshold made 4,096-token attention about 6% slower than original
EinFold. The final `2**33` threshold keeps it on the CPU. The raw calibration
file is retained rather than hiding this regression.

## Repeated ddx gradients

The query computes `Y = A·B`, `L = SUM(Y²)/2`, and `grad(L, B)`;
`A` has 64 features and `B` has 32 outputs. `xarray_sql.ad.prepare` retains
ddx's programs, checks catalog/schema/settings changes, and materializes
final results. Dataset writes use the existing validation and read current
values; regular preparation also supports replacing tables by name.

| Rows in A | Option B SQL every call | Option B prepared | Repeated-step speedup |
|---|---:|---:|---:|
| 256 | 14.479 | 9.424 | 1.54× |
| 2,048 | 19.543 | 14.049 | 1.39× |

These small gradients select CPU arithmetic. At 2,048 rows, plain DataFusion
prepared execution remains faster (11.635 ms) than Option B.
GPU kernels cannot eliminate the planning and materialization cost of a
small SQL gradient program. Actual GPU gradient execution is covered by the
forced-offload correctness tests, including attention's backward pass.

## CPU profile and implementation decisions

The initial [cProfile output](cpu-profile.txt) covers attention at 16 and 64
tokens with DataFusion and released EinFold 0.1.0. Of 2.192 seconds in the
demo's main function, 0.977 seconds accumulated under ddx rewriting and
execution across 24 gradient calls. There were 336 intermediate
materializations, taking 0.678 seconds **inside** that time, not in addition
to it. DataFusion Arrow collection accounted for 0.939 seconds across 589
calls. These include native work; cProfile does not separate Rust/Metal
kernel instructions. The profiled initial timings are not mixed with the
unprofiled final tables.

Changes guided by this evidence and the kernel comparisons:

- Add `ad.prepare` to reuse differentiation programs and avoid repeated
  objective planning, program keying and view checks.
- Give EinFold a per-session native executor hook, after its dense input
  validation. The existing row/domain/NULL reconstruction stays intact.
- Replace sufficiently large dense scalar accumulation with NumPy float64
  matrix multiplication on CPU, or explicitly permitted MLX float32 GPU
  arithmetic. Native thresholds avoid Python buffer copies for tiny work.
- Bound attention query tiles, trim keys outside allowed intervals, and use
  MLX's causal or mask-free kernels only when normalized ranges match.
- Retain CPU fallback for special values and unsupported relational shapes;
  never cache input values across queries.
- Preserve xarray Context methods, Dataset metadata and prepared-query
  invalidation when installing DataFusion extension bundles.

## Training overheads and CPU parallelism

The additional overhead pass covers all six optimization directions:

| Direction | Implementation and evidence |
|---|---|
| Reduce planning and repeated CTE expansion | Forward `Context.prepare` retains the optimized logical plan, bypassing repeated SQL parsing and CTE expansion. ddx preparation retains differentiation programs. The repeated-CTE benchmark below checks current results. |
| Reuse plans across training steps | `Context.prepare` now accepts grad/VJP/JVP SQL as well as forward queries. It delegates gradient caching and catalog/settings checks to `ad.prepare`; multiple parameter updates still use `ad.prepare` to share a loss. |
| Reduce Python/Arrow registration | Uniform in-memory Datasets register Arrow batches directly, eliminating the temporary catalog table. Partition counts use arithmetic rather than enumerating the Cartesian chunk grid. Prepared writes retain shared value buffers instead of deregistering and repivoting tables. |
| Fuse numerical operators through EinFold | EinFold's physical planner recognizes contractions and fused attention. Attention scores, shifted exponentials, normalization and weighted sums use its fused pipeline, with original SQL fallbacks. |
| Execute fused operators with MLX | The session-local native hook dispatches eligible validated fused attention to MLX. The GPU measurements above separate its contribution from EinFold's existing rewrites. |
| Increase CPU concurrency | `Context(threads=...)` and the main benchmark's `--threads` expose the partition target. The default already uses 30 CPU cores; a 1/10/20/30/60 sweep and a 30/60 confirmation measure oversubscription rather than assume it helps. |

The new measurements use nine timed repetitions after one warmup. The
fine-chunk baseline was recorded before replacing chunk-grid enumeration;
the final result uses the arithmetic count and direct Arrow registration.
All sums, products, gradients and repeated-CTE query results were checked.

| Overhead case | Baseline | Improved | Speedup |
|---|---:|---:|---:|
| Register a 4,096×16 Dataset with one-cell chunks | 31.372 ms | 0.405 ms | 77.44× |
| Update and sum 1,048,576 values: register each step vs prepared writes | 2.769 ms | 1.027 ms | 2.70× |
| Repeated-CTE product, plain DataFusion: SQL each call vs prepared | 1.709 ms | 1.307 ms | 1.31× |

Registration still scales with Arrow coordinate construction for large
ordinary tables. The dramatic fine-chunk improvement removes 65,536
unneeded chunk-slice constructions; it does not imply that every Dataset
registers 77 times faster. Preparing is also not beneficial for every query.

The independent concurrency confirmation, on identical 2,048-token causal
attention inputs within that run, gave:

| Engine | 30 target partitions | 60 target partitions | Change |
|---|---:|---:|---:|
| datafusion | 790.149 ms | 747.087 ms | 1.06× faster |
| einfold | 8.737 ms | 9.076 ms | 3.9% slower |

The earlier full sweep found 1 → 30 partitions reduced plain DataFusion
attention from 6,797 ms to 795 ms and EinFold attention from 25.1 ms to
9.3 ms. The 30/60 confirmation supports leaving the default at the physical
CPU count and making higher concurrency explicit per workload. The setting
controls target parallel partitions, not a promise of exactly that many OS
threads or a change to NumPy's separate BLAS thread controls.

```python
ctx = Context(threads=60)  # explicit workload-specific concurrency
ctx.from_dataset("weights", initial_weights)
step = ctx.prepare(gradient_sql)
for values in training_batches:
    rows = step(weights=values)  # no table re-registration
```

Limits on plan/CTE reuse are explicit: the installed DataFusion 55 Python
API has no `ExecutionPlan.reset_state`. Forward preparation reuses the
optimized logical plan and uses physical copies only when supported and
cheaper; ddx preparation retains its programs while planning their execution
steps as needed. This does **not** provide a general physical-plan cache or
compute an arbitrary repeated CTE once at runtime. SQL `AS MATERIALIZED` is
rejected by this installed parser. No automatic CTE rewrite was added that
could change volatile expressions or introduce plan nodes ddx cannot
serialize through Substrait.

Raw data: [registration baseline](fine-chunks-before.json),
[initial overhead baseline](training-overheads-before.json),
[final overheads](training-overheads-final.json),
[full concurrency sweep](training-overheads.json), and
[30/60 confirmation](thread-confirmation.json).

```sh
.venv/bin/python -m benchmarks.apple.overheads --skip-thread-sweep \
  --output benchmarks/apple/training-overheads-final.json
.venv/bin/python -m benchmarks.apple.overheads --only-thread-sweep \
  --threads 30 60 --thread-tokens 2048 --output benchmarks/apple/thread-confirmation.json
```

## Correctness and limits

- Full Python unit suite after the registration, preparation and concurrency
  changes: **608 passed, 277 skipped, 102 integration tests deselected**,
  including **all 38 accelerator tests**. Coverage includes bags/duplicates, NULLs, sparse and empty
  inputs, NaN/infinity/signed zero/extreme magnitudes, partitioned batches,
  current-value reads, xarray metadata, prepared writes, grad/VJP/JVP,
  stop-gradient, typed ddx refusals and attention backward derivatives.
- EinFold native suite: **72 passed, 4 ignored**, with one existing
  plain-DataFusion assumption excluded: `nan_compares_by_its_sign`. It
  expects the NaN from `inf - inf` to be negative. Plain DataFusion on this
  ARM Mac produces a positive NaN, without the accelerator installed. The
  original test remains unchanged in the patch. Running the full native
  suite without the exclusion exposes this failure.
- Ruff formatting/lint and patch application checks passed. Focused mypy
  checks of `mlx.py` and `ad.py` passed. Repository-wide mypy reports the
  same five existing errors as an unmodified HEAD copy (in `df.py`,
  `backends/pyarrow.py`, `sql.py`, and `roundtrip.py`), with no added errors.
- Default float64 CPU products matched the NumPy reference exactly in these
  runs. Final GPU maximum absolute errors were **0.000216** for the
  matrix product and **6.08e-07** for attention. All benchmark executions
  checked results; GPU tolerances were `rtol=atol=3e-4`, while accelerator
  unit tests use `3e-5`. Float32 GPU mode explicitly changes numerical
  precision; it does **not** promise float64 rounding/error bounds.
- MLX's default M5 matrix mode initially failed those tolerances. Final
  tests and measurements disable it using `MLX_ENABLE_TF32=0` at process
  launch, as required by the accelerator. See
  [MLX numerical precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html).
- This is an optional integration requiring the included, pinned EinFold
  patch; installing the normal EinFold wheel alone does not provide the
  hook. The patched wheel is installed in this checkout's `.venv`.
- The 512 MiB backend estimate bounds requested working buffers, not all
  driver allocations or caches. Peak process/GPU memory was not measured;
  no peak-memory speedup or hard memory-cap claim is made.

## Reproduce and inspect

[Build and usage instructions](../../accelerators/einfold_mlx/README.md).
Final [default timings](m5-ultra-results.json), [large GPU timings](m5-ultra-large-results.json),
[threshold calibration](m5-ultra-crossover-results.json),
[initial profiled timings](initial-attention.csv), and [CPU profile](cpu-profile.txt).

```sh
MLX_ENABLE_TF32=0 .venv/bin/python -m benchmarks.apple.benchmark \
  --matrix-rows 256 1024 --features 512 --outputs 256 \
  --attention-tokens 1024 4096 --gradient-rows 256 2048 \
  --reps 7 --output benchmarks/apple/m5-ultra-results.json

MLX_ENABLE_TF32=0 .venv/bin/python -m benchmarks.apple.benchmark \
  --hosts einfold einfold_blas einfold_mlx \
  --matrix-rows 1024 --features 1024 --outputs 1024 \
  --attention-tokens 8192 16384 --gradient-rows 256 \
  --min-gpu-ops 0 --reps 7 --output benchmarks/apple/m5-ultra-large-results.json

MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python -m pytest -m 'not integration'
```

## nanoGPT model comparison

Benchmark 2 was also run with `--steps 5` across the original hosts and
explicit prepared CPU, automatic GPU and forced-kernel controls. See the
[full nanoGPT comparison](nanogpt-5-steps.md) for all eight variants, cold
versus warm timings, numerical checks and actual GPU kernel counts. GPU
microbenchmark gains do not imply gains for this small end-to-end model.

The subsequent [shared-CTE comparison](nanogpt-shared.md) combines shared
common table expression execution, cached plans and MLX GPU arithmetic,
with five-step training and larger nanoGPT configurations. It records
successful checks separately from memory-limited full-vocabulary runs.

The [native JAX and nanoGPT comparison](nanogpt-frameworks.md) measures the
same full-parameter model directly in JAX and the upstream PyTorch model
on this M5 Ultra. It checks every update and shows the remaining large
performance gap between SQL execution and native array training.

The later [complete performance report](../../NANOGPT_PERFORMANCE.md)
tracks compact execution and the resident Shakespeare trainer, with
startup, per-step and memory measurements. Its [resident summary](resident-summary.json)
records the 100-step result and the updated native PyTorch comparison.
