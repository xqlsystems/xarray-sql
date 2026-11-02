# nanoGPT benchmark 2: five-step CPU/GPU comparison

Every variant ran **`--steps 5`**, SGD, batch 4, one layer, 16 embedding
channels, two heads, context 16, 4,416 parameters and the same Tiny
Shakespeare batches. This keeps the model and SQL from
[`02_nanogpt.py`](../ml/02_nanogpt.py); JAX checks the loss and every
parameter table after each update. Original inputs are copied into the
resident hosts so in-place updates cannot mutate another host's initial
weights or batch data. The five reference losses are identical across
all eight variants.

Machine: Apple M5 Ultra, 30 CPU target partitions, 64 GPU cores, 256 GiB.
MLX 0.32.4 with `MLX_ENABLE_TF32=0`; patched EinFold 0.2.0+xqlmlx.1,
DataFusion 55.0.0, ddxdb 0.4.6 and DuckDB 1.5.6. The seven-way comparison
ran in one process; the extra forced CPU-kernel control ran separately.
One sample per step, as requested by the script's default `--reps 1`.
These are not seven-repetition kernel microbenchmarks.

The measured five-step total adds the five loss queries, five training
queries, cumulative table load/update time and model-view creation time.
It includes first-use preparation/compilation. It excludes JAX reference
work, correctness checking, result-to-Dataset reconstruction, operator
inspection and console/reporting time, so it is **not wall-clock runtime**.
The warm step column averages paired loss + training timings for steps
1–4, after the initial step. Each step uses a different batch.

| Variant | Five loss queries | Five training queries | Measured five-step total | Warm step mean | Completed GPU matrix kernels |
|---|---:|---:|---:|---:|---:|
| DataFusion baseline | 5.331 s | 1.043 s | 6.504 s | 1227.4 ms | 0 |
| EinFold baseline | 5.832 s | 1.894 s | 7.856 s | 1499.0 ms | 0 |
| DuckDB baseline | 0.127 s | 1.315 s | 1.481 s | 278.5 ms | 0 |
| Prepared DataFusion CPU | 6.093 s | 0.910 s | 7.036 s | 1103.2 ms | 0 |
| Prepared EinFold + CPU auto | 6.736 s | 1.621 s | 8.392 s | 1343.4 ms | 0 |
| Prepared EinFold + CPU forced | 6.687 s | 1.600 s | 8.324 s | 1341.2 ms | 0 |
| Prepared EinFold + GPU auto | 6.740 s | 1.611 s | 8.385 s | 1345.8 ms | 0 |
| Prepared EinFold + GPU forced | 7.316 s | 1.955 s | 9.305 s | 1480.7 ms | 2015 |

## What the variants actually execute

- Baselines recreate the model view and register updated parameter tables
  as the original demo does. They use the current checkout's registration
  fixes, but do not enable prepared training or the new kernel bridge.
- `datafusion_prepared` registers tables once, writes updated values into
  their existing buffers, keeps the unchanged model view, caches the
  forward `Context.prepare` result and reuses `ad.prepare` for all eleven
  parameter statements sharing the loss.
- `cpu_optimized` adds the float64 CPU matrix backend and EinFold planning
  with the normal operation threshold. At this tiny model size, no new
  CPU backend kernel passes the threshold.
- `gpu_optimized` uses the same prepared/resident path and normal GPU
  dispatch thresholds. **It executes zero GPU kernels on this model**.
  It should not be reported as a GPU speedup.
- `cpu_forced` lowers the operation threshold to zero for float64 CPU
  backend calls. It completes **2,015 CPU matrix kernels**.
- `gpu_forced` lowers the CPU/Python and GPU thresholds to zero. It
  completes **2,015 actual MLX GPU matrix kernels**. No GPU fused-attention
  kernels were recorded. This exercises the GPU bridge while exposing
  dispatch and buffer-conversion costs of tiny products.

The resident hosts each recorded 13 initial table registrations, 59
in-place updates, one forward-query preparation and one shared gradient
program preparation. The original loop instead reloads the tables and
recreates its view after each batch.

## Findings

- Warm prepared DataFusion steps improved **1227.4 → 1103.2 ms (1.11×)**. Warm training-query time alone improved **161.8 → 125.9 ms**.
- Over only five steps, prepared DataFusion's initial planning cost outweighed the warm gains: **6.504 s baseline versus 7.036 s prepared**. The first prepared loss took 2.183 s; later forward losses still took about 0.977 s.
- Forced GPU execution was **10.4% slower** per warm step than the forced float64 CPU backend. The CPU/GPU bridge and physical-plan work outweigh GPU arithmetic savings at these shapes.
- The automatic GPU threshold correctly keeps this small model on CPU.
  The cached optimized logical plan does not remove the remaining large
  physical-planning cost of the forward loss. This benchmark does not
  reproduce the large-attention GPU gains from the separate tensor tests.
- DuckDB remains the fastest complete five-step variant in this run.
- **All 80 loss/update records passed their JAX checks.** Float64 hosts use
  `rtol=atol=1e-9`. GPU-enabled hosts use explicit float32 permission and
  `rtol=atol=3e-5`. Forced GPU's largest reported error relative to the
  reference's maximum magnitude was about **3e-7**, versus about **1e-15**
  for float64 controls. Every individual array also passed the elementwise
  tolerance check. Reference losses were 4.179251, 4.104661, 3.883857,
  3.641763 and 3.842234 (different batches).

## Reproduce and inspect

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/02_nanogpt.py \
  --steps 5 --hosts datafusion einfold duckdb datafusion_prepared \
  cpu_optimized gpu_optimized gpu_forced \
  --csv benchmarks/apple/nanogpt-5-steps-comparison.csv

MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/02_nanogpt.py \
  --steps 5 --hosts cpu_forced \
  --csv benchmarks/apple/nanogpt-5-steps-cpu-forced.csv
```

[Combined CSV](nanogpt-5-steps-all-variants.csv),
[JSON summary](nanogpt-5-steps-summary.json),
[seven-way log](nanogpt-5-steps-comparison.log),
[forced CPU log](nanogpt-5-steps-cpu-forced.log).
The earlier original-default-host run is also retained in
[nanogpt-5-steps.csv](nanogpt-5-steps.csv).
