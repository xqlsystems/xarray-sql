# Native JAX and nanoGPT comparison on the M5 Ultra

The native frameworks are substantially faster than the SQL training path.
These local measurements use the same model dimensions, initial weights,
Shakespeare batches, loss and SGD update as the SQL benchmark, on the same
Apple M5 Ultra with serial execution and no game contention.

[nanoGPT](https://github.com/karpathy/nanoGPT) is Andrej Karpathy's PyTorch
implementation of GPT-2. Its [bench.py](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/bench.py)
provides a training benchmark: 12 layers, 12 heads, width 768, batch 12,
context 1024, AdamW, optional compilation, 10 warmup steps and 20 timed
steps. Its vocabulary is padded to 50304 and its timing loop uses CUDA
synchronization. Those defaults differ from our five-step SQL run.

The local harness imports the upstream model at revision
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. It sets vocabulary 50257,
batch 1, no biases/dropout, SGD at 0.01 and tanh GELU to match SQL. Upstream
normally uses exact GELU and its benchmark uses AdamW. These measurements
describe a matched configuration of the actual nanoGPT model, rather than
an unmodified execution of stock `bench.py`.

## Parameter count and checks

**The training model has 123,563,520 trainable parameters.** The harness
computes the number from the allocated NumPy arrays. PyTorch independently
checks its model's parameter count against it.

| Parameter group | Elements |
|---|---:|
| Tied token/output embedding | 38,597,376 |
| Positional embedding, context 16 | 12,288 |
| Q, K, V and output projections, 12 layers | 28,311,552 |
| MLP expansion and projection, 12 layers | 56,623,104 |
| LayerNorm gains | 19,200 |
| **Total** | **123,563,520** |

Each training step computes a loss, backpropagates through all 12 layers
and updates every parameter array. Timers wait for both loss and parameter
updates to finish. Loss and every updated parameter are checked against
the SQL demo's float64 JAX reference outside the timer, after every step.
All **45 native training/forward checkpoint records pass** across nine
runs. CPU uses `rtol=atol=1e-9`; GPU uses `3e-5`.

Batch 1 and context 16 mean 16 target positions per training step, or 80
across five steps. The output projection computes all 50257 classes.
This is a full-parameter, short-context training benchmark. Context 1024
adds positional weights, giving 124,337,664 parameters.

## Five-step training at context 16

Warm time averages steps 1–4. The five-step column includes initial model
setup and explicit JIT compilation where applicable, and excludes imports,
common input/weight generation and correctness validation. JAX JIT takes
3.00 seconds to compile its training step; that cost is included below.

| Implementation | Device / arithmetic | Warm step | Five steps + setup/compilation |
|---|---|---:|---:|
| SQL shared CPU | CPU / float64 | 164.034 s | 829.992 s |
| SQL shared MLX | GPU kernels float32, Arrow state float64 | 163.646 s | 829.973 s |
| Native JAX, separate loss/backward calls | CPU / float64 | 0.3943 s | 4.767 s |
| Native JAX, eager value-and-gradient | CPU / float64 | 0.3344 s | 4.433 s |
| Native JAX, compiled training step | CPU / float64 | 0.1364 s | 3.883 s |
| nanoGPT PyTorch, eager | CPU / float64, 30 threads | 0.0953 s | 4.122 s |
| nanoGPT PyTorch, eager | Apple MPS GPU / float32 | 0.02375 s | 3.085 s |

Relative to SQL shared CPU, the separate-call JAX case is about 416 times
faster per warm step, compiled JAX about 1203 times, and native PyTorch CPU
about 1722 times. Native PyTorch MPS is about 6890 times faster than SQL
shared MLX per warm step. Including first-use setup reduces these ratios:
five steps plus setup take about 830 s in SQL versus 3–5 s in native runs.
Four warm observations are not a confidence interval.

Native training normally reuses one forward pass for its backward pass.
The SQL interface separately evaluates a loss query and a gradient/update
program. `jax_separate` reproduces that two-call structure and remains
hundreds of times faster. Native state stays in framework arrays between
steps. SQL timings include collecting updated parameters into Arrow;
native parameter readback for correctness checking is outside its timers.
PyTorch MPS keeps float32 state, while the SQL GPU bridge retains float64
Arrow storage. These ratios include those execution/storage differences;
they measure complete implementations, with different output interfaces.

## Context 1024: forward-only comparison

All five losses cover all 1024 target positions and all 50257 output
classes. There are no updates, matching the existing SQL forward run.

| Implementation | Warm forward loss |
|---|---:|
| SQL shared CPU | 6.5726 s |
| SQL shared MLX | 6.8571 s |
| Native JAX CPU, eager | 0.4639 s |
| Native JAX CPU, JIT | 0.3199 s |
| nanoGPT PyTorch CPU float64 | 0.3453 s |
| nanoGPT PyTorch MPS float32 | 0.03117 s |

SQL shared MLX is about 220 times slower than native PyTorch MPS on this
forward loss. Full-context SQL backward performance remains unestablished
within the configured memory bound.

## Interpretation and reproduction

The SQL path remains far behind native array training. Integer coordinate
processing, Arrow storage/conversion, planning and large relational
backward intermediates dominate. GPU matrix and attention kernels alone
do not remove those costs. The training comparison exposes a much larger
gap than the forward-only comparison. See the
[SQL stage profile and results](nanogpt-shared.md).

Versions: JAX/jaxlib 0.12.0.1, PyTorch 2.15.0, NumPy 2.5.4. JAX devices
are CPU-only in this environment. PyTorch was installed in a temporary
environment, leaving the checkout's dependencies unchanged. PyTorch CPU
uses 30 threads. MPS timing calls `torch.mps.synchronize()` and JAX timing
blocks on all returned state.

```sh
git clone https://github.com/karpathy/nanoGPT.git /private/tmp/xql-upstream-nanogpt
git -C /private/tmp/xql-upstream-nanogpt checkout 3adf61e154c3fe3fca428ad6bc3818b27a3b8291
uv venv --python "$PWD/.venv/bin/python" /private/tmp/xql-framework-venv
uv pip install --python /private/tmp/xql-framework-venv/bin/python torch==2.15.0

MLX_ENABLE_TF32=0 PYTHONPATH="$PWD:$PWD/.venv/lib/python3.12/site-packages" \
  /private/tmp/xql-framework-venv/bin/python benchmarks/apple/compare_nanogpt_frameworks.py \
  --backend jax_jit --context 16 --steps 5 \
  --output benchmarks/apple/local-jax-jit.json

MLX_ENABLE_TF32=0 PYTHONPATH="$PWD:$PWD/.venv/lib/python3.12/site-packages" \
  /private/tmp/xql-framework-venv/bin/python benchmarks/apple/compare_nanogpt_frameworks.py \
  --backend torch_mps --context 16 --steps 5 \
  --nanogpt-source /private/tmp/xql-upstream-nanogpt \
  --output benchmarks/apple/local-torch-mps.json
```

Use `--context 1024 --forward-only` for the larger forward comparison.
The harness also supports `jax_eager`, `jax_separate` and `torch_cpu`.
[All results and ratios](nanogpt-framework-summary.json),
[serial run records](framework-runs.json),
[JAX training result](framework-gpt2-16-jax_jit.json),
[PyTorch MPS training result](framework-gpt2-16-torch_mps.json).
Source revisions/hashes, losses, errors and per-step timings are retained
alongside each run's JSON and log.
