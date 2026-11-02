# EinFold kernel bridge for Apple silicon

[EinFold](https://github.com/xqlsystems/einfold) recognizes tensor operations
in Apache DataFusion SQL plans. [MLX](https://github.com/ml-explore/mlx) is
Apple's array framework for Apple silicon. This implements Option B:

```text
SQL → DataFusion logical plan → ddx rewrite, when requested
    → EinFold detection and SQL input validation
    → float64 CPU matrix multiplication or float32 MLX GPU kernels
    → EinFold row/domain reconstruction → DataFusion → Arrow/xarray
```

[ddx](https://github.com/xqlsystems/ddx), the SQL automatic differentiation
engine, constructs every gradient, VJP and JVP. Logical plans remain normal
DataFusion plans until physical planning, so ddx can serialize them through
Substrait. MLX only evaluates arithmetic inside those plans.

The released EinFold Python API has no external executor hook. The optional
builder pins the feature commit `feaca83e4cc1c14c7cbb5f5acc734133005ad57f` in the [EinFold feature fork](https://github.com/cromwellian/einfold),
which contains the native bridge and compact array module. Its tested base
is `39b25210d3b39027ef91584383fb114dfa8a4a48` (Python 0.2.0). Historical
benchmark wheels were labeled `0.2.0+xqlmlx.6`; the contributed source retains
version 0.2.0. This feature is not in the ordinary PyPI wheel yet. The
xarray-sql Rust extension supplies SQL parsing and verified ordered Arrow
providers used by shared execution.

## Owned state and complete training graphs

`xarray_sql.training.TrainingSession` adds an explicit ownership boundary
for complete, finite, nonnullable rectangular state tables. It imports
weights and optimizer moments once, then retains MLX arrays between SQL
updates. Ordinary `DenseExecutor` queries retain their existing live-input
and Arrow snapshot behavior. Use a dedicated Context for a session and
read its current state through `session.query` or `session.export`.

`session.prepare_step` receives the original SQL gradient statements,
ordinary SQL intermediate/optimizer statements and a mapping from their
outputs to persistent state tables. It evaluates gradients, clipping and
all parameter updates in one invocation. The optional
[ddx math/primal build](../ddx_math/README.md) exposes the already computed
primal loss. No independent loss-forward call is needed.

With `compile=True`, the session traces these SQL/ddx array programs into
one cached `mx.compile` function, with mutable numerical state passed as
explicit array arguments. Schemas, coordinate layouts, view definitions
and session settings guard reuse. Structural checks still run or must be
proven from the coordinate/count facts; a check that requires an
unsupported dynamic host decision declines compilation. Device finite,
NULL and complete-domain checks return a status scalar before all new
state tables are published together. Failed updates retain the previous
state and RNG counter.

The pinned EinFold integration supplies device tensor leaves, finite integer
domain metadata, GPU index joins and embedding scatter grouping. Declare
bounded integer inputs with `session.index_domains` before binding them.
Actual input values are checked against those domains. Numerical index
arithmetic requires an explicit floating cast; unsupported integer/domain
behavior declines instead of silently changing SQL arithmetic. Schema or
coordinate changes require a new session. Explicit `bind` replacements
copy caller inputs and preserve ownership; exports are independent Arrow
snapshots. Numerical host conversion during compilation is refused.

`session.dropout` generates masks from an explicit seed and step key on
GPU. Save the seed and next step with the checkpoint. Complete training
results can retain masks and raw gradients for correctness inspection;
these exports are outside benchmark timers. Ordinary steps read only the
validation and loss scalars. Full state is exported for checkpoints.

From the xarray-sql directory, the matching Shakespeare recipe uses:

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python benchmarks/ml/03_shakespeare.py \
  --compile-training --gpu-dropout --steps 100
```

Omit `--steps` for the recipe's full training run. `--resident` exercises
the intermediate implementation with device state and shared primal loss,
without whole-step compilation. `--compile-training` without GPU dropout
keeps NumPy-generated masks as explicit inputs, useful for comparison with
the earlier deterministic trajectory. CPU resident execution uses
`--resident --backend dense_cpu --cache-gib 32`.

## Compact execution across forward and backward stages

`xarray_sql.dense.DenseExecutor` is an opt-in implementation of
SQL → DataFusion logical plans → EinFold array programs → NumPy/MLX.
It retains shared CTE and ddx stage values as private arrays with explicit
coordinate axes. Numerical joins become gathers or contractions, window
reductions broadcast, and reduced UNION branches accumulate gradients.
Coordinates expand into scalar Arrow columns at public output boundaries.
The executor orders compact axes before expansion where their exact domains
prove the requested SQL ordering.

```python
from einfold.datafusion import EinFold
from xarray_sql import Context
from xarray_sql.dense import DenseExecutor

ctx = Context(threads=30).with_extensions(EinFold())
# Register input and parameter Datasets on ctx.
executor = DenseExecutor(ctx)  # float64 CPU by default
# Explicitly permit float32 GPU arithmetic, with MLX_ENABLE_TF32=0 at launch:
# executor = DenseExecutor(ctx, precision="float32")
executor.register_view("model", model_select_sql)
loss = executor.prepare(loss_select_sql)
step = executor.prepare_all(gradient_update_sqls)
loss_rows = loss()
updated_parameter_rows = step()
print(executor.statistics())
```

ddx differentiates the original SQL. This path does not use MLX automatic
differentiation or replace a model with framework code. Logical programs
and compiled elementwise functions are reused across steps; compiled
functions receive every numerical leaf as an argument. Input NumPy buffers
are imported afresh each call, preserving in-place writes and registration
changes. Private schema-only providers serve ddx's named reads; they are
never scanned to produce numerical results. Validation checks evaluate
actual rows. Public outputs are independent Arrow snapshots, and input
parameters change only when the caller writes an update.

The compact source contract is a rectangular Dataset with unique integer
or string coordinates and finite float64 numerical buffers. The initial
implementation supports a closed set of arithmetic, CASE/NULL operations,
gathers, grouped reductions, full-partition windows, axis joins and domain
left joins. Unsupported layouts or operations restart the call through
the existing shared SQL executor before any parameter is changed.
`fallback=False` reports unsupported cases directly and is used by the
benchmarks. Custom scalar-function overrides stay in SQL. GPU precision is
explicit; it retains the existing `3e-5` benchmark tolerance and the
float32 AdamW limitations documented in the measurements.

The `.5` integration also retains multiplicative factors through aliases
and zero-valued CASE branches when reducing a conditional product. This
keeps attention backward masks inside contractions instead of expanding a
batch × heads × context × context × channels temporary. Exceptional
arithmetic retains the normal SQL fallback. Exact GELU is available with
the optional [ddx erf build](../ddx_math/README.md) and
`xarray_sql.math.register_erf(ctx)`; arbitrary UDF overrides remain in SQL.
The [Shakespeare trainer](../../benchmarks/ml/03_shakespeare.py) uses
these features for dropout and AdamW at nanoGPT's character-model size.

GPU intermediates remain MLX arrays through their supported operations.
Compiled pointwise regions use `mx.compile`; contractions use MLX matrix
operations and embedding gradients use scatter reductions. Exact counts
use integer arithmetic, with bounded int32 scatter where Metal requires
it. GPU readbacks occur at SQL validation/output boundaries. Caches retain
program structure across calls; numerical graphs are released after each
call. This API still imports and exports the full parameter state each
training call. An API that owns persistent GPU training state is a future
extension with a different ownership and synchronization contract.

The computed-value cache defaults to 8 GiB per invocation; adjust it with
`max_cached_bytes`. Source buffers remain available throughout the call;
computed values may be evicted and recomputed. `max_array_bytes` defaults
to a conservative 1 GiB estimate for a numerical temporary. These are
array/cache bounds, not an RSS limit: source tables, public Arrow results,
working allocations and driver memory are additional. The benchmark
runner enforces its own process limit and accepts `--dense-cache-gib`.
Exact coordinate and nonnullable-domain facts keep row and NULL masks
compact, including UNION branches whose domains differ on one axis.

The [dense execution report](../../benchmarks/apple/dense-execution.md)
contains measured speedups, memory, copy counters, failed attempts and
remaining overheads. The
[research and execution plan](dense-execution-plan.md) records the source
research and implementation sequence.

## Install and run

Requires Apple silicon, a working Metal GPU, Python 3.10+, Rust 1.94+,
Git, uv and maturin. In the xarray-sql checkout:

```sh
uv sync --extra mlx --extra ml
.venv/bin/python accelerators/einfold_mlx/build.py
MLX_ENABLE_TF32=0 .venv/bin/python your_program.py
```

The builder checks out the pinned feature commit into a temporary directory,
builds its locked wheel, and installs only into the interpreter's environment.
No patch is applied and no EinFold source is duplicated in this repository.
A later `uv sync` can replace the custom wheel; rerun the builder afterward.
Use `--source /path/to/einfold` to build a clean checkout already at the pin.
For supplied benchmarks, `.venv/bin/python -m benchmarks.apple.benchmark`
runs this checkout directly. Directly running an ML demo needs
`PYTHONPATH="$PWD"` unless xarray-sql is installed as editable.

The pinned EinFold wheel includes `einfold.dense_program`, used by the
optional `DenseExecutor` to retain compact arrays across SQL stages.

```python
from xarray_sql import Context, ad
from xarray_sql.mlx import MLX

accelerator = MLX(precision="float32")
ctx = Context(threads=30).with_extensions(accelerator)
ctx.from_dataset("x", dataset)
rows = ctx.sql(statement).to_arrow_table()
print(accelerator.statistics())

# One SQL gradient query can use the same preparation API as forward SQL.
single_step = ctx.prepare(gradient_statement)
rows = single_step(x=new_dataset_values)

# Several parameter updates can share one loss and its ddx programs.
step = ad.prepare(ctx, [gradient_statement, update_statement])
gradient_rows, updated_rows = step(x=new_dataset_values)
```

`Context.with_extensions()` returns another xarray-aware handle, sharing
Dataset metadata and prepared-query invalidation with the original handle.
Always use the returned handle, which carries the extension's codecs.
`threads=` controls DataFusion's target parallel partitions. The default is
all 30 CPU cores on the measured M5 Ultra; 60 partitions modestly helped
plain DataFusion attention but slightly slowed EinFold in the confirmation
run. The [overhead report](../../benchmarks/apple/README.md#training-overheads-and-cpu-parallelism)
contains both results.

## Numerical and SQL contracts

- `MLX()` defaults to **float64 CPU** matrix multiplication through NumPy;
  attention keeps EinFold's original CPU implementation. No implicit
  conversion of double precision scientific data is enabled by default.
- `MLX(precision="float32")` explicitly permits single precision GPU
  arithmetic, including intermediate values used by ddx. Outputs keep the
  SQL Arrow dtype. **It does not preserve float64 rounding or error bounds.**
- Start Python with `MLX_ENABLE_TF32=0`. On M5, MLX otherwise uses arithmetic
  with less precision than float32 for matrix operations. The extension
  requires this setting and does not mutate process-wide MLX settings.
  See [MLX's numerical precision documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html).
- EinFold retains key equality, bag semantics, NULLs, row existence,
  grouping, domain defaults, and output mask reconstruction. Duplicate,
  sparse, non-positional and unsupported inputs keep EinFold/DataFusion
  fallback behavior. Non-finite factors stay on the CPU. Very large or tiny
  finite factors also decline GPU execution.
- GPU attention is limited to complete finite inputs, product scores,
  shifted exponentials and normalized weights. Unshifted exponentials,
  general score expressions and other pipelines use existing CPU kernels.
  Key intervals come from EinFold's normalized mask conditions. Masked
  infinities never enter GPU arithmetic.
- There is no input-value cache: repeated queries and in-place parameter
  writes read current values. Backend ownership is session-scoped; ordinary
  DataFusion and EinFold contexts can coexist with accelerated contexts.
- Gradient definitions and typed ddx refusals are unchanged. `ad.prepare`
  reuses ddx programs while checking catalog/schema/settings changes;
  it materializes final results so later calls cannot overwrite them.
  It intentionally uses ddx's regular preparation, which supports table
  replacement, rather than experimental in-place physical-plan reuse.

## Dispatch and memory

The native side declines small requests before copying buffers into Python
(default `min_ops=2**20` scalar products). In float32 mode, GPU dispatch
starts at `min_gpu_ops=2**33`; smaller products use float64 CPU matrix
multiplication and smaller attention uses EinFold's CPU tiles. This avoids
GPU dispatch regressions on small workloads. Both limits can be adjusted;
`min_ops=0, min_gpu_ops=0` forces eligible computations through the GPU for
correctness testing or threshold experiments.

Attention uses at most 1024 queries per tile, with a smaller tile when the
backend's working-memory estimate requires it. Masks and temporary score
storage grow with the number of keys and tile width, rather than the full
number of query/key pairs. Each tile trims keys outside the union of its
allowed intervals. Rectangular and causal intervals use MLX's mask-free or
causal kernels when their ranges match exactly; other masks remain explicit. Backend buffers are bounded by a configurable
512 MiB estimate and reserved through EinFold's DataFusion memory budget.
This estimate is not a hard limit on the GPU driver's allocator or its
kernel cache. All GPU results synchronize before Arrow conversion.

`statistics()` counts completed `cpu_matmul`, `gpu_matmul`, and
`gpu_attention` kernels, plus requests declined inside Python. Requests
rejected by native thresholds are not counted. Planning a GPU-capable
operator alone does not increment a completion counter.

## Validation

See [the measured report](../../benchmarks/apple/README.md), the standalone
benchmark, raw timings and CPU profile in `benchmarks/apple`.

```sh
MLX_ENABLE_TF32=0 PYTHONPATH="$PWD" .venv/bin/python -m pytest -m 'not integration'
MLX_ENABLE_TF32=0 .venv/bin/python -m pytest tests/test_mlx.py
```

The native contribution also passes EinFold's kernel and randomized invariant
suite, except for the existing plain-DataFusion `nan_compares_by_its_sign`
assumption on this ARM Mac. That test expects `inf - inf` to produce a
negative NaN. Here plain DataFusion produces a positive NaN, without any
extension installed. The test remains unchanged in the contribution and was
documented in the measured report; it is not silently removed.

## Shared CTE execution

The historical `0.2.0+xqlmlx.3` build added safe prepared-plan resets and multi-key
attention detection. Batch and head keys must match in the scores, values,
softmax partition and normalizer. The kernel validates complete positional
domains before treating those groups as independent attention problems.

`xarray_sql.shared.SharedExecutor` materializes reachable CTEs read more than
once and caches each stage's physical plan. It writes new stage values into
owned Arrow buffers. A changed coordinate domain, row count or schema
invalidates plans that read the replaced buffer. Dynamic filter pushdown is
disabled on this explicit executor's context, because its cached filters
would otherwise retain values from an earlier training step.

This reset also runs when DataFusion's FFI wrapper replaces children:
DataFusion 55 does not forward a separate reset callback through that
wrapper. Ordinary `Context.prepare` conservatively retains only the
optimized logical plan when dynamic filters are enabled.

```python
from xarray_sql import Context
from xarray_sql.mlx import MLX
from xarray_sql.shared import SharedExecutor

accelerator = MLX(precision="float32", min_ops=0, min_gpu_ops=0)
ctx = Context().with_extensions(accelerator)
# Register parameter/input Datasets on ctx first.
executor = SharedExecutor(ctx, inline_ctes=["scores", "weights", "norm"])
executor.register_view("model", model_select_sql)
loss = executor.prepare(loss_select_sql)
step = executor.prepare_all(gradient_update_sqls)
```

Virtual views retain their original SQL. ddx receives those definitions,
not the forward executor's materialized tables, and builds the backward
pass before its own stages execute. Numerical work in both paths passes
through DataFusion and EinFold before the MLX callback. Returned Arrow
tables are independent snapshots.

DataFusion 55's Substrait consumer cannot apply a projected read to an
expanded view. When a ddx stage reads a deferred view, the executor binds
the same program through ddx's SQL writer instead. It retains the original
SQL differentiation and avoids substituting forward materializations into
the derivative's lineage.

Join-only ddx stages remain logical views until a consumer reduces their
rows, avoiding storage of the full contraction join. Very large cotangent
sums (at least 32 UNION branches) materialize their reductions sequentially
to limit concurrent join builds. Compound reduction stages rebind transient
ddx providers on each call. They retain output storage, while ordinary
forward and backward stages also retain physical plans.

An exact, bounded coordinate cache shares immutable integer/string buffers;
floating point values always have separate writable storage. Every cache
hit checks all values and NULLs. The default budget is 256 MiB, adjustable
with `max_key_cache_bytes`. Canonically ordered stage tables expose verified
ordering to DataFusion without an additional sort. Coordinate/domain/schema
changes invalidate dependent plans. See the
[shared nanoGPT measurements](../../benchmarks/apple/nanogpt-shared.md)
for measured benefits and remaining large-model memory limits.

This API explicitly supports deterministic numerical SELECTs with flat,
nonrecursive WITH clauses. It refuses unknown functions and nested WITH
scopes. Callers identify CTEs that should remain inside a fused region with
`inline_ctes`. Stage boundaries still exchange Arrow data: this does not
implement a GPU-resident training graph or replace SQL derivatives with
MLX autodiff. The native xarray-sql extension must also be rebuilt for its
SQL-parser helper (`maturin develop --release` in the checkout).
