# Persistent GPU execution of SQL training

The execution changes in this plan are implemented in
[`TrainingSession`](../../xarray_sql/training.py) and the optional
EinFold `.6` / ddx `.3` builds. The original plan below records their
implementation order and acceptance criteria. Whole-step compilation,
persistent state, shared primal loss, GPU dropout and explicit scalar
validation are available in the Shakespeare trainer. Final validation and
measurements are recorded in the resident execution report.

[ddx](https://github.com/xqlsystems/ddx) differentiates SQL;
[EinFold](https://github.com/xqlsystems/einfold) lowers numerical query
plans to arrays; [MLX](https://github.com/ml-explore/mlx) executes the
supported array arithmetic on Apple silicon. The goal is to retain this
pipeline while keeping numerical training state in MLX between updates.

## Current boundaries

The [Shakespeare trainer](../../benchmarks/ml/03_shakespeare.py) calls a
loss query, exports all gradients, registers them again, exports the
clipping coefficient, then exports and registers weights and AdamW
moments. Each `DenseExecutor._invoke` clears its transient tensor store
and creates a new computed-value cache. These are deliberate properties
of the current live-Dataset/Arrow snapshot API.

The current [measurements](../../benchmarks/apple/shakespeare-training.md)
are 2.024 s per warm SQL GPU update versus 0.02935 s in native PyTorch MPS
for the 10.7M-parameter recipe. This plan does not attribute the entire
difference to exports; Python plan interpretation, memory allocation,
kernel dispatch and numerical operation selection also remain.

Apple silicon uses [unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).
The costs to eliminate are conversion, packing, allocation and forced
synchronization, rather than assuming every CPU/GPU boundary transfers
data over PCIe.

## Implementation order

1. **Retain device tables within a complete training step.** Add an
   opt-in internal execution path returning compact tables whose numerical
   columns are MLX arrays. Execute gradients, global clipping and every
   existing SQL AdamW statement in one invocation before exporting results.
   Bind temporary gradient/state names to these actual device tables.
   Schema providers supply planning metadata and must never substitute for
   numerical evaluation or required validation. This first phase can still
   export the final updated state, establishing correctness independently
   of persistence.

2. **Expose the primal value from ddx's backward program.** In the pinned
   ddx release, `BackwardProgram.value` already identifies the loss output;
   the Python binding exposes it. Read that output before transient tables
   are released, together with the existing gradients. If the high-level
   SQL preparation API does not expose the corresponding output handle,
   add that bridge or execute the public backward program directly.
   Do not assume adding an independent SELECT to a batch automatically
   shares its forward computation. The loss and backward must use the same
   saved forward values and dropout masks.

3. **Add explicit persistent state ownership.** A new training-state
   object owns weight and moment arrays, dtype/layout metadata, coordinate
   domains, version and RNG state. Import the initial arrays once. Bind
   the next update's SQL table names directly to the preceding update's
   device outputs. Retain persistent state separately from transient
   activations/cotangents; release temporary graphs after each step.
   Publish all updated tables together only after checks succeed, retaining
   the previous version on failure. Do not silently change the existing
   API's handling of caller edits to NumPy buffers. Explicit replacement,
   snapshot/export and close operations define the new ownership boundary.

4. **Compile the complete numerical program.** Lower the prepared
   forward/backward/optimizer DAG to stable array instructions once rather
   than walking hundreds of DataFusion plans each update. Compile that
   program with explicit weights, moments, batch, learning-rate/step scalars
   and RNG inputs. Outputs include next state and the primal loss.
   Cache by SQL/program identity, schema, coordinate/layout facts, shapes,
   precision and UDF versions; invalidate when these change. Changing
   scalar values must not trigger a compilation or capture stale weights.
   Whole-program compilation can fuse pointwise work and share expressions;
   it does not imply that every GEMM becomes one GPU kernel.

5. **Keep masks, reductions and checks on the device.** Generate dropout
   masks on MLX with checkpointed RNG state. Keep an explicit-mask mode for
   exact comparisons against the current trainer/native oracle. Compute
   the global gradient norm and all AdamW arithmetic on GPU. Prove static
   coordinate uniqueness/domain properties from immutable metadata where
   possible. Otherwise execute checks on actual device values and read a
   scalar status before publishing an update. Preserve errors, NULL and
   row-domain behavior; do not skip validation because providers have zero
   rows. Unsupported plans decline before persistent state changes.

6. **Make synchronization explicit.** Evaluate the step's device outputs
   together and release references to previous computational histories.
   Read the loss when logging rather than before backward. Export complete
   arrays only for checkpoints, explicit inspection and correctness checks.
   A required scalar validation result may still synchronize once per
   update. Timed benchmarks must wait for device completion; an asynchronous
   submission time is not an update time.

MLX documents [whole-training-graph compilation and explicit state](https://ml-explore.github.io/mlx/build/html/usage/compile.html#compiling-training-graphs),
including RNG state for dropout. Its examples use MLX differentiation;
this executor would instead compile the array operations produced from
ddx's SQL derivatives. The same compilation mechanism can be used without
replacing the derivative source. MLX also documents how
[scalar extraction forces evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

## Contract and acceptance criteria

- SQL remains the source of the forward model, clipping and optimizer;
  ddx remains the source of derivatives. Dense-only assumptions, precision
  and ownership are explicit. Ordinary Arrow-returning SQL APIs continue
  to produce independent snapshots.
- Loss, raw gradients, weights, both moments and evaluation results match
  the existing float64 CPU and native reference checks at the established
  float32 tolerance. Exports for checks occur outside the performance timer.
- Tests cover mask reuse, fresh parameters after every update, all-table
  atomic publication, invalid row/domain checks, UDF/schema invalidation,
  independent snapshots, explicit host replacement and checkpoint resume.
- After initialization, the resident step performs no parameter-,
  moment- or gradient-sized host exports/imports and creates no Arrow
  gradient/update tables. Batch data and optional scalar control/status
  are the remaining routine boundaries. Count both bytes and sync events.
- Report staged timings: current baseline; combined device step; reused
  primal loss; persistent state; compiled program. Measure compilation
  separately. Compare five checked steps and at least 100 steady-state
  updates, including synchronized duration and peak memory. Confirm bounded
  memory and unchanged SQL/gradient semantics before claiming speedups.

If a substantial gap remains after these boundaries are removed, profile
the resulting compiled execution. Candidate work includes backward
contraction scheduling, common expressions, activation liveness and fused
normalization/attention kernels selected from proven plan patterns.
Native MPS parity cannot be predicted from the current measurements.
