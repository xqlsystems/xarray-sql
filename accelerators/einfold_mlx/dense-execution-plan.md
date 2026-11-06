# Dense execution of SQL training programs

Implementation and validation results are available in the
[dense execution report](../../benchmarks/apple/dense-execution.md).
The local executor now retains compact forward/backward arrays, reuses
structural plans, runs the supported program on MLX, preserves the original
ddx differentiation, and has completed checked five-step GPU training at
context 1024. Default APIs still return Arrow snapshots. An explicit API
for persistent GPU-owned training state remains future work; long-context
CPU cache sensitivity and the dense AdamW refusal are reported as limits.

Research and execution plan, November 1, 2026. The requested execution
threshold is 06:00 America/Los_Angeles (14:00 UTC); research began before
it, and implementation may proceed now that this time has passed.

## Evidence and objective

The current SQL → DataFusion → EinFold → MLX bridge accelerates individual
matrix and attention kernels, but retains relational intermediates. The
123,563,520-parameter model (12 layers, width 768, 12 heads, vocabulary
50257, batch 1, context 16) takes 163.65 s per warm SQL GPU step. Across
five steps, counters record 131 s planning, 169 s stage storage and 504 s
execution, with 7,148 physical plans built. Native JAX CPU takes 0.334 s
eager, 0.394 s with separate loss/backward calls and 0.136 s compiled;
upstream nanoGPT's PyTorch model takes 0.095 s CPU float64 and 0.024 s MPS
float32. All parameter updates were checked. These are short-context
training results; the existing context-1024 comparison is forward-only.

See [SQL measurements](../../benchmarks/apple/nanogpt-shared.md),
[native measurements](../../benchmarks/apple/nanogpt-frameworks.md) and
their linked raw records. They are immutable baselines for this work.

The objective is to execute validated tensor-shaped regions of the actual
SQL and ddx programs on compact arrays, keeping intermediates dense and
avoiding per-operator Arrow reconstruction. ddx remains the source of
derivatives. The implementation must work on plans and axis relationships,
without recognizing a benchmark's name or substituting a handwritten GPT.

## Research findings

[EinFold](https://github.com/xqlsystems/einfold) is a tensor optimizer for
SQL. Its [design](https://github.com/xqlsystems/einfold/blob/main/docs/design.md)
already describes exact shape/layout facts in a side table, separate
symbolic and numerical passes, and optimization of a whole program of
plans. Its current DataFusion operators still collect Arrow inputs and
return relational Arrow rows. This supports adding a dense program path
without changing the SQL surface, while device-specific execution remains
in the host integration. A design aspiration is not an implemented API.

[ddx](https://github.com/xqlsystems/ddx) differentiates SQL. Its
[program representation](https://github.com/xqlsystems/ddx/blob/main/crates/ddx-ad/src/program.rs)
contains input tables, validation checks, forward steps, backward steps
and output tables. Each step has a name and an unbound Substrait plan;
reads can be bound to schemas. The Python backend currently exposes
per-step `compile`, `materialize_compiled` and existence checks. Those
hooks permit an incremental dense backend. A whole-program preparation
hook is a possible later extension, not a precondition for useful progress.

[DataFusion](https://datafusion.apache.org/library-user-guide/extending-operators.html)
supports custom logical/physical nodes through optimizer rules. A dense
region can therefore be represented internally while preserving its
ordinary relational input/output contract. One physical region should
contain many array operations, rather than returning scalar Arrow rows
between every pair of operators.

[MLX](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
is Apple's array framework for unified CPU/GPU memory. The important costs
here are allocation, packing, conversion and synchronization; CPU/GPU
crossings on this machine are not necessarily discrete-device PCIe copies.
[Its conversion documentation](https://ml-explore.github.io/mlx/build/html/usage/numpy.html)
states that `mx.array` copies, and float64 NumPy input converts to float32.
Buffer/DLPack interop has ownership and synchronization requirements.
Unified memory alone does not make our current bridge zero-copy.

[MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
can eliminate common work and fuse operations. Shape/type/input changes
can cause recompilation. Build functions over explicit array arguments;
avoid capturing mutable parameter values in cached closures. Defer
evaluation to region/output boundaries, with explicit synchronization for
timing. Execute ddx's derivative expressions; do not use MLX autodiff to
replace them. Keep `MLX_ENABLE_TF32=0` as required by the
[precision documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html).

[Arrow's fixed-shape tensor type](https://arrow.apache.org/docs/format/CanonicalExtensions.html#fixed-shape-tensor)
is useful for tensor interchange but changes a scalar field into a tensor
field. It cannot silently replace a public SQL table of scalar values.
Use a private dense representation and reconstruct the ordinary rows at
real relational/output boundaries. The
[Arrow C interface](https://arrow.apache.org/docs/format/CDataInterface.html)
can share owned Arrow buffers across runtimes, but does not prove tensor
shape, uniqueness, absence of NULLs, or MLX compatibility.

## Recommended design

Use a program-scoped dense store and a small array-operation graph.
Preserve existing public SQL/Arrow schemas and a relational fallback.

### Compact values and exact domains

A dense value contains its array/buffer, ordered axes, dtype, shape,
strides/view mapping and lifetime owner. Store coordinate dictionaries
once per axis/domain, rather than once per tensor element. Column lineage
maps SQL fields to axis coordinates, indexed gathers or scalar array
expressions. Attach exact facts to the plan node where they are valid.

Initial eligibility is deliberately narrow: verified rectangular domains,
unique keys and supported finite numerical inputs. Arbitrary coordinate
labels may use exact dictionaries. Unsupported duplicate, NULL, sparse or
value-dependent domains retain relational execution until their handling
is proved. Shape is insufficient evidence of completeness or uniqueness.

Cache structural proofs under coordinate/schema epochs. Refresh numerical
values under separate value epochs. Ordinary mutable NumPy/Arrow inputs
are imported once per call, so direct external writes remain visible.
Cross-call value caching is allowed only for explicitly owned, versioned
state with a defined write path. Do not infer that an untracked external
array has stayed unchanged because its schema or pointer is unchanged.

### Dense lowering of backward subgraphs

Lower a closed set of verified operations: reads, aliases, projections,
views, elementwise arithmetic/CASE, broadcasts, gathers, reductions,
contractions and required domain-fill operations. Use field lineage and
axis relationships from actual DataFusion/Substrait plans. Avoid SQL text
regexes and neural-network-specific substitutions.

Recognize shared-axis join plus grouped product as a contraction before
forming its Cartesian products. For example, the tied embedding gradient
has the form `X.T @ dlogits`: context 16, width 768 and vocabulary 50257
represent 617,558,016 possible product contributions, but only a
38,597,376-element output. Retain that matrix formulation through normalized
and residual operands. Extend existing EinFold fold/chain detection where
that removes a dominant missed pattern; use a general dense operation
graph for the surrounding maps and reductions.

Priority patterns are reconstructed layer-normalized operands, residual
joins, null-safe key joins over proven nonnullable axes, gradient domain
left joins with defaults, gather/scatter contributions for embeddings,
and sums of many reduced gradient contributions. A UNION is a bag union,
not elementwise addition unless a following reduction and domain proof
establish that equivalence. Group existence and all-NULL SUM behavior must
be retained. Masks describe absent rows, with no multiplication of masked
NaN/Inf values by zero.

### Stable bindings and compiled programs

Bind intermediate reads to stable logical slots rather than new physical
table providers on each run. Each slot has an immutable structural
descriptor and a current value generation. The graph cache key includes
schema, coordinate/domain facts, function semantics/settings, dtype and
precision mode. Changing only parameter values updates bindings; changed
domains or semantics invalidate affected regions.

Start through ddx's existing backend hooks: compiled steps may hold dense
plans and `materialize_compiled` may store a lazy internal value. Named
read schemas must remain real and fallback scans must materialize current
rows on demand. Existence checks can use exact domain facts only when
their result is proved; otherwise evaluate the check or cross a relational
boundary. Public methods returning Arrow still return complete Arrow
snapshots.

If per-step hooks prevent adequate fusion, add an optional ddx
whole-program compilation interface using its existing program data.
Existing backends keep their current behavior. Preserve validation checks,
typed refusals and the ordering of externally visible effects.

### CPU first, then resident MLX regions

Implement the dense path on NumPy float64 first to distinguish relational
lowering problems from GPU precision/device behavior. Then execute the
same operation graph with MLX arrays in explicitly permitted float32 mode.
Keep maximal supported regions as arrays; convert at imports, unsupported
relational boundaries and requested public outputs. Avoid `np.asarray`,
`.item()` and `mx.eval` inside each individual numerical operator.

The current slice-based native backend API forces packing into f64 vectors
and returning f64 vectors per call. The new path needs an owned array/graph
interface with stable handles and views, while retaining the existing
callback as a fallback. Keep EinFold's IR hardware-neutral and MLX objects
in the host bridge. Profile before adding custom C++/Metal kernels.

Only after the compatible Arrow-output path is correct, consider an
explicit resident training-state API. It must state who owns parameters,
how writes are versioned and when host snapshots are produced. Existing
`Context.prepare`/`SharedExecutor` APIs must not silently change their
return types or ignore external writes. Compare resident compute and
Arrow-materializing API time separately.

### Memory and fallback

Use last-use analysis to release intermediate arrays and accumulate
gradient contributions as they become available. Bound allocations using
known shapes, include conversion/output costs and avoid retaining every
branch's large result. MLX's lazy graph must also be bounded and evaluated
at deliberate points rather than growing across unlimited training steps.

Failed runtime guards must fall back before committing parameter updates.
Retain an unchanged input snapshot or otherwise transactional ownership
until guards pass. An unsupported region is a normal relational boundary,
not permission to change NULL/duplicate/default semantics. Use float64 CPU
as the default exact-precision path. Do not weaken tolerances or silently
alter AdamW to hide the existing float32 parity failure.

## Execution sequence

1. **Expose the actual bottlenecks.** Add opt-in per-stage timings and
   dense/fallback reasons, row/byte counts, coordinate packing costs,
   compiled-plan counts and synchronization/conversion counters. Profile
   one representative backward pass, not repeated five-step large runs.
   Save top offenders and minimal SQL reproductions. Native opaque nodes
   require internal metrics; their outer node count is not an operator
   count comparable with plain DataFusion.
2. **Build dense storage and lowering on CPU.** Add exact domains, compact
   arrays and validated operator lowering. First cover ordinary matrix
   gradients and reconstructed normalized/residual operands. Route those
   through actual SQL/ddx plans. Keep unsupported operators on fallback.
   Test different names, shapes and coordinate labels to prevent accidental
   model-specific behavior.
3. **Integrate stable staged execution.** Connect dense values to ddx's
   compiled backend hooks and stable slots. Share intermediates and
   eliminate per-step replanning for unchanged shapes. Verify repeated
   updates, domain changes, empty/NULL results and snapshots. Preserve
   dynamic-filter safeguards and fresh join state for relational regions.
4. **Execute dense regions with MLX.** Compile supported graphs over
   explicit parameters, retain arrays between their operators, evaluate
   at required boundaries and record all copies/casts/synchronizations.
   Add explicit owned resident state only after the compatible path works.
5. **Expand supported backward patterns.** Use the measured remaining
   fallback list to add domain fill, scatter/reduction, nonlinear maps and
   broader contraction recognition. Recheck memory liveness and accuracy
   after each coherent change; avoid an unbounded optimizer rewrite search.
6. **Validate and benchmark.** Run focused public-result tests, then the
   full local suite and native tests appropriate to changed crates. Start
   with tiny/213K five-step cases and normalized-gradient reproductions,
   then 10.7M and 123.6M context-16 five-step comparisons. Attempt larger
   contexts progressively only after short-context correctness and memory
   are established. Run hosts serially without builds/profilers during
   timings. Compare unchanged JAX/PyTorch controls; keep failed/bounded runs
   out of speedup tables. Include DuckDB controls where bounded runs
   complete. Write measured speedups and remaining limitations.

These are ordered milestones, not permission to stop after a fast kernel
microbenchmark. Adapt implementation choices to measured blockers while
preserving the scope and correctness requirements.

## Completion criteria

- SQL and ddx remain the source of the computation and derivatives.
- Supported dense backward paths avoid product-sized relational
  intermediates and per-element coordinate storage between their operators.
- Steady-state structural planning is independent of training-step count
  for unchanged domains; value changes remain visible.
- GPU work is verified by execution counters. No per-operator host
  materialization/synchronization remains inside a supported dense region.
- Tests cover values, dimensions, coordinates, attributes, bags, NULLs,
  empty groups, defaults, masked exceptional values, grad/VJP/JVP, typed
  refusals, re-registration and repeated updates. Existing tolerances stay.
- Reports include cold/compile/setup and warm times, import/export costs,
  memory, actual device work, fallback coverage and all five checked steps.

An initial engineering target is at least a 10× reduction of the current
123M warm-step time, followed by reducing the remaining gap to native
array execution. This is a target, not a measured or promised speedup.
Separate compatible Arrow API costs from optional resident-state costs.

## Resume details

Work directly in `/Users/cromwellian/xql/xarray-sql`, preserving existing
uncommitted work. Read its `AGENTS.md`; when modifying the EinFold patch,
read `/private/tmp/xql-einfold/AGENTS.md` and the relevant design sections.
The user explicitly authorizes local implementation of this plan after
06:00 Pacific. No publishing, merging or messages to maintainers are
requested. Keep design changes and the reproducible local patch together.

The checkout uses `.venv` Python 3.12.14, DataFusion Python 55.0.0,
ddxdb 0.4.6 and custom EinFold `0.2.0+xqlmlx.3`, pinned to
`39b25210d3b39027ef91584383fb114dfa8a4a48`. Native Rust DataFusion is 55.2.0.
Use `PYTHONPATH=$PWD`; installed Python sources can differ. Do not run
`uv sync`, which can replace the custom wheel. Metal access requires the
appropriate execution permissions. Use `MLX_ENABLE_TF32=0` before import.

EinFold sources/build cache: `/private/tmp/xql-einfold`; clean pinned tree:
`/private/tmp/xql-einfold-clean`. Isolated Rust toolchain:
`PATH=/private/tmp/xql-cargo/bin:$PATH`,
`RUSTUP_HOME=/private/tmp/xql-rustup`,
`CARGO_HOME=/private/tmp/xql-cargo`. Build EinFold from its `python`
directory with maturin, locked dependencies and the checkout's Python.
Retain all new native changes in the EinFold repository and verify that the pinned build
applies cleanly to the pinned tree. Existing xarray-sql native builds must
retain Flight support. No build processes remain paused.

The upstream nanoGPT model and isolated PyTorch 2.15.0 environment are in
`/private/tmp/xql-upstream-nanogpt` and `/private/tmp/xql-framework-venv`.
The latter needs the checkout's `.venv/lib/python3.12/site-packages` on
`PYTHONPATH` for JAX/NumPy. Baseline scripts are
`benchmarks/apple/run_nanogpt_sizes.py` and
`benchmarks/apple/compare_nanogpt_frameworks.py`. The last full local suite
passed 637 tests (277 skipped, 102 integration deselected); native suites
passed 60, with four upstream ignored tests and one unchanged plain-
DataFusion ARM NaN-sign assumption excluded. Keep those qualifications.
