# Exact GELU for SQL training

[ddx](https://github.com/xqlsystems/ddx) differentiates SQL query plans.
Its released scalar rules do not include the error function, `erf`, used
by [nanoGPT](https://github.com/karpathy/nanoGPT)'s exact GELU activation.
The pinned feature commit adds `erf'(x) = 2/sqrt(pi) * exp(-x*x)` to the scalar
registry and retains `erf` in derivative plans. The rule composes with
the chain rule and higher derivatives; it does not substitute a model's
handwritten backward pass.

The builder uses the [ddx feature fork](https://github.com/cromwellian/ddx), pinned to commit
`a941d46195632c9385a37e9812dd45dc10a50e94`, based on release commit
`af71683f6678bb093e600f948c212490a4d8b74b`. Historical benchmark wheels used
`ddxdb==0.4.6+xqlmath.3`; the contribution retains upstream version 0.4.6.
The feature commit also exposes
`Prepared.values`: metadata for the primal outputs computed by prepared
derivative programs. These outputs can be read during a run before its
temporary tables are released. This lets the GPU training session reuse
the actual loss from backward preparation instead of executing a separate
forward query. It does not alter existing derivative rules.

Engines must
register an `erf(DOUBLE)` implementation; `xarray_sql.math.register_erf`
does this for DataFusion and marks its equivalent array primitive as safe
for EinFold CPU and MLX GPU execution. Replacing the SQL UDF removes that
permission. This contribution does not add an erf implementation to DuckDB.

From the xarray-sql directory, after installing the normal ML dependencies
and the [compact EinFold executor](../einfold_mlx/README.md):

```bash
source .venv/bin/activate
uv pip install cmake maturin
python accelerators/ddx_math/build.py
```

Requires git, a Rust toolchain, CMake and uv on PATH. The builder clones
an isolated checkout, validates the pinned feature revision, builds its
locked wheel and installs it in the active virtual environment. No patch
is applied. Use `--source /path/to/ddx` for a clean checkout already at the pin. `uv sync` may
replace locally built wheels; rerun the optional builders if necessary.

The public SQL numerical checks are in `tests/test_math.py`. The ddx contribution
also adds a Rust check of the chain rule and second derivative.
