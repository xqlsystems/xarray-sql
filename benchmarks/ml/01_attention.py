# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql[ml,duckdb]",
#   "ddxdb[duckdb]",
#   "einfold[datafusion]>=0.1.0rc2",
#   "jax",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Demo 1: causal attention in SQL, on several engines, with its gradient.

Attention, the core of a transformer, is ``softmax(Q·Kᵀ/√d)·V``: each
token's output is an average of the value vectors ``V``, weighted by how
well the token's query in ``Q`` matches each earlier token's key in ``K``.
Here ``Q``, ``K`` and ``V`` are tables of ``(t, d, x)`` (token, feature,
value), registered from xarray Datasets, and attention is two sums over
joins: the scores, a sum of ``Q`` times ``K`` over features, and the
average of ``V`` weighted by ``exp(score)``.

As a SQL engine runs it, this is the naive plan: it stores every one of
the N×N scores. FlashAttention (Dao et al., 2022,
https://arxiv.org/abs/2205.14135) computes the same result without storing
them, by streaming blocks of keys past blocks of queries and keeping a
running softmax. einfold (https://github.com/xqlsystems/einfold), a query
optimizer that rewrites SQL plans into array algorithms, finds that plan
from this SQL, in both of the spellings below, with no rule written for
attention: its rule fuses any softmax-weighted average whose scores are
computed from a join, and the same rule fuses the control below, which is
not attention. This is the first of einfold's three target demos
(``docs/demos.md`` in its repository). The script checks:

* **forward**: each spelling of the model on each *host* (einfold's word
  for an engine that runs a plan; here DataFusion, DataFusion with
  einfold, and DuckDB), checked against the same attention in JAX,
  Google's library for differentiable array programs;
* **gradient**: a loss that reads the model, and its gradient with
  respect to every value of Q, K and V, as one SQL statement. ddx (https://github.com/xqlsystems/ddx),
  which differentiates SQL queries, computes it, and ``jax.grad`` checks
  it;
* **masks are absent rows**: an infinity in V reaches only the queries
  allowed to see it, as SQL's semantics require of any faster plan;
* **a control that is not attention**: the Nadaraya–Watson estimator, a
  Gaussian-kernel weighted average used in regridding. Any rule that
  speeds up attention must speed up this query too;
* **einfold finds the fused plan**: each spelling, and the control, runs
  as one of einfold's fused pipelines.

Run ``uv run benchmarks/ml/01_attention.py --help`` for the options.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import xarray as xr
from _bench import Report, assert_close, cli, hosts
from _hosts import EinFold

import xarray_sql as xql

jax.config.update("jax_enable_x64", True)

HEAD_DIM = 64  # features per head; the scores are scaled by 1/√64 = 1/8.

# The model and its loss are separate, as in a neural network written in
# Python (compare attention() and loss() below). The model is a view, a
# named query: each spelling of attention defines the view ``attention``,
# and the forward pass and the loss read it as if it were a table.

# Attention in einfold's window spelling. The causal mask is a join
# predicate: a masked (i, j) pair is a row that does not exist. Subtracting
# each row's max only keeps exp() from overflowing: it cancels out of the
# ratio w / z, so its gradient sums to zero, and ddx differentiates it
# like any other window.
ATTENTION_WINDOW = """
CREATE OR REPLACE VIEW attention AS
WITH scores AS (
  -- Q·Kᵀ/√64: the dot product of query i and key j over the features d,
  -- for each causal pair, a key j at or before the query i.
  SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s
  FROM q JOIN k ON q.d = k.d
  WHERE q.t >= k.t
  GROUP BY q.t, k.t),
weights AS (
  -- The softmax's numerators, exp(s), shifted by the row's max.
  SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w
  FROM scores),
norm AS (
  -- The softmax's denominator: each query's sum of its weights.
  SELECT i, SUM(w) AS z FROM weights GROUP BY i)
-- For each query, the average of the values, weighted by the softmax w / z.
SELECT w.i AS t, v.d, SUM(w.w / n.z * v.x) AS x
FROM weights w
JOIN v ON w.j = v.t
JOIN norm n ON w.i = n.i
GROUP BY w.i, v.d
"""

# The same attention as ddx spells it: each row's max as a GROUP BY,
# joined back to the scores.
ATTENTION_GROUP_BY = """
CREATE OR REPLACE VIEW attention AS
WITH scores AS (
  -- Q·Kᵀ/√64, for each causal pair, as in the window spelling.
  SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s
  FROM q JOIN k ON q.d = k.d
  WHERE q.t >= k.t
  GROUP BY q.t, k.t),
row_max AS (
  -- Each query's largest score.
  SELECT i, MAX(s) AS m FROM scores GROUP BY i),
weights AS (
  -- The softmax's numerators, exp(s), shifted by the row's max.
  SELECT s.i, s.j, exp(s.s - r.m) AS w
  FROM scores s JOIN row_max r ON s.i = r.i),
norm AS (
  -- The softmax's denominator: each query's sum of its weights.
  SELECT i, SUM(w) AS z FROM weights GROUP BY i)
-- For each query, the average of the values, weighted by the softmax w / z.
SELECT w.i AS t, v.d, SUM(w.w / n.z * v.x) AS x
FROM weights w
JOIN v ON w.j = v.t
JOIN norm n ON w.i = n.i
GROUP BY w.i, v.d
"""

SPELLINGS = {"window": ATTENTION_WINDOW, "group by": ATTENTION_GROUP_BY}

# The forward pass: run the model.
FORWARD = "SELECT t, d, x FROM attention ORDER BY t, d"

# The loss, half the output's squared norm, calls the model by reading the
# view. The SELECT asks for its gradient with respect to every value of Q,
# K and V. The three grad() calls differentiate one loss, so ddx runs one
# backward pass for all three, as jax.grad(loss, argnums=(0, 1, 2)) does.
GRADIENT = """
WITH loss AS (
  SELECT SUM(x * x) / 2 AS l FROM attention)
-- One row per token and feature: the loss's derivative with respect to
-- that entry of Q, of K and of V.
SELECT q.t, q.d,
       grad(loss, q.x) AS dq,
       grad(loss, k.x) AS dk,
       grad(loss, v.x) AS dv
FROM q
JOIN k ON q.t = k.t AND q.d = k.d
JOIN v ON q.t = v.t AND q.d = v.d
ORDER BY q.t, q.d
"""

# The Nadaraya–Watson estimator (E. A. Nadaraya, "On Estimating
# Regression", 1964, https://doi.org/10.1137/1109020; G. S. Watson, "Smooth
# Regression Analysis", 1964): at each target point, the average of the
# sources' y weighted by a Gaussian kernel of their distance (bandwidth
# h = 0.5). Structurally attention: a weighted average with weights
# exp(score).
NADARAYA_WATSON = """
WITH kernel AS (
  -- Each source's weight at each target: exp(-distance² / 2h²).
  SELECT t.p, s.q, exp(-(power(t.x - s.x, 2) + power(t.y - s.y, 2)) / 0.25) AS w
  FROM target t CROSS JOIN source s)
-- Each target's average of the sources' values, weighted by the kernel.
SELECT k.p, SUM(k.w * s.v) / SUM(k.w) AS v
FROM kernel k JOIN source s ON k.q = s.q
GROUP BY k.p
ORDER BY k.p
"""


def attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """The reference: causal softmax attention, as JAX writes it.

    Args:
        q: The queries, an (N, 64) matrix with one row per token.
        k: The keys, shaped like ``q``.
        v: The values, shaped like ``q``.

    Returns:
        Each token's average of the values, shaped like ``q``.
    """
    n = q.shape[0]
    scores = q @ k.T / np.sqrt(HEAD_DIM)
    causal = jnp.tril(jnp.ones((n, n), dtype=bool))
    weights = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1)
    return weights @ v


def loss(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """The reference loss: half the squared norm of attention's output."""
    return jnp.sum(attention(q, k, v) ** 2) / 2


def table(values: np.ndarray) -> xr.Dataset:
    """A (t, d) matrix as the XQL table ``(t, d, x)``."""
    t, d = values.shape
    return xr.Dataset(
        {"x": (("t", "d"), values)},
        coords={"t": np.arange(t), "d": np.arange(d)},
    )


def matrix(rows: pa.Table, name: str = "x") -> np.ndarray:
    """The rows of a ``(t, d, …)`` result as a (t, d) matrix."""
    return xql.to_dataset(rows, dims=["t", "d"])[name].values


def main() -> None:
    parser = cli(__doc__.splitlines()[0], sizes=[64, 256], size_name="tokens")
    args = parser.parse_args()
    report = Report("attention", "tokens", args.reps)
    rng = np.random.default_rng(0)

    for n in args.sizes:
        q, k, v = (rng.normal(size=(n, HEAD_DIM)) for _ in range(3))
        want = attention(q, k, v)
        want_grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

        for host in hosts(args.hosts):
            for name, data in zip("qkv", (q, k, v)):
                host.load(name, table(data))

            for spelling, model in SPELLINGS.items():
                host.execute(model)
                rows = report.time(
                    f"forward ({spelling})", host, n, host.query, FORWARD
                )
                report.check(assert_close("forward", matrix(rows), want))
                report.note("operators", host.operators(FORWARD))

                step = f"gradient ({spelling})"
                if not host.differentiates:
                    report.skip(step, host, n, "no ddx backend")
                    continue
                rows = report.attempt(step, host, n, host.query, GRADIENT)
                if rows is None:
                    continue
                errors = [
                    assert_close(f"d/d{name}", matrix(rows, f"d{name}"), ref)
                    for name, ref in zip("qkv", want_grads)
                ]
                report.check(max(errors))

    check_masks_are_absent_rows(args.hosts)
    check_nadaraya_watson(args.hosts, report)
    check_einfold_fuses(args.hosts)
    report.finish(args.csv)


def check_masks_are_absent_rows(host_names: Sequence[str]) -> None:
    """An infinity in V at token j reaches the queries at or after j, and
    no others: a masked pair is skipped, not multiplied by a zero weight,
    which would turn 0·∞ into NaN. (xarray-sql reads NaN as missing data,
    a NULL that SUM skips, so an infinity is the value that tells.)"""
    n, j = 16, 9
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(n, HEAD_DIM)) for _ in range(3))
    v[j, 0] = np.inf
    print(f"\n  masks: +inf in V at token {j}, feature 0")
    for host in hosts(host_names):
        for name, data in zip("qkv", (q, k, v)):
            host.load(name, table(data))
        for spelling, model in SPELLINGS.items():
            host.execute(model)
            out = matrix(host.query(FORWARD))
            assert np.isfinite(out[:j]).all(), (host.name, spelling)
            assert np.isposinf(out[j:, 0]).all(), (host.name, spelling)
            assert np.isfinite(out[:, 1:]).all(), (host.name, spelling)
            print(f"  {spelling:<9} {host.name:<11} ok")


def check_nadaraya_watson(host_names: Sequence[str], report: Report) -> None:
    """The control: a kernel-weighted average that is not attention."""
    rng = np.random.default_rng(2)
    targets, sources = rng.uniform(size=(64, 2)), rng.uniform(size=(256, 2))
    y = np.sin(4 * sources[:, 0]) + np.cos(3 * sources[:, 1])
    dist2 = ((targets[:, None, :] - sources[None, :, :]) ** 2).sum(-1)
    w = np.exp(-dist2 / 0.25)
    want = (w @ y) / w.sum(axis=1)

    target = xr.Dataset(
        {"x": ("p", targets[:, 0]), "y": ("p", targets[:, 1])},
        coords={"p": np.arange(len(targets))},
    )
    source = xr.Dataset(
        {"x": ("q", sources[:, 0]), "y": ("q", sources[:, 1]), "v": ("q", y)},
        coords={"q": np.arange(len(sources))},
    )
    for host in hosts(host_names):
        host.load("target", target)
        host.load("source", source)
        rows = report.time(
            f"nadaraya-watson ({len(sources)} sources)",
            host,
            len(targets),
            host.query,
            NADARAYA_WATSON,
        )
        got = xql.to_dataset(rows, dims=["p"])["v"].values
        report.check(assert_close("nadaraya-watson", got, want))


def check_einfold_fuses(host_names: Sequence[str]) -> None:
    """einfold runs each spelling of attention, and the control, as one
    fused pipeline, computed tile by tile with a running softmax, rather
    than as the joins and aggregates the SQL describes."""
    if EinFold.name not in host_names:
        return
    print("\n  einfold's plans")
    host = EinFold()
    rng = np.random.default_rng(3)
    for name in "qkv":
        host.load(name, table(rng.normal(size=(128, HEAD_DIM))))
    for spelling, model in SPELLINGS.items():
        host.execute(model)
        before = host.einfold.plans_with_pipelines()
        host.query(FORWARD)
        assert host.einfold.plans_with_pipelines() > before, spelling
        print(f"  {spelling:<16} fused")
    targets, sources = rng.uniform(size=(16, 2)), rng.uniform(size=(64, 2))
    host.load(
        "target",
        xr.Dataset(
            {"x": ("p", targets[:, 0]), "y": ("p", targets[:, 1])},
            coords={"p": np.arange(len(targets))},
        ),
    )
    host.load(
        "source",
        xr.Dataset(
            {
                "x": ("q", sources[:, 0]),
                "y": ("q", sources[:, 1]),
                "v": ("q", sources.sum(axis=1)),
            },
            coords={"q": np.arange(len(sources))},
        ),
    )
    before = host.einfold.plans_with_pipelines()
    host.query(NADARAYA_WATSON)
    assert host.einfold.plans_with_pipelines() > before, "nadaraya-watson"
    print(f"  {'nadaraya-watson':<16} fused")


if __name__ == "__main__":
    main()
