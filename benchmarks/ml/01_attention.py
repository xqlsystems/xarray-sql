# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql[ml,duckdb]",
#   "ddxdb[duckdb]",
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

This SQL is the naive plan: it stores every one of the N×N scores.
FlashAttention (Dao et al., 2022, https://arxiv.org/abs/2205.14135)
computes the same result without storing them, by streaming blocks of
keys past blocks of queries and keeping a running softmax. einfold
(https://github.com/xqlsystems/einfold), a query optimizer that rewrites
SQL plans into array algorithms like this one, aims to find that plan
from this SQL, in both of the spellings below, with no rule written for
attention. It is the first of einfold's three target demos
(``docs/demos.md`` in its repository). This script is the baseline that
plan must match and then beat:

* **forward**: each spelling of the model on each *host* (einfold's word for an engine
  that runs a plan; here DataFusion and DuckDB), checked against the same
  attention in JAX, Google's library for differentiable array programs;
* **gradient**: a loss that reads the model, and its gradient with
  respect to every value of Q, K and V, as one SQL statement. ddx (https://github.com/xqlsystems/ddx),
  which differentiates SQL queries, computes it, and ``jax.grad`` checks
  it;
* **masks are absent rows**: an infinity in V reaches only the queries
  allowed to see it, as SQL's semantics require of any faster plan;
* **a control that is not attention**: the Nadaraya–Watson estimator, a
  Gaussian-kernel weighted average used in regridding. Any rule that
  speeds up attention must speed up this query too.

Run ``uv run benchmarks/ml/01_attention.py --help`` for the options.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from _bench import Report, assert_close, cli, hosts

import xarray_sql as xql

jax.config.update("jax_enable_x64", True)

D = 64  # features per head; the scores are scaled by 1/√64 = 1/8.

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
  SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s
  FROM q JOIN k ON q.d = k.d
  WHERE q.t >= k.t
  GROUP BY q.t, k.t),
weights AS (
  SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w
  FROM scores),
norm AS (
  SELECT i, SUM(w) AS z FROM weights GROUP BY i)
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
  SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s
  FROM q JOIN k ON q.d = k.d
  WHERE q.t >= k.t
  GROUP BY q.t, k.t),
row_max AS (
  SELECT i, MAX(s) AS m FROM scores GROUP BY i),
weights AS (
  SELECT s.i, s.j, exp(s.s - r.m) AS w
  FROM scores s JOIN row_max r ON s.i = r.i),
norm AS (
  SELECT i, SUM(w) AS z FROM weights GROUP BY i)
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
SELECT q.t, q.d,
       grad(loss, q.x) AS dq,
       grad(loss, k.x) AS dk,
       grad(loss, v.x) AS dv
FROM q
JOIN k ON q.t = k.t AND q.d = k.d
JOIN v ON q.t = v.t AND q.d = v.d
ORDER BY q.t, q.d
"""

# Nadaraya–Watson: at each target point, the average of the sources' y
# weighted by a Gaussian kernel of their distance (bandwidth h = 0.5).
# Structurally attention: a weighted average with weights exp(score).
NADARAYA_WATSON = """
WITH kernel AS (
  SELECT t.p, s.q, exp(-(power(t.x - s.x, 2) + power(t.y - s.y, 2)) / 0.25) AS w
  FROM target t CROSS JOIN source s)
SELECT k.p, SUM(k.w * s.v) / SUM(k.w) AS v
FROM kernel k JOIN source s ON k.q = s.q
GROUP BY k.p
ORDER BY k.p
"""


def attention(q, k, v):
    """The reference: causal softmax attention, as JAX writes it."""
    n = q.shape[0]
    scores = q @ k.T / np.sqrt(D)
    causal = jnp.tril(jnp.ones((n, n), dtype=bool))
    weights = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1)
    return weights @ v


def loss(q, k, v):
    return jnp.sum(attention(q, k, v) ** 2) / 2


def table(values: np.ndarray) -> xr.Dataset:
    """A (t, d) matrix as the XQL table ``(t, d, x)``."""
    t, d = values.shape
    return xr.Dataset(
        {"x": (("t", "d"), values)},
        coords={"t": np.arange(t), "d": np.arange(d)},
    )


def matrix(rows, name: str = "x") -> np.ndarray:
    """The rows of a ``(t, d, …)`` result as a (t, d) matrix."""
    return xql.to_dataset(rows, dims=["t", "d"])[name].values


def main():
    parser = cli(__doc__.splitlines()[0], sizes=[64, 256], size_name="tokens")
    args = parser.parse_args()
    report = Report("attention", "tokens", args.reps)
    rng = np.random.default_rng(0)

    for n in args.sizes:
        q, k, v = (rng.normal(size=(n, D)) for _ in range(3))
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
                known = {}
                if spelling == "window":
                    # TODO(xqlsystems/ddx#204): drop once ddx differentiates
                    # a view whose CTE computes a window on DuckDB.
                    known = {"duckdb": "Not implemented Error: CTE"}
                rows = report.attempt(
                    step, host, n, host.query, GRADIENT, known=known
                )
                if rows is None:
                    continue
                errors = [
                    assert_close(f"d/d{name}", matrix(rows, f"d{name}"), ref)
                    for name, ref in zip("qkv", want_grads)
                ]
                report.check(max(errors))

    check_masks_are_absent_rows(args.hosts)
    check_nadaraya_watson(args.hosts, report)
    report.finish(args.csv)


def check_masks_are_absent_rows(host_names):
    """An infinity in V at token j reaches the queries at or after j, and
    no others: a masked pair is skipped, not multiplied by a zero weight,
    which would turn 0·∞ into NaN. (xarray-sql reads NaN as missing data,
    a NULL that SUM skips, so an infinity is the value that tells.)"""
    n, j = 16, 9
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(n, D)) for _ in range(3))
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


def check_nadaraya_watson(host_names, report):
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


if __name__ == "__main__":
    main()
