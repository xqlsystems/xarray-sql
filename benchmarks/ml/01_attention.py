# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql[ml,duckdb]",
#   "jax",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Demo 1: attention in SQL, the query einfold should turn into FlashAttention.

Attention, the core of a transformer, is ``softmax(Q·Kᵀ/√d)·V``. In the
XQL model, ``Q``, ``K`` and ``V`` are tables of ``(t, d, x)`` (token,
feature, value), and attention is two folds over joins: an einsum for the
scores, and a weighted average of ``V`` with weights ``exp(score)``.

The SQL below is the naive plan, which stores every one of the N×N
scores. FlashAttention never does: it streams tiles of keys past tiles of
queries, with an online softmax. einfold's first target is to find that
plan from this SQL, in both of its spellings, with no attention-specific
rule (einfold's ``docs/demos.md``, Demo 1). This script is the baseline
that plan must match and beat:

* **forward**: each spelling on each host, against a JAX reference;
* **gradient**: ddx differentiates the same SQL with respect to Q, K
  and V, against ``jax.grad``;
* **masks are absent rows**: an infinity in V reaches only the queries
  allowed to see it, as SQL's semantics require of any faster plan;
* **a control that is not attention**: the Nadaraya–Watson estimator, a
  Gaussian-kernel weighted average used in regridding. The rules that
  fuse attention must fuse it too.

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

# The scores, softmax and weighted average, in einfold's window spelling.
# The causal mask is a join predicate: a masked (i, j) pair is a row that
# does not exist. The max only keeps exp() from overflowing and cancels
# out of the ratio, so it carries no gradient (ddx_stop_gradient).
ATTENTION_WINDOW = """
scores AS (
  SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s
  FROM q JOIN k ON q.d = k.d
  WHERE q.t >= k.t
  GROUP BY q.t, k.t),
weights AS (
  SELECT i, j, exp(s - ddx_stop_gradient(MAX(s) OVER (PARTITION BY i))) AS w
  FROM scores),
norm AS (
  SELECT i, SUM(w) AS z FROM weights GROUP BY i),
attention AS (
  SELECT w.i AS t, v.d, SUM(w.w / n.z * v.x) AS x
  FROM weights w
  JOIN v ON w.j = v.t
  JOIN norm n ON w.i = n.i
  GROUP BY w.i, v.d)
"""

# The same attention as ddx spells it: the row max as a GROUP BY, joined
# back to the scores.
ATTENTION_GROUP_BY = """
scores AS (
  SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s
  FROM q JOIN k ON q.d = k.d
  WHERE q.t >= k.t
  GROUP BY q.t, k.t),
row_max AS (
  SELECT i, ddx_stop_gradient(MAX(s)) AS m FROM scores GROUP BY i),
weights AS (
  SELECT s.i, s.j, exp(s.s - r.m) AS w
  FROM scores s JOIN row_max r ON s.i = r.i),
norm AS (
  SELECT i, SUM(w) AS z FROM weights GROUP BY i),
attention AS (
  SELECT w.i AS t, v.d, SUM(w.w / n.z * v.x) AS x
  FROM weights w
  JOIN v ON w.j = v.t
  JOIN norm n ON w.i = n.i
  GROUP BY w.i, v.d)
"""

SPELLINGS = {"window": ATTENTION_WINDOW, "group by": ATTENTION_GROUP_BY}

# A scalar of the output to differentiate: half its squared norm.
LOSS = "loss AS (SELECT SUM(x * x) / 2 AS l FROM attention)"

# Nadaraya–Watson: at each target point, the average of the sources' y
# weighted by a Gaussian kernel of their distance (bandwidth h = 0.5).
# Structurally attention: a weighted average with weights exp(score).
NADARAYA_WATSON = """
kernel AS (
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

            for spelling, ctes in SPELLINGS.items():
                forward = f"WITH {ctes} SELECT * FROM attention ORDER BY t, d"
                rows = report.time(
                    f"forward ({spelling})", host, n, host.query, forward
                )
                report.check(assert_close("forward", matrix(rows), want))
                report.note("operators", host.operators(forward))

                step = f"gradient ({spelling})"
                if not host.differentiates:
                    report.skip(step, host, n, "no ddx backend")
                    continue
                # One statement per input, sharing one backward pass.
                gradients = [
                    f"WITH {ctes}, {LOSS} "
                    f"SELECT t, d, grad(loss, x) AS x FROM {name} ORDER BY t, d"
                    for name in "qkv"
                ]
                grads = report.time(step, host, n, host.query_all, gradients)
                errors = [
                    assert_close(f"d/d{name}", matrix(got), ref)
                    for name, got, ref in zip("qkv", grads, want_grads)
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
        for spelling, ctes in SPELLINGS.items():
            out = matrix(
                host.query(f"WITH {ctes} SELECT * FROM attention ORDER BY t, d")
            )
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
            f"WITH {NADARAYA_WATSON}",
        )
        got = xql.to_dataset(rows, dims=["p"])["v"].values
        report.check(assert_close("nadaraya-watson", got, want))


if __name__ == "__main__":
    main()
