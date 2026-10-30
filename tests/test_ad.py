"""Tests for the ddx gradient extension (`xarray_sql.ad`)."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("ddxdb")

from ddxdb import UnsupportedExpression

from xarray_sql import Context, ad

# Least squares: pred = x @ w, loss = sum((pred - y)^2).
LOSS = """
pred AS (
  SELECT x.sample, SUM(x.x * w.w) AS p
  FROM data x JOIN weights w ON x.feature = w.feature
  GROUP BY x.sample),
loss AS (
  SELECT SUM(power(pred.p - y.y, 2)) AS l
  FROM pred JOIN targets y ON pred.sample = y.sample)
"""


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 3))
    w = rng.normal(size=3)
    y = rng.normal(size=6)
    sample, feature = np.arange(6), np.arange(3)
    ctx = Context()
    ctx.from_dataset(
        "data",
        xr.Dataset(
            {"x": (("sample", "feature"), x)},
            coords={"sample": sample, "feature": feature},
        ),
    )
    ctx.from_dataset(
        "weights",
        xr.Dataset({"w": ("feature", w)}, coords={"feature": feature}),
    )
    ctx.from_dataset(
        "targets", xr.Dataset({"y": ("sample", y)}, coords={"sample": sample})
    )
    return ctx, x, w, y


def test_grad_matches_closed_form(problem):
    ctx, x, w, y = problem
    got = ctx.sql(
        f"WITH {LOSS} SELECT feature, grad(loss, w) AS w FROM weights ORDER BY feature"
    ).to_dataset()
    np.testing.assert_allclose(got["w"].values, 2 * x.T @ (x @ w - y))


def test_gradient_step_reads_like_sql(problem):
    ctx, x, w, y = problem
    got = ctx.sql(
        f"WITH {LOSS} "
        "SELECT feature, w - 0.01 * grad(loss, w) AS w FROM weights "
        "ORDER BY feature"
    ).to_pandas()
    np.testing.assert_allclose(got["w"], w - 0.01 * 2 * x.T @ (x @ w - y))


def test_sql_all_shares_one_loss(problem):
    ctx, x, w, y = problem
    grad = 2 * x.T @ (x @ w - y)
    select = "SELECT feature, {} AS w FROM weights ORDER BY feature"
    gradient, step = ad.sql_all(
        ctx,
        [
            f"WITH {LOSS} " + select.format("grad(loss, w)"),
            f"WITH {LOSS} " + select.format("w - 0.01 * grad(loss, w)"),
        ],
    )
    np.testing.assert_allclose(gradient.to_dataset()["w"].values, grad)
    np.testing.assert_allclose(step.to_dataset()["w"].values, w - 0.01 * grad)


def test_vjp_matches_closed_form(problem):
    # Pulled back along the targets, pred = x @ w gives x.T @ y.
    ctx, x, _, y = problem
    got = ctx.sql(
        f"WITH {LOSS}, cotangent AS (SELECT sample, y AS p FROM targets) "
        "SELECT feature, vjp(pred, w, cotangent) AS w FROM weights "
        "ORDER BY feature"
    ).to_dataset()
    np.testing.assert_allclose(got["w"].values, x.T @ y)


def test_jvp_matches_closed_form(problem):
    # Pushed forward along t, pred = x @ w gives x @ t.
    ctx, x, _, _ = problem
    got = ctx.sql(
        "WITH tangent AS ("
        "  SELECT feature, CAST(feature AS DOUBLE) AS w FROM weights) "
        "SELECT x.sample, jvp(SUM(x.x * weights.w), weights.w, tangent) AS p "
        "FROM data x JOIN weights ON x.feature = weights.feature "
        "GROUP BY x.sample ORDER BY sample"
    ).to_dataset()
    np.testing.assert_allclose(got["p"].values, x @ np.arange(3))


def test_stop_gradient_blocks_the_derivative(problem):
    ctx, _, w, _ = problem
    got = ctx.sql(
        "WITH loss AS (SELECT SUM(w * ddx_stop_gradient(w)) AS l "
        "FROM weights) "
        "SELECT feature, grad(loss, w) AS w FROM weights ORDER BY feature"
    ).to_dataset()
    np.testing.assert_allclose(got["w"].values, w)


def test_unsupported_expression_raises(problem):
    # The loss groups by a value of w, and then reads it.
    ctx, *_ = problem
    with pytest.raises(UnsupportedExpression, match="GROUP BY key"):
        ctx.sql(
            "WITH buckets AS ("
            "  SELECT ROUND(w) AS k, SUM(w * w) AS s FROM weights"
            "  GROUP BY ROUND(w)), "
            "loss AS (SELECT SUM(k * s) AS l FROM buckets) "
            "SELECT feature, grad(loss, w) AS g FROM weights"
        ).collect()


def test_a_mention_in_a_literal_is_not_a_call():
    ctx = Context()
    rows = ctx.sql(
        "SELECT 'grad(x)' AS s, $v AS v -- vjp(y)", param_values={"v": 1}
    ).to_pylist()
    assert rows == [{"s": "grad(x)", "v": 1}]
