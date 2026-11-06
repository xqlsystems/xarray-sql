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
        """SELECT 'grad(x)' AS s, $v AS "jvp(v)" -- vjp(y)""",
        param_values={"v": 1},
    ).to_pylist()
    assert rows == [{"s": "grad(x)", "jvp(v)": 1}]


def test_context_prepare_runs_gradient_steps_and_reads_writes(problem):
    ctx, x, w, y = problem
    step = ctx.prepare(
        f"WITH {LOSS} SELECT feature, grad(loss, w) AS w FROM weights ORDER BY feature"
    )
    old = step()
    original = old["w"].to_numpy().copy()
    replacement = xr.Dataset(
        {"w": ("feature", w.copy() * 2)}, coords={"feature": np.arange(3)}
    )
    got = step(weights=replacement)
    np.testing.assert_allclose(
        got["w"], 2 * x.T @ (x @ replacement.w.values - y)
    )
    np.testing.assert_array_equal(old["w"], original)
    with pytest.raises(ValueError, match="coordinates"):
        step(weights=replacement.assign_coords(feature=[3, 4, 5]))


def test_prepared_gradients_read_new_values_and_keep_old_results(problem):
    ctx, x, w, y = problem
    statements = [
        f"WITH {LOSS} SELECT feature, grad(loss, w) AS w FROM weights ORDER BY feature",
        f"WITH {LOSS} SELECT feature, w - 0.01 * grad(loss, w) AS w FROM weights ORDER BY feature",
    ]
    run = ad.prepare(ctx, statements)
    [first, _] = run()
    original = first["w"].to_numpy().copy()
    for scale in (0.5, -1.0, 2.0):
        values = xr.Dataset(
            {"w": ("feature", w.copy() * scale)},
            coords={"feature": np.arange(3)},
        )
        gradient, step = run(weights=values)
        want = 2 * x.T @ (x @ values.w.values - y)
        np.testing.assert_allclose(gradient["w"], want)
        np.testing.assert_allclose(step["w"], values.w.values - 0.01 * want)
        np.testing.assert_array_equal(first["w"], original)
    # Regular ddx preparation resolves a replacement table by name.
    ctx.deregister_table("weights")
    ctx.from_dataset("weights", values * 2)
    [gradient, _] = run()
    np.testing.assert_allclose(
        gradient["w"], 2 * x.T @ (x @ (values.w.values * 2) - y)
    )


def test_prepared_statements_require_a_sequence(problem):
    ctx, *_ = problem
    with pytest.raises(TypeError, match="list"):
        ad.prepare(ctx, "SELECT 1")
