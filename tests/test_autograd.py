"""Tests for the SQL autograd surface: ``SELECT grad(expr, column) ...``.

These exercise the full path — XarrayContext.sql() -> Substrait -> native
grad_rewrite -> Substrait -> execute — and compare results against analytic
derivatives computed with numpy.
"""

import numpy as np
import pytest
import xarray as xr

import xarray_sql as xql


@pytest.fixture
def ctx():
    val = np.linspace(0.1, 3.0, 16)
    ds = xr.Dataset(
        {"val": (("i",), val)},
        coords={"i": np.arange(16)},
    )
    context = xql.XarrayContext()
    context.from_dataset("t", ds, chunks={"i": 5})
    return context


@pytest.fixture
def ctx_xy():
    rng = np.random.default_rng(0)
    n = 16
    ds = xr.Dataset(
        {
            "x": (("i",), rng.uniform(0.5, 2.5, n)),
            "y": (("i",), rng.uniform(0.5, 2.5, n)),
        },
        coords={"i": np.arange(n)},
    )
    context = xql.XarrayContext()
    context.from_dataset("g", ds, chunks={"i": 5})
    return context, ds


def _ordered(df, key="i"):
    """Collect a result DataFrame into a dict of column -> numpy array, sorted
    by the integer key column so comparisons are index-aligned."""
    pdf = df.to_pandas().sort_values(key)
    return {c: pdf[c].to_numpy() for c in pdf.columns}


def test_grad_sin_is_cos(ctx):
    val = np.linspace(0.1, 3.0, 16)
    res = _ordered(ctx.sql("SELECT i, grad(sin(val), val) AS d FROM t"))
    np.testing.assert_allclose(res["d"], np.cos(val))


def test_grad_product_rule(ctx):
    val = np.linspace(0.1, 3.0, 16)
    res = _ordered(ctx.sql("SELECT i, grad(sin(val) * val, val) AS d FROM t"))
    np.testing.assert_allclose(res["d"], np.cos(val) * val + np.sin(val))


def test_grad_exp_equals_value(ctx):
    val = np.linspace(0.1, 3.0, 16)
    res = _ordered(
        ctx.sql("SELECT i, exp(val) AS v, grad(exp(val), val) AS d FROM t")
    )
    np.testing.assert_allclose(res["d"], np.exp(val))
    np.testing.assert_allclose(res["d"], res["v"])


def test_grad_quotient_and_power(ctx):
    val = np.linspace(0.1, 3.0, 16)
    res = _ordered(
        ctx.sql(
            "SELECT i, grad(1.0 / val, val) AS dinv, "
            "grad(power(val, 3), val) AS dcube FROM t"
        )
    )
    np.testing.assert_allclose(res["dinv"], -1.0 / val**2)
    np.testing.assert_allclose(res["dcube"], 3.0 * val**2)


def test_non_grad_query_is_unaffected(ctx):
    # Queries without grad() bypass the rewrite and behave normally.
    res = _ordered(ctx.sql("SELECT i, val FROM t"))
    np.testing.assert_allclose(res["val"], np.linspace(0.1, 3.0, 16))


def test_unsupported_function_raises(ctx):
    # atan2 has no derivative rule yet -> a clear error, not a wrong answer.
    with pytest.raises(Exception):
        ctx.sql("SELECT grad(atan2(val, val), val) AS d FROM t").to_pandas()


def test_multi_input_grad_columns(ctx_xy):
    # A full Jacobian written as separate scalar grad() columns:
    # f = x*y  ->  df/dx = y, df/dy = x.
    context, ds = ctx_xy
    res = _ordered(
        context.sql(
            "SELECT i, grad(x * y, x) AS dfdx, grad(x * y, y) AS dfdy FROM g"
        )
    )
    np.testing.assert_allclose(res["dfdx"], ds["y"].values)
    np.testing.assert_allclose(res["dfdy"], ds["x"].values)


def test_jacobian_array(ctx_xy):
    # jacobian(f, [x, y]) returns the gradient row [df/dx, df/dy] per row.
    context, ds = ctx_xy
    res = _ordered(
        context.sql("SELECT i, jacobian(x * y, [x, y]) AS jac FROM g")
    )
    jac = np.stack([np.asarray(v, dtype=float) for v in res["jac"]])
    # column 0 is df/dx = y, column 1 is df/dy = x
    np.testing.assert_allclose(jac[:, 0], ds["y"].values)
    np.testing.assert_allclose(jac[:, 1], ds["x"].values)


def test_jacobian_array_nonlinear(ctx_xy):
    # jacobian(sin(x) * y, [x, y]) = [cos(x)*y, sin(x)]
    context, ds = ctx_xy
    x, y = ds["x"].values, ds["y"].values
    res = _ordered(
        context.sql("SELECT i, jacobian(sin(x) * y, [x, y]) AS jac FROM g")
    )
    jac = np.stack([np.asarray(v, dtype=float) for v in res["jac"]])
    np.testing.assert_allclose(jac[:, 0], np.cos(x) * y)
    np.testing.assert_allclose(jac[:, 1], np.sin(x))
