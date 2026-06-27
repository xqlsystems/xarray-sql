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
