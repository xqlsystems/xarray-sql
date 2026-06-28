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


def test_higher_order_grad(ctx):
    # Nested grad() differentiates repeatedly: the inner call is rewritten
    # first, then the outer differentiates its result.
    val = np.linspace(0.1, 3.0, 16)
    res = _ordered(
        ctx.sql(
            "SELECT i, "
            "grad(grad(sin(val), val), val) AS d2_sin, "
            "grad(grad(power(val, 3), val), val) AS d2_cube FROM t"
        )
    )
    np.testing.assert_allclose(res["d2_sin"], -np.sin(val))  # -sin
    np.testing.assert_allclose(res["d2_cube"], 6.0 * val)  # d2/dx2 x^3 = 6x


def test_third_order_grad(ctx):
    val = np.linspace(0.1, 3.0, 16)
    res = _ordered(
        ctx.sql(
            "SELECT i, grad(grad(grad(sin(val), val), val), val) AS d3 FROM t"
        )
    )
    np.testing.assert_allclose(res["d3"], -np.cos(val))  # d3/dx3 sin = -cos


def test_non_grad_query_is_unaffected(ctx):
    # Queries without grad() bypass the rewrite and behave normally.
    res = _ordered(ctx.sql("SELECT i, val FROM t"))
    np.testing.assert_allclose(res["val"], np.linspace(0.1, 3.0, 16))


def test_unsupported_function_raises(ctx):
    # atan2 has no derivative rule yet -> a clear error, not a wrong answer.
    with pytest.raises(Exception):
        ctx.sql("SELECT grad(atan2(val, val), val) AS d FROM t").to_pandas()


def test_grad_inside_aggregate(ctx):
    # Differentiation through an aggregate is just linearity:
    #   AGG(grad(f, x)) == d/dx AGG(f). grad rewrites to plain SQL before the
    #   aggregate runs, so this composes with no special machinery.
    val = np.linspace(0.1, 3.0, 16)
    res = ctx.sql(
        "SELECT SUM(grad(val * val, val)) AS s, "
        "AVG(grad(sin(val), val)) AS a FROM t"
    ).to_pandas()
    np.testing.assert_allclose(res["s"][0], np.sum(2 * val))
    np.testing.assert_allclose(res["a"][0], np.mean(np.cos(val)))


def test_gradient_descent_in_sql():
    # End to end: fit y ~= a*x + b by minimising MSE, with the gradients
    # w.r.t. the parameters computed in SQL via AVG(grad(loss, param)).
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0, n)
    a_true, b_true = 2.0, -1.0
    y = a_true * x + b_true + rng.normal(0.0, 0.01, n)
    data = xr.Dataset(
        {"x": (("i",), x), "y": (("i",), y)}, coords={"i": np.arange(n)}
    )
    ctx = xql.XarrayContext()
    ctx.from_dataset("d", data, chunks={"i": n})

    resid = "(y - (a * x + b))"
    loss = f"{resid} * {resid}"
    a, b, lr = 0.0, 0.0, 0.4
    losses = []
    for _ in range(120):
        if "params" in ctx._registered_datasets:
            ctx.deregister_table("params")
            del ctx._registered_datasets["params"]
        params = xr.Dataset(
            {"a": (("p",), [a]), "b": (("p",), [b])}, coords={"p": [0]}
        )
        ctx.from_dataset("params", params, chunks={"p": 1})
        row = ctx.sql(
            f"SELECT AVG({loss}) AS loss, "
            f"AVG(grad({loss}, a)) AS dl_da, "
            f"AVG(grad({loss}, b)) AS dl_db FROM d CROSS JOIN params"
        ).to_pandas()
        losses.append(float(row["loss"][0]))
        a -= lr * float(row["dl_da"][0])
        b -= lr * float(row["dl_db"][0])

    assert losses[-1] < losses[0]  # loss decreased
    np.testing.assert_allclose([a, b], [a_true, b_true], atol=0.05)


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


def test_jvp_forward_directional_derivative(ctx_xy):
    # jvp(f, x, dx) = df/dx * dx. With f = sin(x)*y and a constant tangent.
    context, ds = ctx_xy
    x, y = ds["x"].values, ds["y"].values
    res = _ordered(context.sql("SELECT i, jvp(sin(x) * y, x, 2.0) AS t FROM g"))
    np.testing.assert_allclose(res["t"], (np.cos(x) * y) * 2.0)


def test_jvp_multi_input_is_sum(ctx_xy):
    # A full directional derivative is the sum of per-input jvp terms:
    # df/dx*dx + df/dy*dy for f = x*y, with dx=1, dy=1 -> y + x.
    context, ds = ctx_xy
    res = _ordered(
        context.sql(
            "SELECT i, jvp(x * y, x, 1.0) + jvp(x * y, y, 1.0) AS t FROM g"
        )
    )
    np.testing.assert_allclose(res["t"], ds["y"].values + ds["x"].values)


def test_vjp_reverse_pullback(ctx_xy):
    # vjp(f, x, w) = w * df/dx. With f = sin(x)*y and cotangent w = 3.0.
    context, ds = ctx_xy
    x, y = ds["x"].values, ds["y"].values
    res = _ordered(context.sql("SELECT i, vjp(sin(x) * y, x, 3.0) AS s FROM g"))
    np.testing.assert_allclose(res["s"], 3.0 * (np.cos(x) * y))


@pytest.fixture
def ctx_mixed():
    # A mixed-dimension dataset registers as schema-qualified tables:
    #   era5.time_x        (surface, 2 dims)
    #   era5.time_x_level  (atmosphere, 3 dims)
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "sfc": (("time", "x"), rng.uniform(0.5, 2.5, (3, 4))),
            "atm": (("time", "x", "level"), rng.uniform(0.5, 2.5, (3, 4, 2))),
        },
        coords={"time": [0, 1, 2], "x": np.arange(4.0), "level": [0, 1]},
    )
    context = xql.XarrayContext()
    context.from_dataset("era5", ds, chunks={"time": 1})
    return context, ds


def test_grad_on_qualified_surface_table(ctx_mixed):
    context, ds = ctx_mixed
    res = _ordered(
        context.sql(
            "SELECT time, x, sfc, grad(sin(sfc), sfc) AS d FROM era5.time_x"
        ),
        key="sfc",
    )
    np.testing.assert_allclose(res["d"], np.cos(res["sfc"]))


def test_grad_on_qualified_atmosphere_table(ctx_mixed):
    context, ds = ctx_mixed
    res = _ordered(
        context.sql(
            "SELECT atm, grad(power(atm, 2), atm) AS d FROM era5.time_x_level"
        ),
        key="atm",
    )
    np.testing.assert_allclose(res["d"], 2.0 * res["atm"])


def test_jvp_and_vjp_agree_for_unit_seed(ctx_xy):
    # Forward (unit tangent) and reverse (unit cotangent) coincide for a
    # scalar output -- both contract the same partial derivative.
    context, _ = ctx_xy
    res = _ordered(
        context.sql(
            "SELECT i, jvp(sin(x) * y, x, 1.0) AS fwd, "
            "vjp(sin(x) * y, x, 1.0) AS rev FROM g"
        )
    )
    np.testing.assert_allclose(res["fwd"], res["rev"])
