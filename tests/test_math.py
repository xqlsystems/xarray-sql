"""Error-function SQL values and analytic gradients match the array paths."""

import math
import os

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr
from datafusion import udf

from xarray_sql import Context
from xarray_sql.math import register_erf


def test_erf_sql_nulls_and_replacement():
    ctx = Context()
    register_erf(ctx)
    rows = ctx.sql(
        "SELECT erf(0.0) AS zero, erf(CAST(NULL AS DOUBLE)) AS missing"
    ).to_arrow_table()
    assert rows.to_pydict() == {"zero": [0.0], "missing": [None]}
    ctx.register_udf(
        udf(
            lambda x: pa.array(np.asarray(x) + 10),
            [pa.float64()],
            pa.float64(),
            "immutable",
            name="erf",
        )
    )
    assert ctx.sql("SELECT erf(1.0) AS v").to_arrow_table().to_pydict() == {
        "v": [11.0]
    }


def test_erf_replacement_is_preserved_by_dense_fallback():
    pytest.importorskip("einfold.dense_program")
    from xarray_sql.dense import DenseExecutor

    ctx = Context()
    register_erf(ctx)
    ctx.from_dataset("a", xr.Dataset({"v": ("i", [0.0, 1.0])}))
    executor = DenseExecutor(ctx)
    run = executor.prepare("SELECT i,erf(v) AS x FROM a ORDER BY i")
    np.testing.assert_allclose(run()["x"].to_numpy(), [0.0, math.erf(1.0)])
    ctx.register_udf(
        udf(
            lambda x: pa.array(np.asarray(x) + 10),
            [pa.float64()],
            pa.float64(),
            "immutable",
            name="erf",
        )
    )
    assert run().to_pydict() == {"i": [0, 1], "x": [10.0, 11.0]}


@pytest.mark.parametrize(
    "precision",
    [
        "sql",
        "float64",
        pytest.param(
            "float32",
            marks=pytest.mark.skipif(
                os.environ.get("MLX_ENABLE_TF32") != "0",
                reason="GPU tests require MLX_ENABLE_TF32=0",
            ),
        ),
    ],
)
def test_exact_gelu_sql_gradient(precision):
    ddx = pytest.importorskip("ddxdb")
    if "erf" not in ddx.supported_functions():
        pytest.skip("requires the optional ddx erf patch")
    ctx = Context()
    register_erf(ctx)
    values = np.array([-4.0, -1.0, 0.0, 0.5, 2.0])
    ctx.from_dataset(
        "a",
        xr.Dataset({"v": ("i", values)}, coords={"i": np.arange(len(values))}),
    )
    sql = """WITH loss AS (SELECT SUM(0.5*v*(1.0+erf(v/sqrt(2.0)))) AS l FROM a)
        SELECT i, grad(loss,v) AS g FROM a ORDER BY i"""
    if precision == "sql":
        rows = ctx.sql(sql).to_arrow_table()
    else:
        pytest.importorskip("einfold.dense_program")
        from xarray_sql.dense import DenseExecutor

        executor = DenseExecutor(ctx, precision=precision, fallback=False)
        rows = executor.prepare(sql)()
    expected = 0.5 * (
        1 + np.vectorize(math.erf)(values / np.sqrt(2))
    ) + values * np.exp(-(values**2) / 2) / np.sqrt(2 * np.pi)
    np.testing.assert_allclose(
        rows["g"].to_numpy(), expected, atol=1e-6, rtol=1e-6
    )
