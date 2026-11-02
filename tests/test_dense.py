"""Compact execution preserves SQL rows and live Dataset inputs."""

import os

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr
from datafusion import udf

from xarray_sql import Context

pytest.importorskip("ddxdb")
pytest.importorskip("einfold.dense_program")
from ddxdb import UnsupportedExpression
from einfold.datafusion import EinFold

from xarray_sql.dense import DenseExecutor

_PRECISIONS = [
    "float64",
    pytest.param(
        "float32",
        marks=pytest.mark.skipif(
            os.environ.get("MLX_ENABLE_TF32") != "0",
            reason="GPU tests require MLX_ENABLE_TF32=0",
        ),
    ),
]


@pytest.mark.parametrize("precision", _PRECISIONS)
def test_conditional_contraction_stays_within_array_bound(precision):
    ctx = Context()
    left = np.arange(64, dtype=float).reshape(8, 8) / 64
    right = np.arange(64, dtype=float).reshape(8, 8) / 32
    ctx.from_dataset("a", xr.Dataset({"v": (("i", "k"), left)}))
    ctx.from_dataset("b", xr.Dataset({"v": (("k", "j"), right)}))
    executor = DenseExecutor(
        ctx, precision=precision, fallback=False, max_array_bytes=2048
    )
    rows = executor.prepare("""WITH products AS (
      SELECT a.i,b.j,a.k,a.v*b.v AS p FROM a JOIN b ON a.k=b.k)
      SELECT i,j,SUM(CASE WHEN i>=j THEN p ELSE 0.0 END) AS x,
                 SUM(CASE WHEN i>=j THEN 0.0 ELSE p END) AS y
      FROM products GROUP BY i,j ORDER BY i,j""")()
    result = left @ right
    i, j = np.indices(result.shape)
    np.testing.assert_allclose(
        np.asarray(rows["x"]).reshape(8, 8), np.where(i >= j, result, 0)
    )
    np.testing.assert_allclose(
        np.asarray(rows["y"]).reshape(8, 8), np.where(i >= j, 0, result)
    )


@pytest.fixture(params=_PRECISIONS)
def executor(request):
    ctx = Context().with_extensions(EinFold())
    ctx.from_dataset(
        "a",
        xr.Dataset(
            {"v": (("i", "j"), np.arange(6, dtype=float).reshape(2, 3))},
            coords={"i": [-2, 7], "j": [8, 2, 5]},
        ),
    )
    ctx.from_dataset(
        "b",
        xr.Dataset(
            {"w": (("j", "k"), np.arange(6, dtype=float).reshape(3, 2))},
            coords={"j": [5, 8, 2], "k": [9, 1]},
        ),
    )
    return DenseExecutor(ctx, precision=request.param, fallback=False)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.i,b.k,SUM(a.v*b.w) AS v FROM a JOIN b ON a.j=b.j GROUP BY a.i,b.k ORDER BY a.i,b.k",
        "SELECT i,AVG(v) AS m,COUNT(v) AS n FROM a WHERE v>1 GROUP BY i ORDER BY i",
        "SELECT SUM(v) AS s,AVG(v) AS m,COUNT(v) AS n FROM a WHERE v<0",
        "SELECT i,SUM(v) AS s FROM a WHERE v<0 GROUP BY i ORDER BY i",
        "SELECT i,j,MAX(v) OVER(PARTITION BY i) AS m FROM a ORDER BY i,j",
        "SELECT i,j,i/3 AS q,i%3 AS r FROM a ORDER BY i,j",
        "SELECT i,j,(v-3.5)%2.0 AS r FROM a ORDER BY i,j",
        "SELECT i,j,power(2,63) AS a,power(2,-1) AS b FROM a ORDER BY i,j",
        "SELECT i,j,(CASE WHEN v>2 THEN TRUE END) AND FALSE AS a,(CASE WHEN v>2 THEN TRUE END) OR TRUE AS b FROM a ORDER BY i,j",
        "WITH t AS (SELECT i,CASE WHEN v>2 THEN v END AS v FROM a) SELECT i,SUM(v) AS s,COUNT(v) AS n FROM t GROUP BY i ORDER BY i",
        "WITH t AS (SELECT i,v FROM a UNION ALL SELECT i,2*v AS v FROM a) SELECT i,SUM(v) AS s,COUNT(v) AS n FROM t GROUP BY i ORDER BY i",
        "SELECT a.i,a.j,b.w FROM a LEFT JOIN (SELECT j,k,w FROM b WHERE k=1) b ON a.j=b.j ORDER BY a.i,a.j",
        "SELECT a.i,a.j,c.k FROM a LEFT JOIN (SELECT j,k,w FROM b WHERE k=1 AND w>1) b ON a.j=b.j JOIN (SELECT j,k FROM b WHERE k=9) c ON b.j=c.j ORDER BY a.i,a.j,c.k",
        "SELECT a.i,a.j,c.k FROM a LEFT JOIN (SELECT j,k,w FROM b WHERE k=1 AND w>1) b ON a.j=b.j JOIN (SELECT j,k FROM b WHERE k=9) c ON b.j IS NOT DISTINCT FROM c.j ORDER BY a.i,a.j,c.k",
    ],
)
def test_dense_sql_rows_match_datafusion(executor, sql):
    actual = executor.prepare(sql)()
    expected = executor.ctx.sql(sql).to_arrow_table()
    assert actual.column_names == expected.column_names
    for a, e in zip(actual.columns, expected.columns):
        assert a.null_count == e.null_count
        assert a.to_pylist() == e.to_pylist()


@pytest.mark.parametrize("precision", _PRECISIONS)
def test_dense_gather_gradient_and_snapshots(precision):
    ctx = Context().with_extensions(EinFold())
    weights = xr.Dataset(
        {"v": (("tok", "e"), np.arange(6, dtype=float).reshape(3, 2))},
        coords={"tok": [8, -1, 4]},
    )
    ctx.from_dataset("w", weights)
    ctx.from_dataset("batch", xr.Dataset({"tok": ("pos", [4, 4, 8])}))
    executor = DenseExecutor(ctx, precision=precision, fallback=False)
    executor.register_view(
        "model", "SELECT b.pos,w.e,w.v FROM batch b JOIN w ON b.tok=w.tok"
    )
    run = executor.prepare(
        "WITH loss AS (SELECT SUM(v*v)/2 AS l FROM model) SELECT tok,e,grad(loss,v) AS g FROM w ORDER BY tok,e"
    )
    first = run()
    assert first.to_pydict() == {
        "tok": [-1, -1, 4, 4, 8, 8],
        "e": [0, 1, 0, 1, 0, 1],
        "g": [0.0, 0.0, 8.0, 10.0, 0.0, 1.0],
    }
    weights.v.values[:] += 2
    second = run()
    assert second["g"].to_pylist() == [0.0, 0.0, 12.0, 14.0, 2.0, 3.0]
    assert first["g"].to_pylist() == [0.0, 0.0, 8.0, 10.0, 0.0, 1.0]


def test_dense_falls_back_for_overridden_scalar_function(executor):
    ctx = executor.ctx
    ctx.register_udf(
        udf(
            lambda a: pa.array(np.asarray(a) + 100),
            [pa.float64()],
            pa.float64(),
            "immutable",
            "exp",
        )
    )
    run = DenseExecutor(ctx).prepare(
        "SELECT i,j,exp(v) AS v FROM a ORDER BY i,j"
    )
    assert run().equals(
        ctx.sql("SELECT i,j,exp(v) AS v FROM a ORDER BY i,j").to_arrow_table()
    )
    ctx.register_udf(
        udf(
            lambda a: pa.array(np.asarray(a) + 200),
            [pa.float64()],
            pa.float64(),
            "immutable",
            "exp",
        )
    )
    assert run().equals(
        ctx.sql("SELECT i,j,exp(v) AS v FROM a ORDER BY i,j").to_arrow_table()
    )


def test_dense_coordinate_cast_retains_its_integer_width():
    ctx = Context().with_extensions(EinFold())
    ctx.from_dataset(
        "a",
        xr.Dataset(
            {"v": ("i", [1.0, 2.0])},
            coords={"i": np.array([2000000000, 2000000001], dtype=np.int32)},
        ),
    )
    sql = "SELECT i,CAST(i AS BIGINT)*2 AS j FROM a ORDER BY i"
    assert (
        DenseExecutor(ctx, fallback=False)
        .prepare(sql)()
        .equals(ctx.sql(sql).to_arrow_table())
    )


def test_dense_base_log_fallback_never_writes_an_input():
    ctx = Context().with_extensions(EinFold())
    values = np.array([1.0, 2.0, 3.0])
    ctx.from_dataset("x", xr.Dataset({"v": ("i", values)}))
    sql = "SELECT i,log(2.,v) AS v FROM x ORDER BY i"
    expected = ctx.sql(sql).to_arrow_table()
    assert DenseExecutor(ctx).prepare(sql)().equals(expected)
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


def test_dense_integer_sum_uses_the_sql_accumulator_width():
    ctx = Context().with_extensions(EinFold())
    ctx.from_dataset(
        "x",
        xr.Dataset(
            {"v": ("i", np.array([2000000000, 2000000001], dtype=np.int32))}
        ),
    )


@pytest.mark.parametrize("precision", _PRECISIONS)
def test_dense_null_and_empty_filters_do_not_evaluate_deleted_rows(precision):
    ctx = Context().with_extensions(EinFold())
    ctx.from_dataset("x", xr.Dataset({"v": ("i", [1.0, 2.0, 3.0])}))
    executor = DenseExecutor(ctx, precision=precision, fallback=False)
    for sql in [
        "SELECT i,v FROM x WHERE i=NULL ORDER BY i",
        "SELECT i,1/(i-i) AS v FROM x WHERE i<0 ORDER BY i",
    ]:
        assert executor.prepare(sql)().equals(ctx.sql(sql).to_arrow_table())


@pytest.mark.parametrize("precision", _PRECISIONS)
def test_dense_integer_averages_with_repeated_group_keys(precision):
    ctx = Context().with_extensions(EinFold())
    ctx.from_dataset(
        "x", xr.Dataset({"key": ("i", [1, 1, 2, 2]), "v": ("i", [1, 2, 3, 4])})
    )
    sql = "SELECT key,AVG(v) AS m,SUM(v) AS s FROM x GROUP BY key ORDER BY key"
    assert (
        DenseExecutor(ctx, precision=precision, fallback=False)
        .prepare(sql)()
        .equals(ctx.sql(sql).to_arrow_table())
    )
    sql = "SELECT SUM(v) AS s,AVG(v) AS m FROM x"
    assert (
        DenseExecutor(ctx, fallback=False)
        .prepare(sql)()
        .equals(ctx.sql(sql).to_arrow_table())
    )


@pytest.mark.parametrize("precision", _PRECISIONS)
def test_dense_unsupported_backward_region_restarts_with_original_sql(
    precision,
):
    ctx = Context().with_extensions(EinFold())
    values = np.arange(6.0, dtype=float).reshape(2, 3)
    ctx.from_dataset("a", xr.Dataset({"v": (("i", "j"), values)}))
    query = DenseExecutor(ctx, precision=precision).prepare(
        "WITH t AS (SELECT CAST(i AS DOUBLE) AS i,CAST(j AS DOUBLE) AS j,SUM(v*v) AS s FROM a GROUP BY CAST(i AS DOUBLE),CAST(j AS DOUBLE)),loss AS (SELECT SUM(s) AS l FROM t) SELECT i,j,grad(loss,v) AS g FROM a ORDER BY i,j"
    )
    assert query()["g"].to_pylist() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    values[:] += 1
    assert query()["g"].to_pylist() == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def test_dense_directional_derivatives_and_typed_refusal(executor):
    ctx = executor.ctx
    tangent = "WITH tangent AS (SELECT j,k,CAST(k AS DOUBLE) AS w FROM b) "
    rows = executor.prepare(
        tangent
        + "SELECT a.i,b.k,jvp(SUM(a.v*b.w),b.w,tangent) AS p FROM a JOIN b ON a.j=b.j GROUP BY a.i,b.k ORDER BY i,k"
    )()
    assert rows.to_pydict() == {
        "i": [-2, -2, 7, 7],
        "k": [1, 9, 1, 9],
        "p": [3.0, 27.0, 12.0, 108.0],
    }
    ctx.from_dataset(
        "cotangent",
        xr.Dataset(
            {"v": (("i", "k"), [[1.0, 2.0], [3.0, 4.0]])},
            coords={"i": [-2, 7], "k": [9, 1]},
        ),
    )
    predicted = "WITH pred AS (SELECT a.i,b.k,SUM(a.v*b.w) AS v FROM a JOIN b ON a.j=b.j GROUP BY a.i,b.k) "
    rows = executor.prepare(
        predicted + "SELECT j,k,vjp(pred,w,cotangent) AS g FROM b ORDER BY j,k"
    )()
    assert rows.to_pydict() == {
        "j": [2, 2, 5, 5, 8, 8],
        "k": [1, 9, 1, 9, 1, 9],
        "g": [18.0, 13.0, 24.0, 17.0, 12.0, 9.0],
    }
    rows = executor.prepare(
        "WITH loss AS (SELECT SUM(v*ddx_stop_gradient(v)) AS l FROM a) SELECT i,j,grad(loss,v) AS g FROM a ORDER BY i,j"
    )()
    assert rows["g"].to_pylist() == [1.0, 2.0, 0.0, 4.0, 5.0, 3.0]
    with pytest.raises(UnsupportedExpression, match="GROUP BY key"):
        executor.prepare(
            "WITH buckets AS (SELECT ROUND(v) AS k,SUM(v*v) AS s FROM a GROUP BY ROUND(v)),loss AS (SELECT SUM(k*s) AS l FROM buckets) SELECT i,j,grad(loss,v) AS g FROM a"
        )()
