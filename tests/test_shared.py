"""Shared SELECT stages retain SQL row semantics and original ddx lineage."""

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr

from xarray_sql import Context

pytest.importorskip("ddxdb")
einfold = pytest.importorskip("einfold._native")
pytestmark = pytest.mark.skipif(
    not getattr(einfold, "RESET_ON_DECODE", False),
    reason="requires the optional EinFold prepared-plan patch",
)

from einfold.datafusion import EinFold

from xarray_sql.shared import SharedExecutor

try:
    from xarray_sql.dense import DenseExecutor
except ImportError:
    DenseExecutor = None

_EXECUTORS = [
    SharedExecutor,
    pytest.param(
        DenseExecutor,
        marks=pytest.mark.skipif(
            DenseExecutor is None, reason="requires compact EinFold programs"
        ),
    ),
]


@pytest.fixture(params=_EXECUTORS)
def executor(request):
    ctx = Context().with_extensions(EinFold())
    ctx.from_dataset("w", xr.Dataset({"v": ("i", [1.0, 2.0, 3.0])}))
    return request.param(ctx)


def test_shared_join_refreshes_build_side_and_returns_snapshots(executor):
    sql = """
    WITH square AS (SELECT i, v*v AS s FROM w),
    doubled AS (SELECT a.i, a.s + b.s AS s FROM square a
                JOIN square b ON a.i = b.i)
    SELECT * FROM doubled ORDER BY i
    """
    query = executor.prepare(sql)
    first = query()
    assert first.to_pydict() == {"i": [0, 1, 2], "s": [2.0, 8.0, 18.0]}
    for values in ([3.0, 2.0, 1.0], [5.0, 6.0, 7.0]):
        rows = query(w=xr.Dataset({"v": ("i", values)}))
        assert rows.equals(executor.ctx.sql(sql).to_arrow_table())
    assert first.to_pydict()["s"] == [2.0, 8.0, 18.0]


def test_virtual_view_gradients_use_original_dependencies(executor):
    executor.register_view("model", "SELECT i, v*v AS s FROM w")
    loss = "WITH loss AS (SELECT SUM(s) AS l FROM model)"
    forward = executor.prepare("SELECT SUM(s) AS l FROM model")
    step = executor.prepare_all(
        [
            loss + " SELECT i, grad(loss, v) AS g FROM w ORDER BY i",
            loss + " SELECT i, v - 0.1*grad(loss, v) AS v FROM w ORDER BY i",
        ]
    )
    for values in ([1.0, 2.0, 3.0], [4.0, -2.0, 0.0]):
        data = xr.Dataset({"v": ("i", values)})
        assert forward(w=data)["l"].to_pylist() == [np.dot(values, values)]
        gradient, updated = step()
        np.testing.assert_allclose(gradient["g"], 2 * np.array(values))
        np.testing.assert_allclose(updated["v"], 0.8 * np.array(values))


def test_gradients_through_reused_ctes_in_a_virtual_view(executor):
    executor.register_view(
        "model",
        """
        WITH squared AS (SELECT i, v*v AS a FROM w),
        mean AS (SELECT AVG(a) AS m FROM squared),
        centered AS (SELECT i, power(a-m, 2) AS s FROM squared CROSS JOIN mean),
        total AS (SELECT SUM(s) AS t FROM centered)
        SELECT i, s+t AS s FROM centered CROSS JOIN total
    """,
    )
    step = executor.prepare_all(
        [
            (
                "WITH loss AS (SELECT SUM(s) AS l FROM model) "
                "SELECT i, grad(loss,v) AS g FROM w ORDER BY i"
            )
        ]
    )
    for values in ([1.0, 2.0, 3.0], [-2.0, 4.0, 1.0]):
        v = np.array(values)
        [rows] = step(w=xr.Dataset({"v": ("i", v)}))
        np.testing.assert_allclose(rows["g"], 16 * v * (v * v - np.mean(v * v)))


def test_registration_and_coordinate_changes_refresh_plans(executor):
    query = executor.prepare(
        "WITH t AS (SELECT i, v FROM w WHERE v > 0) "
        "SELECT a.i, a.v+b.v AS v FROM t a JOIN t b ON a.i=b.i ORDER BY a.i"
    )
    assert query().num_rows == 3
    for coords, values in (([0, 1, 2], [-1.0, 2.0, 4.0]), ([8, 9], [5.0, 6.0])):
        executor.ctx.deregister_table("w")
        executor.ctx.from_dataset(
            "w", xr.Dataset({"v": ("i", values)}, coords={"i": coords})
        )
        assert query().to_pydict() == {
            "i": [i for i, v in zip(coords, values) if v > 0],
            "v": [2 * v for v in values if v > 0],
        }


def test_empty_stages_can_gain_and_lose_rows_after_inplace_writes(executor):
    query = executor.prepare(
        "WITH t AS (SELECT i,v FROM w WHERE v < 0) "
        "SELECT a.i,a.v+b.v AS v FROM t a JOIN t b ON a.i=b.i ORDER BY a.i"
    )
    for values in (
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [-1.0, -2.0, 3.0],
        [1.0, 2.0, 3.0],
    ):
        rows = query(w=xr.Dataset({"v": ("i", values)}))
        assert rows.to_pydict() == {
            "i": [i for i, v in enumerate(values) if v < 0],
            "v": [2 * v for v in values if v < 0],
        }


def test_changed_domains_refresh_every_shared_stage(executor):
    query = executor.prepare("""
        WITH a AS (SELECT i,v FROM w WHERE v < 0),
        b AS (SELECT x.i,x.v+y.v AS v FROM a x JOIN a y ON x.i=y.i),
        c AS (SELECT x.i,x.v+y.v AS v FROM b x JOIN b y ON x.i=y.i)
        SELECT x.i,x.v+y.v AS v FROM c x JOIN c y ON x.i=y.i ORDER BY x.i
    """)
    for values in (
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, 3.0],
        [1.0, -3.0, -4.0],
        [1.0, 2.0, 3.0],
    ):
        rows = query(w=xr.Dataset({"v": ("i", values)}))
        assert rows.to_pydict() == {
            "i": [i for i, v in enumerate(values) if v < 0],
            "v": [8 * v for v in values if v < 0],
        }


@pytest.mark.parametrize("executor_type", _EXECUTORS)
def test_shared_stages_keep_nulls_and_duplicate_rows(executor_type):
    ctx = Context().with_extensions(EinFold())
    ctx.register_record_batches(
        "w",
        [
            [
                pa.record_batch(
                    {
                        "i": [0, 0, 1],
                        "v": [1.0, None, 3.0],
                    }
                )
            ]
        ],
    )
    executor = executor_type(ctx)
    sql = (
        "WITH t AS (SELECT i, v FROM w) "
        "SELECT a.i, a.v AS a, b.v AS b FROM t a JOIN t b ON a.i=b.i "
        "ORDER BY a.i, a.v NULLS FIRST, b.v NULLS FIRST"
    )
    query = executor.prepare(sql)
    for _ in range(2):
        assert query().equals(ctx.sql(sql).to_arrow_table())


def test_optimizer_scalar_functions_remain_available(executor):
    rows = executor.prepare(
        "WITH norm AS (SELECT sqrt(SUM(v*v)) AS n FROM w), "
        "clip AS (SELECT least(1.0, 1.0/(n+1e-6)) AS c FROM norm) "
        "SELECT i, v*clip.c AS v FROM w CROSS JOIN clip ORDER BY i"
    )()
    np.testing.assert_allclose(
        rows["v"], np.array([1.0, 2.0, 3.0]) / (np.sqrt(14) + 1e-6)
    )


def test_ordered_stages_preserve_quoted_names_and_window_results(executor):
    sql = """
    WITH t AS (SELECT i AS "axis.name", v FROM w),
    doubled AS (SELECT a."axis.name", a.v+b.v AS v FROM t a JOIN t b
                ON a."axis.name"=b."axis.name")
    SELECT "axis.name", SUM(v) OVER (
      ORDER BY "axis.name" ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s
    FROM doubled ORDER BY "axis.name"
    """
    query = executor.prepare(sql)
    for values in ([3.0, 1.0, 2.0], [5.0, -1.0, 4.0]):
        rows = query(w=xr.Dataset({"v": ("i", values)}))
        assert rows.equals(executor.ctx.sql(sql).to_arrow_table())


def test_many_gradient_contributions_share_a_bounded_reduction_graph(executor):
    terms = " UNION ALL ".join(
        f"SELECT i,power(v,{p}) AS y FROM w" for p in range(2, 42)
    )
    step = executor.prepare_all(
        [
            (
                f"WITH terms AS ({terms}), loss AS (SELECT SUM(y) AS l FROM terms) "
                "SELECT i,grad(loss,v) AS g FROM w ORDER BY i"
            )
        ]
    )
    for values in ([1.0, 2.0, 3.0], [2.0, -1.0, 0.0]):
        v = np.array(values)
        [rows] = step(w=xr.Dataset({"v": ("i", v)}))
        np.testing.assert_allclose(
            rows["g"], sum(p * v ** (p - 1) for p in range(2, 42))
        )


@pytest.mark.parametrize("executor_type", _EXECUTORS)
def test_matrix_gradients_through_a_reconstructed_normalized_operand(
    executor_type,
):
    rng = np.random.default_rng(48)
    a, b, g = (
        rng.normal(size=(4, 3)),
        rng.normal(size=(3, 5)),
        rng.normal(size=3),
    )
    ctx = Context().with_extensions(EinFold())
    ctx.from_dataset("a", xr.Dataset({"v": (("i", "j"), a)}))
    ctx.from_dataset("b", xr.Dataset({"w": (("j", "k"), b)}))
    ctx.from_dataset("g", xr.Dataset({"gain": ("j", g)}))
    executor = executor_type(ctx)
    executor.register_view(
        "normalized",
        """
        WITH mean AS (SELECT i,AVG(v) AS m FROM a GROUP BY i),
        variance AS (SELECT a.i,AVG(power(a.v-mean.m,2)) AS v
                     FROM a JOIN mean ON a.i=mean.i GROUP BY a.i)
        SELECT a.i,a.j,(a.v-mean.m)/sqrt(variance.v+1e-5)*g.gain AS v
        FROM a JOIN mean ON a.i=mean.i JOIN variance ON a.i=variance.i
        JOIN g ON a.j=g.j
    """,
    )
    step = executor.prepare_all(
        [
            """
        WITH predicted AS (SELECT x.i,b.k,SUM(x.v*b.w) AS v
              FROM normalized x JOIN b ON x.j=b.j GROUP BY x.i,b.k),
        loss AS (SELECT SUM(v*v)/2 AS l FROM predicted)
        SELECT j,k,grad(loss,w) AS g FROM b ORDER BY j,k
    """
        ]
    )
    centered = a - a.mean(axis=1, keepdims=True)
    normalized = (
        centered
        / np.sqrt(np.mean(centered**2, axis=1, keepdims=True) + 1e-5)
        * g
    )
    for scale in (1.0, -0.25):
        current = b * scale
        expected = normalized.T @ (normalized @ current)
        [rows] = step(b=xr.Dataset({"w": (("j", "k"), current)}))
        np.testing.assert_allclose(
            rows["g"].to_numpy().reshape(3, 5),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "sql",
    [
        "WITH t AS (SELECT random() AS x) SELECT * FROM t",
        "WITH RECURSIVE t AS (SELECT 1) SELECT * FROM t",
        "WITH t AS (WITH s AS (SELECT 1) SELECT * FROM s) SELECT * FROM t",
        "SELECT 1; SELECT 2",
    ],
)
def test_unsupported_staging_fails_explicitly(executor, sql):
    with pytest.raises(ValueError):
        executor.prepare(sql)
