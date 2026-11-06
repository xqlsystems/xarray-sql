"""Public SQL and ddx contracts of the optional Apple silicon executor."""

import os

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr
from datafusion import SessionConfig

pytest.importorskip("mlx.core", exc_type=ImportError)
if os.environ.get("MLX_ENABLE_TF32") != "0":
    pytest.skip(
        "GPU tests require MLX_ENABLE_TF32=0 at launch", allow_module_level=True
    )
pytest.importorskip("einfold")
pytest.importorskip("ddxdb")

from ddxdb import UnsupportedExpression

from xarray_sql import Context, ad
from xarray_sql.mlx import MLX

PRODUCT = """
SELECT a.i, b.k, SUM(a.v * b.v) AS v
FROM a JOIN b ON a.j = b.j GROUP BY a.i, b.k ORDER BY a.i, b.k
"""


def _matrix(value, dims):
    return xr.Dataset(
        {"v": (dims, value)},
        coords={d: np.arange(n) for d, n in zip(dims, value.shape)},
        attrs={"experiment": "accelerated"},
    )


@pytest.fixture(params=["float64", "float32"])
def accelerated(request):
    try:
        extension = MLX(
            precision=request.param, min_ops=0, min_gpu_ops=0, query_block=3
        )
    except ImportError as exc:
        pytest.skip(str(exc))
    ctx = Context().with_extensions(extension)
    return ctx, extension, 3e-5 if request.param == "float32" else 1e-12


def _same(got, want, tolerance):
    assert got.schema == want.schema
    assert got.column_names == want.column_names
    for g, w in zip(got.columns, want.columns):
        if pa.types.is_floating(w.type):
            assert g.null_count == w.null_count
            assert g.is_null().to_pylist() == w.is_null().to_pylist()
            np.testing.assert_allclose(
                g.to_numpy(),
                w.to_numpy(),
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
            )
            gv, wv = g.to_numpy(), w.to_numpy()
            zero = (gv == 0) & (wv == 0)
            np.testing.assert_array_equal(
                np.signbit(gv[zero]), np.signbit(wv[zero])
            )
        else:
            assert g.equals(w)


def test_dense_shapes_row_order_and_current_values(accelerated):
    ctx, extension, tolerance = accelerated
    rng = np.random.default_rng(412)
    for m, k, n in [(3, 7, 5), (17, 3, 1), (1, 9, 13), (23, 11, 19)]:
        a, b = rng.normal(size=(m, k)), rng.normal(size=(k, n))
        ctx.deregister_table("a")
        ctx.deregister_table("b")
        ctx.from_dataset("a", _matrix(a, ("i", "j")))
        ctx.from_dataset("b", _matrix(b, ("j", "k")))
        frame = ctx.sql(PRODUCT)
        for scale in (1.0, -0.75):
            a[:] *= scale
            got = frame.to_dataset(dims=["i", "k"], template="a")
            np.testing.assert_allclose(
                got.v, a @ b, rtol=tolerance, atol=tolerance
            )
            assert got.attrs == {"experiment": "accelerated"}
    device = "gpu" if extension.precision == "float32" else "cpu"
    assert extension.statistics()[f"{device}_matmul"] > 0


@pytest.mark.parametrize(
    "case",
    [
        "null",
        "duplicate",
        "missing",
        "infinity",
        "nan",
        "empty",
        "keys",
        "signed_zero",
        "extreme",
    ],
)
def test_sql_bags_nulls_special_values_and_empty_inputs(accelerated, case):
    ctx, _, tolerance = accelerated
    a = {"i": [0, 0, 1, 1], "j": [0, 1, 0, 1], "v": [1.0, 2.0, 3.0, 4.0]}
    b = {"j": [0, 0, 1, 1], "k": [0, 1, 0, 1], "v": [0.25, 0.5, 0.75, 1.0]}
    if case == "null":
        a["v"] = [None, None, 3.0, 4.0]
    elif case == "duplicate":
        for values in a.values():
            values.append(values[0])
    elif case == "missing":
        for values in b.values():
            values.pop()
    elif case in ("infinity", "nan"):
        a["v"][0] = np.inf if case == "infinity" else np.nan
    elif case == "empty":
        a = {key: [] for key in a}
    elif case == "signed_zero":
        a["v"] = [0.0, -0.0, -0.0, 0.0]
    elif case == "extreme":
        a["v"] = [1e300, -1e300, 1e-300, -1e-300]
    elif case == "keys":
        a["j"] = [None, -1, None, -1]
        b["j"] = [None, None, -1, -1]
    plain = Context()
    for name, data in [("a", a), ("b", b)]:
        batch = pa.record_batch(
            data,
            schema=pa.schema(
                [
                    (key, pa.float64() if key == "v" else pa.int64())
                    for key in data
                ]
            ),
        )
        for con in (ctx, plain):
            con.register_record_batches(name, [[batch]])
    _same(
        ctx.sql(PRODUCT).to_arrow_table(),
        plain.sql(PRODUCT).to_arrow_table(),
        tolerance,
    )


def test_prepared_forward_reads_writes_and_keeps_metadata(accelerated):
    ctx, _, tolerance = accelerated
    a = _matrix(np.ones((5, 3)), ("i", "j"))
    b = _matrix(np.ones((3, 4)), ("j", "k"))
    ctx.from_dataset("a", a)
    ctx.from_dataset("b", b)
    run = ctx.prepare(PRODUCT)
    for scale in (1.0, 2.0, -0.5):
        got = run(a=_matrix(np.full((5, 3), scale), ("i", "j")))
        np.testing.assert_allclose(
            got["v"], np.full(20, 3 * scale), atol=tolerance
        )


ATTENTION = """
WITH scores AS (
  SELECT q.i, k.j, SUM(q.v * k.v) / 2.0 AS s
  FROM q JOIN k ON q.d = k.d WHERE {mask} GROUP BY q.i, k.j),
weights AS (
  SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores),
norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i)
SELECT w.i, v.d, SUM(w.w / n.z * v.v) AS v
FROM weights w JOIN v ON w.j = v.j JOIN norm n ON w.i = n.i
GROUP BY w.i, v.d ORDER BY w.i, v.d
"""


@pytest.mark.parametrize(
    "mask", ["q.i >= k.j", "q.i > k.j + 1", "q.i >= k.j AND q.i < k.j + 3"]
)
def test_attention_masks_are_absent_rows(accelerated, mask):
    ctx, extension, tolerance = accelerated
    plain = Context()
    rng = np.random.default_rng(52)
    datasets = [
        (name, _matrix(rng.normal(size=(9, 4)), dims))
        for name, dims in [
            ("q", ("i", "d")),
            ("k", ("j", "d")),
            ("v", ("j", "d")),
        ]
    ]
    for con in (ctx, plain):
        for name, data in datasets:
            con.from_dataset(name, data)
    sql = ATTENTION.format(mask=mask)
    _same(
        ctx.sql(sql).to_arrow_table(),
        plain.sql(sql).to_arrow_table(),
        tolerance,
    )
    if extension.precision == "float32":
        assert extension.statistics()["gpu_attention"] > 0
    # A non-finite value in a masked row must not contaminate earlier rows.
    datasets[-1][1]["v"].values[-1, 0] = np.inf
    _same(
        ctx.sql(sql).to_arrow_table(),
        plain.sql(sql).to_arrow_table(),
        tolerance,
    )


def test_ddx_grad_vjp_jvp_stop_gradient_and_repeated_steps(accelerated):
    ctx, _extension, tolerance = accelerated
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(7, 4)), rng.normal(size=(4, 3))
    ctx.from_dataset("a", _matrix(a, ("i", "j")))
    ctx.from_dataset("b", _matrix(b, ("j", "k")))
    loss = """WITH pred AS (
      SELECT a.i, b.k, SUM(a.v * b.v) AS p FROM a JOIN b ON a.j=b.j GROUP BY a.i,b.k),
      loss AS (SELECT SUM(p*p)/2 AS l FROM pred) """
    select = "SELECT j,k,grad(loss,v) AS g FROM b ORDER BY j,k"
    step = ctx.prepare(loss + select)
    for scale in (1.0, -0.25, 2.0):
        new = _matrix(b * scale, ("j", "k"))
        rows = step(b=new)
        np.testing.assert_allclose(
            rows["g"].to_numpy().reshape(4, 3),
            a.T @ (a @ new.v.values),
            rtol=tolerance,
            atol=tolerance,
        )
    # JVP uses ddx's tangent relation and the same accelerated contractions.
    tangent = _matrix(np.ones((4, 3)), ("j", "k"))
    ctx.from_dataset("tangent", tangent)
    got = ctx.sql("""SELECT a.i,b.k,jvp(SUM(a.v*b.v),b.v,tangent) AS g
      FROM a JOIN b ON a.j=b.j GROUP BY a.i,b.k ORDER BY i,k""").to_arrow_table()
    np.testing.assert_allclose(
        got["g"].to_numpy().reshape(7, 3),
        a @ np.ones((4, 3)),
        rtol=tolerance,
        atol=tolerance,
    )
    ctx.sql(
        "CREATE VIEW pred AS SELECT a.i,b.k,SUM(a.v*b.v) AS p FROM a JOIN b ON a.j=b.j GROUP BY a.i,b.k"
    )
    ctx.sql("CREATE VIEW cotangent AS SELECT i,k,1.0 AS p FROM pred")
    got = ctx.sql(
        "WITH pred AS (SELECT a.i,b.k,SUM(a.v*b.v) AS p FROM a "
        "JOIN b ON a.j=b.j GROUP BY a.i,b.k) "
        "SELECT j,k,vjp(pred,v,cotangent) AS g FROM b ORDER BY j,k"
    ).to_arrow_table()
    np.testing.assert_allclose(
        got["g"].to_numpy().reshape(4, 3),
        a.T @ np.ones((7, 3)),
        rtol=tolerance,
        atol=tolerance,
    )
    got = ctx.sql("""WITH loss AS (SELECT SUM(v*ddx_stop_gradient(v)) AS l FROM b)
      SELECT j,k,grad(loss,v) AS g FROM b ORDER BY j,k""").to_arrow_table()
    np.testing.assert_allclose(
        got["g"].to_numpy().reshape(4, 3), b, rtol=tolerance, atol=tolerance
    )


def test_backend_threshold_retains_cpu_sql():
    try:
        extension = MLX(precision="float32", min_ops=10**12)
    except ImportError as exc:
        pytest.skip(str(exc))
    ctx = Context(SessionConfig().with_target_partitions(2)).with_extensions(
        extension
    )
    ctx.from_dataset("a", _matrix(np.ones((3, 4)), ("i", "j")))
    ctx.from_dataset("b", _matrix(np.ones((4, 5)), ("j", "k")))
    assert ctx.sql(PRODUCT).to_arrow_table()["v"].to_pylist() == [4.0] * 15
    assert extension.statistics() == {}


def test_batched_products_across_partitions(accelerated):
    ctx, _, tolerance = accelerated
    rng = np.random.default_rng(985)
    a, b = rng.normal(size=(3, 5, 7)), rng.normal(size=(3, 7, 4))
    ctx.from_dataset("a", _matrix(a, ("batch", "i", "j")), chunks={"batch": 1})
    ctx.from_dataset("b", _matrix(b, ("batch", "j", "k")), chunks={"batch": 1})
    got = ctx.sql("""SELECT a.batch,a.i,b.k,SUM(a.v*b.v) AS v
      FROM a JOIN b ON a.batch=b.batch AND a.j=b.j
      GROUP BY a.batch,a.i,b.k ORDER BY a.batch,a.i,b.k""").to_arrow_table()
    np.testing.assert_allclose(
        got["v"].to_numpy().reshape(3, 5, 4),
        a @ b,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("mask", ["q.i >= k.j", "q.i < k.j + 2", "TRUE"])
@pytest.mark.parametrize("per_head", [True, False])
def test_attention_keeps_batches_and_heads_independent(
    accelerated, mask, per_head
):
    ctx, extension, tolerance = accelerated
    plain = Context()
    rng = np.random.default_rng(542)
    for name, shape, dims in [
        ("q", (2, 3, 7, 4), ("seq", "h", "i", "d")),
        ("k", (2, 3, 9, 4), ("seq", "h", "j", "d")),
        ("v", (2, 3, 9, 5), ("seq", "h", "j", "e")),
    ]:
        data = _matrix(rng.normal(size=shape), dims)
        ctx.from_dataset(name, data, chunks={"seq": 1})
        plain.from_dataset(name, data, chunks={"seq": 1})
    norm_keys = "seq,h,i" if per_head else "i"
    norm_join = "a.seq=n.seq AND a.h=n.h AND a.i=n.i" if per_head else "a.i=n.i"
    sql = f"""
    WITH scores AS (
      SELECT q.seq,q.h,q.i,k.j,SUM(q.v*k.v)/2 AS s FROM q JOIN k
      ON q.seq=k.seq AND q.h=k.h AND q.d=k.d WHERE {mask}
      GROUP BY q.seq,q.h,q.i,k.j),
    weights AS (
      SELECT seq,h,i,j,exp(s-MAX(s) OVER (PARTITION BY seq,h,i)) AS p
      FROM scores),
    norm AS (SELECT {norm_keys},SUM(p) AS z FROM weights GROUP BY {norm_keys})
    SELECT a.seq,a.h,a.i,v.e,SUM(a.p/n.z*v.v) AS v FROM weights a
    JOIN norm n ON {norm_join}
    JOIN v ON a.seq=v.seq AND a.h=v.h AND a.j=v.j
    GROUP BY a.seq,a.h,a.i,v.e ORDER BY a.seq,a.h,a.i,v.e
    """
    _same(
        ctx.sql(sql).to_arrow_table(),
        plain.sql(sql).to_arrow_table(),
        tolerance,
    )
    if extension.precision == "float32" and per_head:
        assert extension.statistics().get("gpu_attention", 0) > 0


def test_ddx_unsupported_expression_still_raises(accelerated):
    ctx, _, _ = accelerated
    ctx.from_dataset(
        "weights", xr.Dataset({"w": ("i", [1.2, 2.4])}, coords={"i": [0, 1]})
    )
    with pytest.raises(UnsupportedExpression, match="GROUP BY key"):
        ctx.sql("""WITH buckets AS (
          SELECT ROUND(w) AS k,SUM(w*w) AS s FROM weights GROUP BY ROUND(w)),
          loss AS (SELECT SUM(k*s) AS l FROM buckets)
          SELECT i,grad(loss,w) AS g FROM weights ORDER BY i""").collect()


def test_attention_backward_uses_ddx_derivatives(accelerated):
    ctx, _, tolerance = accelerated
    rng = np.random.default_rng(123)
    q, k, v = (rng.normal(size=(7, 4)) for _ in range(3))
    for name, values, dims in [
        ("q", q, ("i", "d")),
        ("k", k, ("j", "d")),
        ("v", v, ("j", "d")),
    ]:
        ctx.from_dataset(name, _matrix(values, dims))
    scores = q @ k.T / 2
    scores[np.triu_indices(7, 1)] = -np.inf
    weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    output = weights @ v
    dw = output @ v.T
    ds = weights * (dw - (dw * weights).sum(axis=1, keepdims=True))
    want = [ds @ k / 2, ds.T @ q / 2, weights.T @ output]
    model = ATTENTION.format(mask="q.i >= k.j")
    statements = [
        f"WITH attention AS ({model}), loss AS (SELECT SUM(v*v)/2 AS l FROM attention) "
        f"SELECT {key},d,grad(loss,v) AS v FROM {name} ORDER BY {key},d"
        for name, key in [("q", "i"), ("k", "j"), ("v", "j")]
    ]
    for frame, expected in zip(ad.sql_all(ctx, statements), want):
        np.testing.assert_allclose(
            frame.to_arrow_table()["v"].to_numpy().reshape(7, 4),
            expected,
            rtol=tolerance,
            atol=tolerance,
        )


def test_low_backend_memory_limit_retains_cpu_sql():
    extension = MLX(
        precision="float32", min_ops=0, min_gpu_ops=0, max_working_bytes=1
    )
    ctx = Context().with_extensions(extension)
    ctx.from_dataset("a", _matrix(np.ones((3, 4)), ("i", "j")))
    ctx.from_dataset("b", _matrix(np.ones((4, 5)), ("j", "k")))
    assert ctx.sql(PRODUCT).to_arrow_table()["v"].to_pylist() == [4.0] * 15
    assert extension.statistics() == {}
