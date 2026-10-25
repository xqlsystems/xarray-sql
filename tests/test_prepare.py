"""Tests for queries planned once and run many times (`Context.prepare`)."""

import numpy as np
import pytest
import xarray as xr
from datafusion import SessionConfig

from xarray_sql import Context

# The build side of the join, `a`, is what a reused hash join would keep.
JOIN = """
SELECT b.i, SUM(t.v2 * b.w) AS s
FROM (SELECT i, j, 2 * v AS v2 FROM a) t JOIN b ON t.j = b.j
GROUP BY b.i ORDER BY b.i
"""


def _a(v):
    return xr.Dataset(
        {"v": (("i", "j"), v)},
        coords={"i": np.arange(v.shape[0]), "j": np.arange(v.shape[1])},
    )


def _b(w):
    return xr.Dataset(
        {"w": (("i", "j"), w)},
        coords={"i": np.arange(w.shape[0]), "j": np.arange(w.shape[1])},
    )


@pytest.fixture
def ctx():
    rng = np.random.default_rng(0)
    ctx = Context()
    ctx.from_dataset("a", _a(rng.normal(size=(4, 3))))
    ctx.from_dataset("b", _b(rng.normal(size=(4, 3))))
    return ctx


def test_a_prepared_query_returns_the_query_rows(ctx):
    query = ctx.prepare(JOIN)
    want = ctx.sql(JOIN).to_arrow_table()
    assert query().equals(want)
    assert query().equals(want)


def test_a_call_reads_the_values_it_writes(ctx):
    query = ctx.prepare(JOIN)
    rng = np.random.default_rng(1)
    for _ in range(3):
        a, b = _a(rng.normal(size=(4, 3))), _b(rng.normal(size=(4, 3)))
        rows = query(a=a, b=b)
        fresh = Context()
        fresh.from_dataset("a", a)
        fresh.from_dataset("b", b)
        assert rows.equals(fresh.sql(JOIN).to_arrow_table())


def test_a_write_changes_the_registered_dataset():
    v = np.zeros((4, 3))
    ctx = Context()
    ctx.from_dataset("a", _a(v))
    query = ctx.prepare("SELECT SUM(v) AS s FROM a")
    assert query(a=_a(np.ones((4, 3)))).column("s").to_pylist() == [12.0]
    assert (v == 1).all()
    assert ctx.sql("SELECT SUM(v) AS s FROM a").to_pydict() == {"s": [12.0]}


def test_a_write_matches_dims_by_name(ctx):
    query = ctx.prepare(JOIN)
    rng = np.random.default_rng(2)
    a = _a(rng.normal(size=(4, 3)))
    want = query(a=a)
    assert query(a=a.transpose("j", "i")).equals(want)


def test_a_write_reaches_a_table_read_lazily():
    """An array that a table reads on every scan, rather than sharing (here,
    one in Fortran order), takes writes too."""
    v = np.asfortranarray(np.zeros((4, 3)))
    ctx = Context()
    ctx.from_dataset("a", _a(v))
    query = ctx.prepare("SELECT SUM(v) AS s FROM a")
    assert query(a=_a(np.ones((4, 3)))).column("s").to_pylist() == [12.0]


def test_a_plan_with_several_partitions_runs_again():
    ctx = Context(SessionConfig().with_target_partitions(4))
    n = 64
    ctx.from_dataset("a", _a(np.ones((n, 3))), chunks={"i": n // 4})
    ctx.from_dataset("b", _b(np.ones((n, 3))), chunks={"i": n // 4})
    query = ctx.prepare(
        "SELECT a.i, SUM(v * w) AS s FROM a JOIN b ON a.i = b.i AND a.j = b.j "
        "GROUP BY a.i"
    )
    for scale in (1.0, 2.0):
        rows = query(a=_a(np.full((n, 3), scale)))
        assert sorted(rows.column("s").to_pylist()) == [3 * scale] * n


def test_a_lazy_table_is_planned_on_each_call():
    pytest.importorskip("dask")
    ctx = Context()
    ctx.from_dataset("a", _a(np.ones((4, 3))).chunk({"i": 2}))
    query = ctx.prepare("SELECT SUM(v) AS s FROM a")
    assert query().column("s").to_pylist() == [12.0]
    assert query().column("s").to_pylist() == [12.0]
    with pytest.raises(ValueError, match="NumPy"):
        query(a=_a(np.ones((4, 3))))


@pytest.mark.parametrize(
    "new, match",
    [
        (_a(np.ones((4, 2))), "coordinates"),
        (_a(np.ones((4, 3), dtype=np.float32)), "float32"),
        (_a(np.ones((4, 3))).rename({"v": "u"}), "data variables"),
        (_a(np.full((4, 3), np.nan)), "NaN"),
    ],
)
def test_a_write_must_keep_the_table_shape(ctx, new, match):
    query = ctx.prepare(JOIN)
    want = query()
    with pytest.raises(ValueError, match=match):
        query(a=new)
    assert query().equals(want)


def test_a_write_needs_the_table_the_query_was_planned_with(ctx):
    query = ctx.prepare(JOIN)
    with pytest.raises(ValueError, match="no Dataset"):
        query(c=_a(np.ones((4, 3))))
    ctx.deregister_table("a")
    ctx.from_dataset("a", _a(np.ones((4, 3))))
    with pytest.raises(ValueError, match="registered again"):
        query(a=_a(np.ones((4, 3))))
