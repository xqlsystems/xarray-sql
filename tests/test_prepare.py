"""Tests for queries planned once and run many times (`Context.prepare`)."""

import time

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


def _times(*days):
    t = np.array(
        [np.datetime64("NaT") if d is None else np.datetime64(d) for d in days],
        dtype="datetime64[ns]",
    )
    return xr.Dataset({"t": ("i", t)}, coords={"i": np.arange(len(t))})


def test_a_write_must_keep_nat_in_place():
    ctx = Context()
    ctx.from_dataset("a", _times("2000-01-01", None, "2000-01-03"))
    query = ctx.prepare("SELECT COUNT(t) AS n, MIN(t) AS m FROM a")
    with pytest.raises(ValueError, match="NaT"):
        query(a=_times(None, "2000-01-02", "2000-01-03"))
    rows = query(a=_times("2000-01-04", None, "2000-01-02")).to_pydict()
    assert rows["n"] == [2]
    assert str(rows["m"][0]) == "2000-01-02 00:00:00"


def test_a_query_reads_its_tables_as_they_were_prepared():
    pytest.importorskip("dask")
    ctx = Context()
    ctx.from_dataset("a", _a(np.ones((4, 3))).chunk({"i": 2}))
    query = ctx.prepare("SELECT SUM(v) AS s FROM a")
    ctx.deregister_table("a")
    ctx.from_dataset("a", _a(np.full((4, 3), 2.0)))
    with pytest.raises(ValueError, match="registered again"):
        query()


def test_a_table_replaced_by_datafusion_counts_as_registered_again(ctx):
    query = ctx.prepare(JOIN)
    ctx.deregister_table("a")
    ctx.register_record_batches("a", [ctx.sql("SELECT * FROM b").collect()])
    with pytest.raises(ValueError, match="registered again"):
        query()


def test_a_table_the_query_doesnt_read_can_be_registered_again(ctx):
    query = ctx.prepare("SELECT SUM(v) AS s FROM a")
    want = query()
    ctx.deregister_table("b")
    ctx.from_dataset("b", _b(np.ones((4, 3))))
    assert query().equals(want)


def test_now_is_the_time_the_query_was_prepared(ctx):
    query = ctx.prepare("SELECT now() AS t")
    first = query()
    time.sleep(0.01)
    assert query().equals(first)


def test_a_write_must_keep_every_coordinate(ctx):
    query = ctx.prepare(JOIN)
    # A coordinate along both dims, which neither dim's coordinate holds.
    labelled = _a(np.ones((4, 3))).assign_coords(
        area=(("i", "j"), np.ones((4, 3)))
    )
    with pytest.raises(ValueError, match="coordinates"):
        query(a=labelled)


def test_a_table_not_from_a_dataset_counts_too(ctx):
    ctx.register_record_batches("c", [ctx.sql("SELECT i, j FROM a").collect()])
    query = ctx.prepare("SELECT COUNT(*) AS n FROM a JOIN c USING (i, j)")
    assert query().column("n").to_pylist() == [12]
    ctx.deregister_table("c")
    ctx.register_record_batches("c", [ctx.sql("SELECT i, j FROM b").collect()])
    with pytest.raises(ValueError, match="registered again"):
        query()


def test_now_is_the_time_a_query_over_a_lazy_table_was_prepared():
    pytest.importorskip("dask")
    ctx = Context()
    ctx.from_dataset("a", _a(np.ones((4, 3))).chunk({"i": 2}))
    query = ctx.prepare("SELECT now() AS t, SUM(v) AS s FROM a")
    first = query()
    time.sleep(0.01)
    assert query().equals(first)


def test_a_table_replaced_in_sql_counts_too():
    ctx = Context()
    ctx.sql("CREATE TABLE c AS VALUES (1), (2)").collect()
    query = ctx.prepare("SELECT COUNT(*) AS n FROM c")
    assert query().column("n").to_pylist() == [2]
    ctx.sql("CREATE OR REPLACE TABLE c AS VALUES (1), (2), (3)").collect()
    with pytest.raises(ValueError, match="registered again"):
        query()
