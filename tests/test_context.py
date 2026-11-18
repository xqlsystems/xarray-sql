"""Tests for what a Context records about its tables and functions:
`datasets`, `write`, `replacements`, `registered_functions` and
`tables_read`."""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import xarray as xr
from datafusion import udaf, udf, udwf
from datafusion.user_defined import Accumulator, WindowEvaluator

from xarray_sql import Context


def _table(v):
    return xr.Dataset({"v": ("i", v)}, coords={"i": np.arange(len(v))})


@pytest.fixture
def ctx():
    ctx = Context()
    ctx.from_dataset("a", _table(np.arange(3.0)))
    ctx.from_dataset("b", _table(np.arange(3.0)))
    return ctx


def _values(ctx, table):
    rows = ctx.sql(f"SELECT v FROM {table} ORDER BY i").to_arrow_table()
    return rows["v"].to_pylist()


def test_datasets_are_the_registered_datasets_and_read_only():
    ctx = Context()
    a = _table(np.arange(3.0))
    ctx.from_dataset("a", a)
    assert ctx.datasets["a"] is a
    with pytest.raises(TypeError):
        ctx.datasets["b"] = a
    with pytest.raises(Exception, match="already exists"):
        ctx.from_dataset("a", _table(np.zeros(3)))
    assert ctx.datasets["a"] is a


def test_a_write_reaches_queries_and_datasets_but_not_replacements(ctx):
    ctx.write({"a": _table(np.array([7.0, 8.0, 9.0]))})
    assert _values(ctx, "a") == [7.0, 8.0, 9.0]
    assert ctx.datasets["a"]["v"].values.tolist() == [7.0, 8.0, 9.0]
    assert dict(ctx.replacements) == {}


@pytest.mark.parametrize(
    "bad, match", [(np.arange(3.0), "float64"), (np.arange(5), "'k': 5")]
)
def test_a_write_that_fails_a_check_writes_no_table(bad, match):
    ctx = Context()
    ctx.from_dataset("a", xr.Dataset({"v": ("i", np.arange(3))}))
    ctx.from_dataset("b", xr.Dataset({"v": ("k", np.arange(3))}))
    with pytest.raises(ValueError, match=match):
        ctx.write(
            {
                "a": xr.Dataset({"v": ("i", np.arange(7, 10))}),
                "b": xr.Dataset({"v": ("k", bad)}),
            }
        )
    assert _values(ctx, "a") == [0, 1, 2]


def test_a_write_needs_a_registered_dataset(ctx):
    ctx.from_pydict({"v": [1.0]}, "c")
    with pytest.raises(ValueError, match="no Dataset is registered as 'c'"):
        ctx.write({"c": _table(np.array([2.0]))})


@pytest.mark.parametrize(
    "change, name",
    [
        (lambda ctx: ctx.deregister_table("a"), "a"),
        (lambda ctx: ctx.sql("CREATE OR REPLACE TABLE a AS SELECT 1"), "a"),
        (lambda ctx: ctx.sql("DROP TABLE public.a"), "a"),
        (lambda ctx: ctx.sql("CREATE OR REPLACE VIEW v AS SELECT 2"), "v"),
        (lambda ctx: ctx.sql('DROP VIEW "v"'), "v"),
    ],
)
def test_replacements_count_each_table_or_view_replaced(ctx, change, name):
    ctx.sql("CREATE VIEW v AS SELECT v FROM b")
    change(ctx)
    assert dict(ctx.replacements) == {name: 1}


def _split(v):
    return xr.Dataset(
        {"v": ("i", v), "t": (("i", "j"), np.ones((len(v), 2)))},
        coords={"i": np.arange(len(v)), "j": np.arange(2)},
    )


def test_registering_a_split_dataset_again_replaces_each_table():
    ctx = Context()
    ctx.from_dataset("g", _split(np.arange(3.0)))
    assert dict(ctx.replacements) == {}
    ctx.from_dataset("g", _split(np.array([7.0, 8.0, 9.0])))
    assert dict(ctx.replacements) == {"g.i": 1, "g.i_j": 1}
    assert _values(ctx, "g.i") == [7.0, 8.0, 9.0]


def test_creating_a_table_or_view_replaces_nothing(ctx):
    ctx.sql("CREATE OR REPLACE VIEW v AS SELECT v FROM b")
    ctx.sql("CREATE TABLE c AS SELECT 1")
    assert dict(ctx.replacements) == {}


def _identity():
    return udf(lambda x: x, [pa.float64()], pa.float64(), "immutable", "Ident")


class _Sum(Accumulator):
    def __init__(self):
        self._sum = 0.0

    def update(self, values):
        self._sum += pc.sum(values).as_py() or 0.0

    def merge(self, sums):
        self._sum += pc.sum(sums).as_py() or 0.0

    def state(self):
        return [pa.scalar(self._sum)]

    def evaluate(self):
        return pa.scalar(self._sum)


class _Same(WindowEvaluator):
    def evaluate_all(self, values, num_rows):
        return values[0]


def test_registered_functions_holds_each_function_registered(ctx):
    ident = _identity()
    total = udaf(
        _Sum, pa.float64(), pa.float64(), [pa.float64()], "immutable", "sum"
    )
    same = udwf(_Same, pa.float64(), pa.float64(), "immutable", "same")
    ctx.register_udf(ident)
    ctx.register_udaf(total)
    ctx.register_udwf(same)
    assert {
        name: ctx.registered_functions[name]
        for name in ("Ident", "sum", "same")
    } == {"Ident": ident, "sum": total, "same": same}
    assert ctx.registered_functions.keys().isdisjoint({"avg", "abs"})
    assert ctx.sql('SELECT "Ident"(v) AS v FROM a ORDER BY i').to_arrow_table()[
        "v"
    ].to_pylist() == [0.0, 1.0, 2.0]


def test_registering_a_name_again_replaces_its_function(ctx):
    ctx.register_udf(_identity())
    again = _identity()
    ctx.register_udf(again)
    assert ctx.registered_functions["Ident"] is again
    ctx.deregister_udf("Ident")
    assert "Ident" not in ctx.registered_functions


def test_an_extension_handle_shares_what_the_context_records(ctx):
    handle = ctx.with_extensions()
    handle.register_udf(_identity())
    handle.deregister_table("b")
    assert "Ident" in ctx.registered_functions
    assert dict(ctx.replacements) == {"b": 1}
    assert set(ctx.datasets) == {"a"}


@pytest.mark.parametrize(
    "query, read",
    [
        ("SELECT v FROM public.a", {"a"}),
        ("SELECT x.v FROM a AS x JOIN b AS y ON x.i = y.i", {"a", "b"}),
        ("SELECT v FROM a WHERE v IN (SELECT v FROM b)", {"a", "b"}),
        ("SELECT v, (SELECT max(v) FROM b) AS m FROM a", {"a", "b"}),
        (
            "SELECT v FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.i = a.i)",
            {"a", "b"},
        ),
        ("SELECT v FROM w", {"w", "v", "b"}),
        ("SELECT x.v FROM datafusion.public.v AS x", {"v", "b"}),
        ('SELECT v FROM "x.y"', {"x.y", "b"}),
        ("SELECT v FROM datafusion.g.i", {"g.i"}),
    ],
)
def test_tables_read_lists_tables_and_views_in_subqueries_too(ctx, query, read):
    ctx.sql("CREATE VIEW v AS SELECT v FROM b")
    ctx.sql("CREATE VIEW w AS SELECT v FROM v")
    ctx.sql('CREATE VIEW "x.y" AS SELECT v FROM b')
    ctx.from_dataset("g", _split(np.arange(3.0)))
    assert ctx.tables_read(ctx.sql(query).logical_plan()) == read
