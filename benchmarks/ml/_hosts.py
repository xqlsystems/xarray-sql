"""The SQL engines ("hosts") the ML demos run on.

einfold (https://github.com/xqlsystems/einfold), a query optimizer that
rewrites SQL plans into array algorithms, calls the engine that runs a
plan its *host*. Every demo writes
its model once, as SQL, and runs that same text on each host, so a
difference in time is a difference between engines, never between
queries. A host does five things:

* ``load(name, data, **options)`` registers an ``xr.Dataset`` as the
  table ``name`` through xarray-sql, replacing any table of that name
  (``options`` such as ``table_names`` go to xarray-sql);
* ``execute(sql)`` runs a statement for its effect, such as
  ``CREATE VIEW``;
* ``query(sql)`` runs a statement and returns its rows as Arrow;
* ``query_all(sqls)`` runs several statements, and, on a host that
  differentiates, shares the backward pass between those that take
  ``grad`` of the same loss;
* ``operators(sql)`` counts the operators in the physical plan it would
  run, which shows, for one, whether a CTE read twice is computed twice.

Forward queries run everywhere. Gradients (``grad``, ``vjp``, ``jvp``)
need ddx (https://github.com/xqlsystems/ddx), which differentiates SQL
queries and runs its programs on DataFusion and DuckDB; a host that
cannot differentiate says so with ``differentiates = False``, and the
demos report its gradient steps as not run rather than skip it silently.
A gradient that ddx (or the engine under it) refuses to plan raises one
of the host's ``refusals``, and the demos report that step with the reason.

Adding a host (einfold's optimizer on DataFusion, or DuckDB with a GPU
extension) is one class here and one entry in ``HOSTS``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import duckdb
import pyarrow as pa
import xarray as xr
from ddxdb import DdxError
from ddxdb import duckdb as ddx_duckdb

import xarray_sql as xql


class Host:
    """An engine that runs the demos' SQL. Subclasses fill in the rest."""

    name = ""
    differentiates = False
    refusals: tuple[type[Exception], ...] = ()

    def load(self, name: str, data: xr.Dataset, **options) -> None:
        raise NotImplementedError

    def execute(self, sql: str) -> None:
        raise NotImplementedError

    def query(self, sql: str) -> pa.Table:
        raise NotImplementedError

    def query_all(self, sqls: Sequence[str]) -> list[pa.Table]:
        return [self.query(sql) for sql in sqls]

    def operators(self, sql: str) -> int:
        raise NotImplementedError


class DataFusion(Host):
    """xarray-sql's ``Context``, with ddx's gradients (the ``ml`` extra)."""

    name = "datafusion"
    differentiates = True
    refusals = (DdxError,)

    def __init__(self):
        self.ctx = xql.Context()

    def load(self, name, data, **options):
        self.ctx.deregister_table(name)
        self.ctx.from_dataset(name, data, **options)

    def execute(self, sql):
        self.ctx.sql(sql).collect()

    def query(self, sql):
        return self.ctx.sql(sql).to_arrow_table()

    def query_all(self, sqls):
        return [df.to_arrow_table() for df in xql.ad.sql_all(self.ctx, sqls)]

    def operators(self, sql):
        plan = self.ctx.sql(sql).execution_plan().display_indent()
        return len(plan.splitlines())


class DuckDB(Host):
    """DuckDB, reading the same Datasets through xarray-sql's adapter, with
    ddx's gradients (``ddxdb[duckdb]``)."""

    name = "duckdb"
    differentiates = True
    # ddx reads DuckDB's plans as Substrait, a portable format for query
    # plans, and DuckDB's Substrait producer raises its own error for a
    # plan it can't write.
    refusals = (DdxError, duckdb.NotImplementedException)

    def __init__(self):
        self.con = duckdb.connect()

    def load(self, name, data, **options):
        xql.register(self.con, name, data, **options)

    def execute(self, sql):
        self.con.execute(sql)

    def query(self, sql):
        # As xarray-sql's Context does, send ddx only the statements that
        # take a gradient.
        if xql.ad.calls_ad(sql):
            return self.query_all([sql])[0]
        return self.con.sql(sql).fetch_arrow_table()

    def query_all(self, sqls):
        return [
            rel.fetch_arrow_table()
            for rel in ddx_duckdb.sql_all(self.con, sqls)
        ]

    def operators(self, sql):
        (_, plan), *_ = self.con.sql(f"EXPLAIN (FORMAT JSON) {sql}").fetchall()
        return sum(_count(node) for node in json.loads(plan))


def _count(node: dict) -> int:
    return 1 + sum(_count(child) for child in node.get("children", []))


HOSTS = {host.name: host for host in (DataFusion, DuckDB)}
