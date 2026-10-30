"""Relations: SQL queries that read each other, composed like functions.

A model in SQL is a chain of intermediate results, each one a SELECT over
earlier ones. SQL names a result that is read more than once with a CTE
(``WITH name AS (...)``), and a model reads most of its results more than
once: a layer norm reads its input three times. So a model is one query
whose ``WITH`` clause lists every intermediate result in order.

Here each piece of a model is a function that takes relations and returns
one, as a layer of a neural network takes and returns a tensor::

    def residual(x: Relation, y: Relation) -> Relation:
        return rel("residual", f"SELECT ... FROM {x} x JOIN {y} y ON ...")

In an f-string, a relation stands for its name, so a SELECT reads it as it
would read a table. A relation remembers the relations its SQL reads, and
``ctes(output)`` lists every relation ``output`` reads, directly or not, in
an order where each comes after those it reads: the CTEs of a query that
computes ``output``. Each gets the name it was given, numbered when two
share one (``mean``, ``mean_2``), so the same model gives the same SQL.
"""

from __future__ import annotations

import itertools
import re
import weakref
from collections.abc import Iterable

# A relation's placeholder for its name, until ``ctes`` names it.
_PLACEHOLDER = re.compile(r"__relation_\d+__")
_counter = itertools.count()
_made: weakref.WeakValueDictionary[str, Relation] = (
    weakref.WeakValueDictionary()
)


class Relation:
    """One SELECT of a model, and the relations it reads."""

    def __init__(self, name: str, sql: str):
        self.name = name
        self.sql = sql
        self.reads = tuple(_made[p] for p in _PLACEHOLDER.findall(sql))
        self._placeholder = f"__relation_{next(_counter)}__"
        _made[self._placeholder] = self

    def __format__(self, spec: str) -> str:
        return self._placeholder


def rel(name: str, sql: str) -> Relation:
    """The relation ``sql``, to be named ``name`` (or ``name_2``, ...)."""
    return Relation(name, sql)


def ctes(output: Relation, taken: Iterable[str] = ()) -> list[tuple[str, str]]:
    """The CTEs, as ``(name, sql)`` pairs, of a query that reads
    ``output``: every relation it reads, then ``output`` itself. No CTE
    is named as a table in ``taken``, which it would hide."""
    order: list[Relation] = []
    seen: set[int] = set()

    def visit(r: Relation) -> None:
        if id(r) in seen:
            return
        seen.add(id(r))
        for read in r.reads:
            visit(read)
        order.append(r)

    visit(output)
    names: dict[str, str] = {}
    used = set(taken)
    for r in order:
        name, n = r.name, 1
        while name in used:
            n += 1
            name = f"{r.name}_{n}"
        used.add(name)
        names[r._placeholder] = name
    return [
        (names[r._placeholder], _PLACEHOLDER.sub(lambda m: names[m[0]], r.sql))
        for r in order
    ]
