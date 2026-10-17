"""Gradients of SQL queries: the optional ml extension, built on ddx.

[ddx](https://github.com/xqlsystems/ddx) differentiates a whole SQL query
the way JAX differentiates a function. With it installed
(``pip install xarray-sql[ml]``), [xarray_sql.Context][]'s ``.sql()``
understands three calls in a select list:

* ``grad(loss, column)``: at each row, the derivative of the one number
  the CTE ``loss`` computes with respect to that row's ``column``
  (``table.column`` when the ``SELECT`` reads several tables).
* ``vjp(f, column, cotangent)``: the same for any CTE ``f``, pulled back
  along ``cotangent``, a table or CTE shaped like ``f``'s output.
* ``jvp(e, table.column, tangent)``: the tangent of the expression ``e``
  along ``tangent``, pushed forward.

A gradient-descent step on a parameter table is then one statement::

    WITH pred AS (
      SELECT x.sample, SUM(x.value * w.val) AS y
      FROM x JOIN w ON x.feature = w.feature GROUP BY x.sample),
    loss AS (
      SELECT AVG(power(pred.y - t.value, 2)) AS l
      FROM pred JOIN t ON pred.sample = t.sample)
    SELECT feature, val - 0.1 * grad(loss, val) AS val FROM w

The query being differentiated is ordinary SQL, over tables registered
with ``from_dataset`` or any other way. ddx plans the backward pass as
Substrait, DataFusion runs it, and the statement reads what it computed.

The one function ddx gives a meaning to is ``ddx_stop_gradient(x)``, the
identity whose derivative is zero (JAX's ``lax.stop_gradient``). It is
registered on every ``Context``. Its typical use is the stable softmax,
whose shift by the row maximum cancels and so carries no gradient::

    exp(s - ddx_stop_gradient(MAX(s) OVER (PARTITION BY i)))

A statement that calls none of ``grad``, ``vjp`` or ``jvp`` is planned
exactly as it would be without ddx. An expression ddx cannot
differentiate raises a ``ddxdb.DdxError`` subclass naming it, never a
wrong number.

[sql_all][xarray_sql.ad.sql_all] runs several statements that take
``grad`` of the same loss for the price of one backward pass, which is
how a training step updates each of a model's parameter tables. For the
programs underneath (``grad``, ``vjp``, ``jvp`` and ``run``, to build a
backward pass once and run it every step), use ``ddxdb.ad`` directly;
they take a ``Context`` like any other ``SessionContext``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from datafusion import SessionContext
from ddxdb import ad as _ddx

__all__ = ["STOP_GRADIENT", "calls_ad", "register", "sql", "sql_all"]

STOP_GRADIENT = _ddx.STOP_GRADIENT
"""The SQL name of the stop-gradient function, ``ddx_stop_gradient``."""

_AD_CALL = re.compile(r"\b(grad|vjp|jvp)\s*\(", re.IGNORECASE)


def register(ctx: SessionContext) -> None:
    """Register ``ddx_stop_gradient`` on ``ctx``.

    Works on any DataFusion ``SessionContext`` (``Context`` registers it
    automatically when ddxdb is installed). ``grad``, ``vjp`` and ``jvp``
    are not UDFs: [sql][xarray_sql.ad.sql] rewrites them before DataFusion
    plans the statement.

    Args:
        ctx: The DataFusion session context to register the UDF on.
    """
    _ddx.register_stop_gradient(ctx)


def calls_ad(statement: str) -> bool:
    """Whether ``statement`` may call ``grad``, ``vjp`` or ``jvp``.

    A cheap textual check, so statements without a call never reach
    ddx's parser. A false positive (the word inside a string literal,
    say) is harmless: ddx plans a statement with no call as it is.
    """
    return _AD_CALL.search(statement) is not None


def sql(ctx: SessionContext, statement: str):
    """Run ``statement``, computing its ``grad``, ``vjp`` and ``jvp`` calls.

    Returns a DataFusion DataFrame; ``Context.sql`` wraps it as an
    [XarrayDataFrame][xarray_sql.ds.XarrayDataFrame].

    Args:
        ctx: Any DataFusion ``SessionContext``.
        statement: A SQL statement.
    """
    return _ddx.sql(ctx, statement)


def sql_all(ctx: SessionContext, statements: Sequence[str]) -> list:
    """Run several statements, sharing the programs their calls need.

    Statements that take ``grad`` of the same loss share one backward
    pass, so a training step can update each parameter table with its
    own statement and pay for the gradient once::

        steps = xql.ad.sql_all(ctx, [
            f"WITH loss AS ({LOSS}) "
            "SELECT i, val - 0.1 * grad(loss, val) AS val FROM w1",
            f"WITH loss AS ({LOSS}) "
            "SELECT j, val - 0.1 * grad(loss, val) AS val FROM w2",
        ])

    On a [Context][xarray_sql.Context], each result is an
    [XarrayDataFrame][xarray_sql.ds.XarrayDataFrame].

    Args:
        ctx: Any DataFusion ``SessionContext``.
        statements: SQL statements.

    Returns:
        One DataFrame per statement, in order.
    """
    if isinstance(statements, str):
        raise TypeError("statements is a list of SQL statements")
    frames = _ddx.sql_all(ctx, list(statements))
    wrap = getattr(ctx, "_wrap", None)
    return [wrap(f) for f in frames] if wrap else frames
