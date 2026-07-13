"""One-time scans into native engine tables: caches and pyramids.

Both helpers dispatch through the engine adapter layer and work on
any connection :func:`xarray_sql.register` accepts — DuckDB and
DataFusion today. The SQL they issue (``CREATE OR REPLACE TABLE ...
AS``, ``INSERT INTO``, ``FLOOR``) is deliberately restricted to what
both dialects share; ``query``/``aggs`` expressions you supply must be
valid in the connected engine's own dialect.

Registered xarray tables are *virtual*: every query re-streams the
source. That is the right default for exploration, but statistics that
get asked repeatedly should pay the scan once. Two helpers formalize
the pattern:

* :func:`materialize` — run a query once into a native engine table,
  ordered so the repetitive coordinate columns compress (DuckDB picks
  ALP/RLE automatically on sorted data) and zone maps prune range
  predicates.
* :func:`pyramid` — a multi-resolution pre-aggregated cube in the
  spirit of CARTO's spatial-index tilesets: level 0 bins the source
  once (the only expensive pass), coarser levels roll up from the level
  below, so any zoom/extent query is a cheap range scan over a small
  table.
"""

from __future__ import annotations

from typing import Any

from .backends.base import get_adapter

AggKind = str
"""One of ``"sum"``, ``"count"``, ``"min"``, ``"max"``.

Pyramid aggregates must be decomposable so coarser levels can roll up
from finer ones without rescanning the source. Averages are derived at
query time from a sum and a count.
"""

_ROLLUP = {"sum": "SUM", "count": "SUM", "min": "MIN", "max": "MAX"}
_BASE = {"sum": "SUM", "count": "COUNT", "min": "MIN", "max": "MAX"}


def _ident(name: str) -> str:
    """Quote a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def materialize(
    con: Any,
    name: str,
    query: str,
    *,
    order_by: list[str] | None = None,
) -> Any:
    """Run *query* once into a native engine table named *name*.

    The one-time scan cost buys native-speed re-querying: e.g. DuckDB
    storage compresses the repetitive coordinate columns (ALP/RLE) and
    prunes range predicates with zone maps — both work best when the
    table is written in coordinate order, so pass ``order_by`` with the
    dimension columns whenever the query preserves them.

    Example::

        xql.register(con, "klass", ds)
        xql.materialize(
            con, "grid_cube",
            "SELECT FLOOR(y) AS lat, FLOOR(x) AS lon, klass, COUNT(*) AS n "
            "FROM grid GROUP BY 1, 2, 3",
            order_by=["lat", "lon"],
        )
        con.sql("SELECT * FROM grid_cube WHERE lat = -32")  # instant

    Args:
        con: An engine connection supported by
            :func:`xarray_sql.register` (DuckDB, DataFusion).
        name: Name of the table to create (replaced if it exists).
        query: Any SELECT statement in the engine's dialect, typically
            over a registered xarray table.
        order_by: Columns to sort the stored table by.

    Returns:
        The connection, to allow chaining.
    """
    sql = f"CREATE OR REPLACE TABLE {_ident(name)} AS SELECT * FROM ({query})"
    if order_by:
        sql += " ORDER BY " + ", ".join(_ident(c) for c in order_by)
    get_adapter(con).run_sql(con, sql)
    return con


def pyramid(
    con: Any,
    name: str,
    table: str,
    *,
    aggs: dict[str, tuple[AggKind, str]],
    base_cell: float,
    levels: int,
    x: str = "x",
    y: str = "y",
    filter: str | None = None,
) -> Any:
    """Build a multi-resolution pre-aggregated cube from a grid table.

    The source is scanned exactly once, binning ``x``/``y`` into square
    cells of ``base_cell`` coordinate units (level 0). Each coarser
    level doubles the cell size and rolls up from the level below, so
    the total cost beyond the single scan is negligible. The result is
    one long table::

        (level INTEGER, x_bin DOUBLE, y_bin DOUBLE, <agg columns...>)

    where ``x_bin``/``y_bin`` are the cell origins at that level's cell
    size (``base_cell * 2**level``). Query the level whose cells match
    the resolution you need::

        SELECT x_bin, y_bin, class4_n / n AS share
        FROM grid_pyramid
        WHERE level = 3 AND x_bin BETWEEN -66 AND -63

    Aggregates must be decomposable (see :data:`AggKind`); express an
    average as a ``sum`` plus a ``count`` and divide at query time.

    Args:
        con: An engine connection supported by
            :func:`xarray_sql.register` (DuckDB, DataFusion).
        name: Name of the pyramid table to create (replaced if it
            exists).
        table: The source table (typically a registered xarray table).
        aggs: Mapping from output column name to ``(kind, expression)``,
            e.g. ``{"n": ("count", "*"), "class4_n": ("sum", "CASE WHEN
            klass >= 4 THEN 1 ELSE 0 END")}``.
        base_cell: Cell size of level 0, in coordinate units.
        levels: Number of levels to build (level 0 .. levels - 1).
        x: Name of the x/longitude column in ``table``.
        y: Name of the y/latitude column in ``table``.
        filter: Optional SQL predicate applied while scanning the
            source (level 0 only) — e.g. a bounding box, which also
            prunes the scan itself.

    Returns:
        The connection, to allow chaining.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    bad = [k for k, (kind, _) in aggs.items() if kind not in _BASE]
    if bad:
        raise ValueError(
            f"Aggregates {bad} have unsupported kinds; expected one of "
            f"{sorted(_BASE)}. Express averages as a sum plus a count."
        )

    base_aggs = ", ".join(
        f"{_BASE[kind]}({expr}) AS {_ident(n)}"
        for n, (kind, expr) in aggs.items()
    )
    where = f"WHERE {filter}" if filter else ""

    run_sql = get_adapter(con).run_sql

    # Level 0: the single scan of the source.
    run_sql(
        con,
        f"""
        CREATE OR REPLACE TABLE {_ident(name)} AS
        SELECT
            0 AS level,
            FLOOR({_ident(x)} / {base_cell}) * {base_cell} AS x_bin,
            FLOOR({_ident(y)} / {base_cell}) * {base_cell} AS y_bin,
            {base_aggs}
        FROM {_ident(table)}
        {where}
        GROUP BY 2, 3
        """,
    )

    # Coarser levels roll up from the level below: cheap, source-free.
    for level in range(1, levels):
        cell = base_cell * (2**level)
        rollup_aggs = ", ".join(
            f"{_ROLLUP[kind]}({_ident(n)}) AS {_ident(n)}"
            for n, (kind, _) in aggs.items()
        )
        run_sql(
            con,
            f"""
            INSERT INTO {_ident(name)}
            SELECT
                {level} AS level,
                FLOOR(x_bin / {cell}) * {cell} AS x_bin,
                FLOOR(y_bin / {cell}) * {cell} AS y_bin,
                {rollup_aggs}
            FROM {_ident(name)}
            WHERE level = {level - 1}
            GROUP BY 2, 3
            """,
        )
    return con
