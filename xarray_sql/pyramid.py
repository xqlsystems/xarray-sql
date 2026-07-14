"""Multi-resolution pre-aggregated cubes from grid tables.

Registered xarray tables are *virtual*: every query re-streams the
source. Statistics that get asked repeatedly at varying resolutions
should pay the scan once — :func:`pyramid` builds a cube in the spirit
of CARTO's spatial-index tilesets: level 0 bins the source once (the
only expensive pass), coarser levels roll up from the level below, so
any zoom/extent query is a cheap range scan over a small table.

The helper dispatches through the engine adapter layer and works on
any connection :func:`xarray_sql.register` accepts — DuckDB and
DataFusion today. The SQL it issues (``CREATE OR REPLACE TABLE ... AS``,
``INSERT INTO``, ``FLOOR``) is deliberately restricted to what both
dialects share; ``aggs`` expressions you supply must be valid in the
connected engine's own dialect. (For a plain one-off cache of a query,
just write ``CREATE OR REPLACE TABLE ... AS ... ORDER BY ...`` in the
engine's SQL — see the performance guide.)
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

        (level, x_idx BIGINT, y_idx BIGINT, x_bin, y_bin, <aggs...>)

    ``x_idx``/``y_idx`` are exact integer cell indices at that level
    (they halve from level to level, so cell membership never drifts
    across levels); ``x_bin``/``y_bin`` are the float cell origins
    (``idx * base_cell * 2**level``) for human-readable querying —
    filter them with ranges rather than equality. Query the level whose
    cells match the resolution you need::

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

    # Cells are tracked as integer indices (x_idx, y_idx) and only
    # labeled with float origins (x_bin, y_bin) for querying: rebinning
    # float origins at each level occasionally lands boundary points in
    # a different cell than direct binning would (float aliasing);
    # integer indices halve exactly at every level.
    # Level 0: the single scan of the source.
    run_sql(
        con,
        f"""
        CREATE OR REPLACE TABLE {_ident(name)} AS
        SELECT
            0 AS level,
            x_idx,
            y_idx,
            x_idx * {base_cell} AS x_bin,
            y_idx * {base_cell} AS y_bin,
            {base_aggs}
        FROM (
            SELECT
                CAST(FLOOR({_ident(x)} / {base_cell}) AS BIGINT) AS x_idx,
                CAST(FLOOR({_ident(y)} / {base_cell}) AS BIGINT) AS y_idx,
                *
            FROM {_ident(table)}
            {where}
        )
        GROUP BY x_idx, y_idx
        """,
    )

    # Coarser levels roll up from the level below: cheap, source-free.
    for level in range(1, levels):
        cell = base_cell * (2**level)
        rollup_aggs = ", ".join(
            f"{_ROLLUP[kind]}({_ident(n)}) AS {_ident(n)}"
            for n, (kind, _) in aggs.items()
        )
        pass_cols = ", ".join(_ident(n) for n in aggs)
        run_sql(
            con,
            f"""
            INSERT INTO {_ident(name)}
            SELECT
                {level} AS level,
                x_idx,
                y_idx,
                x_idx * {cell} AS x_bin,
                y_idx * {cell} AS y_bin,
                {rollup_aggs}
            FROM (
                SELECT
                    CAST(FLOOR(x_idx / 2.0) AS BIGINT) AS x_idx,
                    CAST(FLOOR(y_idx / 2.0) AS BIGINT) AS y_idx,
                    {pass_cols}
                FROM {_ident(name)}
                WHERE level = {level - 1}
            )
            GROUP BY x_idx, y_idx
            """,
        )
    return con
