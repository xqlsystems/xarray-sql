"""Re-executable engine handles behind the lazy chunked round-trip.

The lazy path of ``to_dataset(chunks=...)`` re-executes the engine's
query per accessed chunk, narrowed to that chunk's coordinate window and
columns. That requires the engine result to be *re-executable* — a
handle onto the query, not a one-shot stream of its rows. Each handle
here adapts one engine's native lazy surface to the three operations the
reconstruction needs:

* :meth:`~LazyResultHandle.schema` — result column names/types, without
  executing the query;
* :meth:`~LazyResultHandle.distinct` — one column's distinct values
  (coordinate discovery; the caller sorts);
* :meth:`~LazyResultHandle.fetch` — the result narrowed by per-dimension
  windows and projected to the requested columns, as Arrow batches.

Windows are passed as :data:`DimSpec` values instead of rendered SQL so
each engine can express them with its own *typed* expression API —
strings would re-open every literal-formatting pitfall (timestamps,
floats, quoting) per dialect.

Handles compose with the registration seam: when the wrapped query
scans a Dataset registered through xarray-sql's pushdown machinery, the
per-chunk range filter flows back through the engine into
:class:`~xarray_sql.backends.pyarrow.XarrayPushdownDataset`, so each
output chunk's access reads only the source chunks it maps onto.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datafusion import col, literal

DimSpec = tuple[str, Any, Any]
"""One dimension's window: ``("range", lo, hi)`` (inclusive bounds; the
requested coordinate positions are contiguous) or ``("values", array,
None)`` (explicit value list, for stepped/fancy indexers)."""


def _plain(value: Any) -> Any:
    """A plain-Python literal (numpy scalars don't travel to engines)."""
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, np.timedelta64):
        return pd.Timedelta(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class LazyResultHandle(Protocol):
    """A re-executable query result (see module docstring)."""

    supports_chunked: bool = True
    """Whether fetch() may be driven from consumer worker threads (the
    chunked reconstruction). Handles for engines that cannot safely
    re-execute under foreign threads set this False; the eager path
    remains available."""

    def schema(self) -> pa.Schema: ...

    def distinct(self, column: str) -> np.ndarray: ...

    def fetch(
        self, specs: dict[str, DimSpec], columns: list[str]
    ) -> list[pa.RecordBatch]: ...

    def spill_parquet(self, path: str) -> None: ...


class DataFusionHandle:
    """Handle over a ``datafusion.DataFrame``."""

    supports_chunked = True

    def __init__(self, df: Any) -> None:
        self._df = df

    def schema(self) -> pa.Schema:
        return self._df.schema()

    def distinct(self, column: str) -> np.ndarray:
        dim_only = self._df.select(col(f'"{column}"')).distinct()
        batches = [b.to_pyarrow() for b in dim_only.execute_stream()]
        if not batches:
            return np.asarray([])
        return np.concatenate(
            [b.column(0).to_numpy(zero_copy_only=False) for b in batches]
        )

    def fetch(
        self, specs: dict[str, DimSpec], columns: list[str]
    ) -> list[pa.RecordBatch]:
        predicate = None
        for dim, (kind, a, b) in specs.items():
            c = col(f'"{dim}"')
            if kind == "range":
                p = (c >= literal(a)) & (c <= literal(b))
            else:
                # DataFusion 52.0.0 exposes no clean ``Expr.in_list``
                # from Python; OR-chained equalities constant-fold
                # equivalently and stay typed.
                p = c == literal(a[0])
                for v in a[1:]:
                    p = p | (c == literal(v))
            predicate = p if predicate is None else predicate & p
        out = self._df if predicate is None else self._df.filter(predicate)
        out = out.select(*(col(f'"{n}"') for n in columns))
        return [b.to_pyarrow() for b in out.execute_stream()]

    def spill_parquet(self, path: str) -> None:
        with pq.ParquetWriter(path, self.schema()) as writer:
            for batch in self._df.execute_stream():
                writer.write_batch(batch.to_pyarrow())


class DuckDBHandle:
    """Handle over a ``duckdb.DuckDBPyRelation``.

    Relations are lazy relational algebra: ``filter``/``project`` derive
    new relations and every materialization re-executes, which is
    exactly the re-executable contract. Predicates are built with
    DuckDB's typed expression API, never rendered SQL text.

    Every engine call runs on one dedicated thread owned by the handle.
    A relation is bound to one connection, and a query over a table
    registered through xarray-sql re-enters Python from DuckDB's
    execution threads (the Arrow scan callback); driving such queries
    directly from several consumer threads at once (dask computing
    output chunks of a lazy round-trip) deadlocks between the
    connection's serialization, the callback's need for the GIL, and
    the consumer pool's own thread management. Funnelling execution
    through a single pre-started thread reproduces the topology that is
    known safe — one thread inside the engine, every other thread
    parked on a GIL-releasing wait.
    """

    supports_chunked = False
    """Chunked (lazy) reconstruction is disabled for DuckDB relations.

    Windows of a chunked round-trip re-execute the relation from the
    consumer's worker threads (dask). A DuckDB query whose source is a
    Python-callback Arrow scan (any table registered through xarray-sql)
    intermittently deadlocks inside duckdb-python/CPython when other
    Python threads start or stop during execution — reproduced on
    duckdb 1.4-1.5 / CPython 3.12 / macOS at ~50% of runs, regardless
    of ``SET threads=1``, connection serialization, or pool pre-warming.
    Until that upstream race is fixed, chunked DuckDB round-trips fail
    fast instead of hanging; the eager path (and every other handle
    operation) runs on one dedicated thread and is unaffected.
    """

    def __init__(self, rel: Any) -> None:
        self._rel = rel
        self._runner = ThreadPoolExecutor(max_workers=1)
        self._runner.submit(lambda: None).result()  # start the thread now

    def _run(self, fn: Any) -> Any:
        return self._runner.submit(fn).result()

    @staticmethod
    def _to_arrow_table(rel: Any) -> pa.Table:
        if hasattr(rel, "to_arrow_table"):
            return rel.to_arrow_table()
        return rel.fetch_arrow_table()  # duckdb < 1.5

    @staticmethod
    def _to_arrow_reader(rel: Any) -> pa.RecordBatchReader:
        if hasattr(rel, "to_arrow_reader"):
            return rel.to_arrow_reader()
        return rel.fetch_record_batch()  # duckdb < 1.5

    def schema(self) -> pa.Schema:
        return self._run(
            lambda: self._to_arrow_table(self._rel.limit(0)).schema
        )

    def distinct(self, column: str) -> np.ndarray:
        import duckdb

        table = self._run(
            lambda: self._to_arrow_table(
                self._rel.project(duckdb.ColumnExpression(column)).distinct()
            )
        )
        return np.asarray(table.column(0).to_numpy(zero_copy_only=False))

    def fetch(
        self, specs: dict[str, DimSpec], columns: list[str]
    ) -> list[pa.RecordBatch]:
        import duckdb

        predicate = None
        for dim, (kind, a, b) in specs.items():
            c = duckdb.ColumnExpression(dim)
            if kind == "range":
                p = (c >= duckdb.ConstantExpression(_plain(a))) & (
                    c <= duckdb.ConstantExpression(_plain(b))
                )
            else:
                p = c.isin(*(duckdb.ConstantExpression(_plain(v)) for v in a))
            predicate = p if predicate is None else predicate & p
        rel = self._rel if predicate is None else self._rel.filter(predicate)
        rel = rel.project(*(duckdb.ColumnExpression(n) for n in columns))

        return cast(
            list[pa.RecordBatch],
            self._run(lambda: list(self._to_arrow_reader(rel))),
        )

    def spill_parquet(self, path: str) -> None:
        def run() -> None:
            reader = self._to_arrow_reader(self._rel)
            with pq.ParquetWriter(path, reader.schema) as writer:
                for batch in reader:
                    writer.write_batch(batch)

        self._run(run)


class PolarsHandle:
    """Handle over a ``polars.LazyFrame``.

    Per-window fetches run on the streaming engine, so a window read
    never materializes more than the window even when the frame scans
    an out-of-core source.
    """

    supports_chunked = True

    def __init__(self, lf: Any) -> None:
        self._lf = lf

    def schema(self) -> pa.Schema:
        import polars as pl

        return pl.DataFrame(schema=self._lf.collect_schema()).to_arrow().schema

    def distinct(self, column: str) -> np.ndarray:
        import polars as pl

        out = self._lf.select(pl.col(column).unique()).collect(
            engine="streaming"
        )
        return out.to_series().to_numpy()

    def fetch(
        self, specs: dict[str, DimSpec], columns: list[str]
    ) -> list[pa.RecordBatch]:
        import polars as pl

        exprs = []
        for dim, (kind, a, b) in specs.items():
            if kind == "range":
                exprs.append(pl.col(dim).is_between(_plain(a), _plain(b)))
            elif getattr(a, "dtype", None) is not None and a.dtype.kind == "f":
                # Upstream Polars translates float ``is_in`` literals
                # imprecisely (silently matching nothing); degenerate
                # ranges compare exactly. Reproduced on polars 1.42.
                vals = iter(a)
                expr = pl.col(dim).is_between(*(_plain(next(vals)),) * 2)
                for v in vals:
                    expr = expr | pl.col(dim).is_between(*(_plain(v),) * 2)
                exprs.append(expr)
            else:
                exprs.append(pl.col(dim).is_in([_plain(v) for v in a]))
        lf = self._lf.filter(*exprs) if exprs else self._lf
        out = lf.select([pl.col(n) for n in columns]).collect(
            engine="streaming"
        )
        return cast(list[pa.RecordBatch], out.to_arrow().to_batches())

    def spill_parquet(self, path: str) -> None:
        self._lf.sink_parquet(path)


def resolve_lazy_handle(result: Any) -> LazyResultHandle | None:
    """Adapt an engine result to a handle, or ``None`` if it is one-shot.

    Recognizes DuckDB relations, Polars lazy *and* eager frames (an
    eager frame re-executes trivially over its in-memory data), and
    DataFusion DataFrames. ``pyarrow`` tables/readers and bare
    ``__arrow_c_stream__`` objects are one-shot streams: there is no
    query to re-execute, so the lazy path cannot serve them.
    """
    root = type(result).__module__.split(".")[0]
    if root in ("duckdb", "_duckdb") and hasattr(result, "filter"):
        return DuckDBHandle(result)
    if root == "polars":
        import polars as pl

        if isinstance(result, pl.LazyFrame):
            return PolarsHandle(result)
        if isinstance(result, pl.DataFrame):
            return PolarsHandle(result.lazy())
    if hasattr(result, "execute_stream") and hasattr(result, "logical_plan"):
        return DataFusionHandle(result)
    return None
