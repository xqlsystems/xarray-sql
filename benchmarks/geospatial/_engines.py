"""Engine-portable SQL layer for the geospatial suite.

``GEOBENCH_ENGINE`` selects which SQL engine executes each case's query,
so the same case scripts (same SQL, same datasets, same correctness
assertions) can be measured across engines:

* ``datafusion`` (default) — ``xql.XarrayContext`` over the native
  DataFusion table provider, the suite's original path. Requires the
  compiled ``xarray_sql._native`` module; raises at startup when it is
  missing instead of falling back.
* ``datafusion-arrow`` — a plain ``datafusion.SessionContext`` scanning
  ``xql.arrow_dataset`` (pure Python).
* ``duckdb`` — DuckDB over the same pyarrow pushdown datasets.
* ``polars`` — ``polars.SQLContext`` over ``scan_pyarrow_dataset`` frames.

Every case builds one :class:`EngineContext`, registers datasets exactly
as it always registered them on ``XarrayContext``, and calls
:meth:`EngineContext.sql_to_dataset`. On the ``datafusion`` path this
is byte-for-byte the original behavior (``from_dataset`` + ``sql`` +
``XarrayDataFrame.to_dataset``); the other engines register one pyarrow
dataset per dimension group under flattened table names
(``era5.surface`` → ``era5_surface`` — rewritten in the SQL text) and the
result rows are round-tripped to an ``xr.Dataset`` through pandas.

The DataFusion-only UDF cases (07 and the UDF half of 09) build
``xql.XarrayContext`` directly rather than through this layer; the suite
runner records them as n/a for every engine except ``datafusion``.
"""

from __future__ import annotations

import datetime
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def engine_name() -> str:
    """The engine selected for this process (``GEOBENCH_ENGINE``)."""
    engine = os.environ.get("GEOBENCH_ENGINE", "datafusion")
    if engine not in _ENGINES:
        raise ValueError(f"GEOBENCH_ENGINE={engine!r}; expected {_ENGINES}")
    return engine


def _group_tables(name, ds, table_names):
    """Split ``ds`` into per-dimension-group tables like XarrayContext does.

    Returns ``[(flat_name, dotted_name, sub_dataset)]``; a uniform dataset
    keeps its plain name (flat == dotted == name).
    """
    groups: dict[tuple, list] = {}
    for var, v in ds.data_vars.items():
        groups.setdefault(tuple(v.dims), []).append(var)
    if len(groups) == 1:
        return [(name, name, ds)]
    out = []
    for dims, variables in groups.items():
        sub = (table_names or {}).get(dims) or "_".join(dims)
        out.append((f"{name}_{sub}", f"{name}.{sub}", ds[variables]))
    return out


def _literal(value: Any) -> str:
    """Render a parameter value as a SQL literal (for engines without binds)."""
    if isinstance(value, (datetime.datetime, pd.Timestamp, np.datetime64)):
        return (
            f"TIMESTAMP '{pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S')}'"
        )
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return repr(value)


def _to_ns(pdf: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Normalize datetime/timedelta dim columns to ns for label alignment."""
    for col in dims:
        dtype = pdf[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            pdf[col] = pdf[col].astype("datetime64[ns]")
        elif pd.api.types.is_timedelta64_dtype(dtype):
            pdf[col] = pdf[col].astype("timedelta64[ns]")
    return pdf


def _pandas_to_dataset(pdf: pd.DataFrame, dims: list[str]) -> xr.Dataset:
    """Round-trip a SQL result table to a gridded ``xr.Dataset`` by ``dims``."""
    pdf = _to_ns(pdf.copy(), dims)
    return xr.Dataset.from_dataframe(pdf.set_index(dims).sort_index())


class EngineContext:
    """Uniform register-and-query facade over the suite's SQL engines.

    ``EngineContext(engine)`` instantiates the subclass ``_IMPLS`` maps
    the engine name to (default: :func:`engine_name`). Subclasses set
    ``flavor`` and implement three hooks: ``_connect`` (open the
    engine's connection/context), ``_register`` (attach one pyarrow
    dataset under a flat table name), and ``_execute`` (run SQL,
    returning a ``pandas.DataFrame``). Engines that bypass the shared
    pyarrow-dataset path override :meth:`from_dataset` /
    :meth:`sql_to_dataset` instead.
    """

    flavor = ""

    def __new__(cls, engine: str | None = None):
        if cls is EngineContext:
            cls = _IMPLS[engine or engine_name()]
        return super().__new__(cls)

    def __init__(self, engine: str | None = None):
        self.engine = engine or engine_name()
        self._renames: dict[str, str] = {}
        self._connect()

    def _connect(self) -> None:
        raise NotImplementedError

    def _register(self, flat: str, dataset) -> None:
        raise NotImplementedError

    def _execute(self, sql: str, param_values) -> pd.DataFrame:
        raise NotImplementedError

    # -- registration -----------------------------------------------------

    def from_dataset(self, name, ds, *, chunks=None, table_names=None):
        """Register ``ds`` as SQL table(s), mirroring XarrayContext naming."""
        import xarray_sql as xql

        for flat, dotted, sub in _group_tables(name, ds, table_names):
            if dotted != flat:
                self._renames[dotted] = flat
            sub_chunks = (
                {d: c for d, c in chunks.items() if d in sub.dims}
                if isinstance(chunks, dict)
                else chunks
            ) or None
            self._register(flat, xql.arrow_dataset(sub, sub_chunks))

    # -- querying ----------------------------------------------------------

    def _rewrite(self, sql: str, param_values) -> str:
        for dotted, flat in self._renames.items():
            sql = re.sub(rf"\b{re.escape(dotted)}\b", flat, sql)
        return sql

    def sql_to_dataset(
        self, sql: str, *, dims: list[str], param_values=None
    ) -> xr.Dataset:
        """Run ``sql`` and round-trip the result to an ``xr.Dataset``."""
        pdf = self._execute(self._rewrite(sql, param_values), param_values)
        return _pandas_to_dataset(pdf, dims)


class _DataFusionNative(EngineContext):
    """``xql.XarrayContext`` over the native DataFusion table provider."""

    flavor = "datafusion (XarrayContext, native)"

    def _connect(self):
        try:
            import xarray_sql._native  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "GEOBENCH_ENGINE=datafusion requires the compiled "
                "xarray_sql._native module (`maturin develop`); use "
                "GEOBENCH_ENGINE=datafusion-arrow for the pure-Python "
                "pyarrow-dataset path."
            ) from exc
        import xarray_sql as xql

        self._ctx = xql.XarrayContext()

    def from_dataset(self, name, ds, *, chunks=None, table_names=None):
        self._ctx.from_dataset(name, ds, chunks=chunks, table_names=table_names)

    def sql_to_dataset(self, sql, *, dims, param_values=None):
        df = (
            self._ctx.sql(sql, param_values=param_values)
            if param_values
            else self._ctx.sql(sql)
        )
        return df.to_dataset(dims=dims)


class _DataFusionArrow(EngineContext):
    """Plain ``datafusion.SessionContext`` over ``xql.arrow_dataset``."""

    flavor = "datafusion-arrow (pyarrow dataset, pure Python)"

    def _connect(self):
        from datafusion import SessionContext

        self._ctx = SessionContext()

    def _register(self, flat, dataset):
        self._ctx.register_dataset(flat, dataset)

    def _execute(self, sql, param_values):
        df = (
            self._ctx.sql(sql, param_values=param_values)
            if param_values
            else self._ctx.sql(sql)
        )
        return df.to_pandas()


class _DuckDB(EngineContext):
    """DuckDB over the same pyarrow pushdown datasets."""

    flavor = "duckdb"

    def _connect(self):
        import duckdb

        self._con = duckdb.connect()

    def _register(self, flat, dataset):
        self._con.register(flat, dataset)

    def _execute(self, sql, param_values):
        return self._con.execute(sql, param_values or {}).df()


class _Polars(EngineContext):
    """``polars.SQLContext`` over ``scan_pyarrow_dataset`` frames.

    Keeps the pyarrow datasets and builds the SQLContext per query.
    Polars' SQL layer renders TIMESTAMP literals as strptime-plus-cast
    expressions it cannot convert to pyarrow filters, so a WHERE over
    the full archive would scan everything; the same bounds applied as
    native expressions *do* push down. ``_execute`` therefore
    pre-filters each frame with the query's window parameters
    (identical predicate to the SQL WHERE, which still runs on top).
    """

    flavor = "polars (SQLContext + expression window pushdown)"

    # The window bounds a query passes as parameters, as (column, low
    # param, high param); applied per registered frame when the column
    # exists — the same inclusive predicate the SQL WHERE states.
    _BOUND_PARAMS = (
        ("time", "start", "end"),
        ("latitude", "lat_s", "lat_n"),
        ("longitude", "lon_w", "lon_e"),
    )

    def _connect(self):
        self._tables: dict[str, Any] = {}

    def _register(self, flat, dataset):
        self._tables[flat] = dataset

    def _rewrite(self, sql, param_values):
        sql = super()._rewrite(sql, param_values)
        for key, value in (param_values or {}).items():
            sql = re.sub(rf"\${key}\b", _literal(value), sql)
        return sql

    def _execute(self, sql, param_values):
        import polars as pl

        ctx = pl.SQLContext()
        params = param_values or {}
        for flat, dataset in self._tables.items():
            lf = pl.scan_pyarrow_dataset(dataset)
            names = set(dataset.schema.names)
            for col, lo, hi in self._BOUND_PARAMS:
                if col in names and lo in params and hi in params:
                    lf = lf.filter(
                        (pl.col(col) >= params[lo])
                        & (pl.col(col) <= params[hi])
                    )
            ctx.register(flat, lf)
        return ctx.execute(sql, eager=True).to_pandas()


_IMPLS = {
    "datafusion": _DataFusionNative,
    "datafusion-arrow": _DataFusionArrow,
    "duckdb": _DuckDB,
    "polars": _Polars,
}
_ENGINES = tuple(_IMPLS)
