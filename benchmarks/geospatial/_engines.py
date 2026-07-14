"""Engine-portable SQL layer for the geospatial suite.

``GEOBENCH_ENGINE`` selects which SQL engine executes each case's query,
so the same case scripts (same SQL, same datasets, same correctness
assertions) can be measured across engines:

* ``datafusion`` (default) — ``xql.XarrayContext``, the suite's original
  path, when the native module is importable; otherwise a plain
  ``datafusion.SessionContext`` over ``xql.arrow_dataset`` (pure Python).
  Which one ran is recorded in :attr:`EngineContext.flavor`.
* ``duckdb`` — DuckDB over the same pyarrow pushdown datasets.
* ``polars`` — ``polars.SQLContext`` over ``scan_pyarrow_dataset`` frames.

Every case builds one :class:`EngineContext`, registers datasets exactly
as it always registered them on ``XarrayContext``, and calls
:meth:`EngineContext.sql_to_dataset`. On the DataFusion-native path this
is byte-for-byte the original behavior (``from_dataset`` + ``sql`` +
``XarrayDataFrame.to_dataset``); the other engines register one pyarrow
dataset per dimension group under flattened table names
(``era5.surface`` → ``era5_surface`` — rewritten in the SQL text) and the
result rows are round-tripped to an ``xr.Dataset`` through pandas.

The DataFusion-only UDF cases (07 and the UDF half of 09) do not use
this layer; the suite runner records them as n/a for other engines.
"""

from __future__ import annotations

import datetime
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

_ENGINES = ("datafusion", "duckdb", "polars")


def engine_name() -> str:
    """The engine selected for this process (``GEOBENCH_ENGINE``)."""
    engine = os.environ.get("GEOBENCH_ENGINE", "datafusion")
    if engine not in _ENGINES:
        raise ValueError(f"GEOBENCH_ENGINE={engine!r}; expected {_ENGINES}")
    return engine


def _native_available() -> bool:
    if os.environ.get("GEOBENCH_NO_NATIVE"):  # test hook: force fallback
        return False
    try:
        import xarray_sql._native  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 — pure-Python source tree
        return False


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
    if len(dims) == 1:
        return xr.Dataset.from_dataframe(pdf.set_index(dims[0]).sort_index())
    return xr.Dataset.from_dataframe(pdf.set_index(dims).sort_index())


class EngineContext:
    """Uniform register-and-query facade over the suite's SQL engines."""

    def __init__(self, engine: str | None = None):
        self.engine = engine or engine_name()
        self.flavor = self.engine
        self._native = False
        self._renames: dict[str, str] = {}
        if self.engine == "datafusion":
            if _native_available():
                import xarray_sql as xql

                self.flavor = "datafusion (XarrayContext, native)"
                self._native = True
                self._ctx = xql.XarrayContext()
            else:
                from datafusion import SessionContext

                self.flavor = "datafusion (pyarrow dataset, no native)"
                self._ctx = SessionContext()
        elif self.engine == "duckdb":
            import duckdb

            self._con = duckdb.connect()
        else:
            # Polars: keep the pyarrow datasets and build the SQLContext
            # per query. Polars' SQL layer renders TIMESTAMP literals as
            # strptime-plus-cast expressions it cannot convert to pyarrow
            # filters, so a WHERE over the full archive would scan
            # everything; the same bounds applied as native expressions
            # *do* push down. sql_to_dataset therefore pre-filters each
            # frame with the query's window parameters (identical
            # predicate to the SQL WHERE, which still runs on top).
            self.flavor = "polars (SQLContext + expression window pushdown)"
            self._polars_tables: dict[str, Any] = {}

    # -- registration -----------------------------------------------------

    def from_dataset(self, name, ds, *, chunks=None, table_names=None):
        """Register ``ds`` as SQL table(s), mirroring XarrayContext naming."""
        if self._native:
            self._ctx.from_dataset(
                name, ds, chunks=chunks, table_names=table_names
            )
            return
        import xarray_sql as xql

        for flat, dotted, sub in _group_tables(name, ds, table_names):
            if dotted != flat:
                self._renames[dotted] = flat
            sub_chunks = (
                {d: c for d, c in chunks.items() if d in sub.dims}
                if isinstance(chunks, dict)
                else chunks
            ) or None
            dataset = xql.arrow_dataset(sub, sub_chunks)
            if self.engine == "duckdb":
                self._con.register(flat, dataset)
            elif self.engine == "polars":
                self._polars_tables[flat] = dataset
            else:
                self._ctx.register_dataset(flat, dataset)

    # -- querying ----------------------------------------------------------

    def _rewrite(self, sql: str, param_values) -> str:
        for dotted, flat in self._renames.items():
            sql = re.sub(rf"\b{re.escape(dotted)}\b", flat, sql)
        if self.engine == "polars" and param_values:
            for key, value in param_values.items():
                sql = re.sub(rf"\${key}\b", _literal(value), sql)
        return sql

    def sql_to_dataset(
        self, sql: str, *, dims: list[str], param_values=None
    ) -> xr.Dataset:
        """Run ``sql`` and round-trip the result to an ``xr.Dataset``."""
        sql = self._rewrite(sql, param_values)
        if self.engine == "datafusion":
            df = (
                self._ctx.sql(sql, param_values=param_values)
                if param_values
                else self._ctx.sql(sql)
            )
            if self._native:
                return df.to_dataset(dims=dims)
            return _pandas_to_dataset(df.to_pandas(), dims)
        if self.engine == "duckdb":
            pdf = self._con.execute(sql, param_values or {}).df()
            return _pandas_to_dataset(pdf, dims)
        out = self._polars_execute(sql, param_values)
        return _pandas_to_dataset(out.to_pandas(), dims)

    # The window bounds a query passes as parameters, as (column, low
    # param, high param); applied per registered frame when the column
    # exists — the same inclusive predicate the SQL WHERE states.
    _BOUND_PARAMS = (
        ("time", "start", "end"),
        ("latitude", "lat_s", "lat_n"),
        ("longitude", "lon_w", "lon_e"),
    )

    def _polars_execute(self, sql: str, param_values):
        import polars as pl

        ctx = pl.SQLContext()
        params = param_values or {}
        for flat, dataset in self._polars_tables.items():
            lf = pl.scan_pyarrow_dataset(dataset)
            names = set(dataset.schema.names)
            for col, lo, hi in self._BOUND_PARAMS:
                if col in names and lo in params and hi in params:
                    lf = lf.filter(
                        (pl.col(col) >= params[lo])
                        & (pl.col(col) <= params[hi])
                    )
            ctx.register(flat, lf)
        return ctx.execute(sql, eager=True)
