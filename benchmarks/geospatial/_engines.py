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
* ``adbc-<backend>`` — a database behind an ADBC driver, one of the
  backends the ADBC tests run against (``tests/_adbc.py``: ``sqlite``,
  ``duckdb``, ``postgresql``, ``mssql``, ...), connected the same way.
  Registration copies data into the database, so each query's window is
  ingested when the query runs; see :class:`_ADBC`.

Every case builds one :class:`EngineContext`, registers datasets exactly
as it always registered them on ``XarrayContext``, and calls
:meth:`EngineContext.sql_to_dataset`. On the ``datafusion`` path this
is byte-for-byte the original behavior (``from_dataset`` + ``sql`` +
``XarrayDataFrame.to_dataset``); the other engines register one pyarrow
dataset per dimension group under the flat table names
``xql.arrow_datasets`` returns (``era5.surface`` → ``era5_surface`` —
rewritten in the SQL text, since these paths register frames one at a
time rather than through a schema) and the result rows are
round-tripped to an ``xr.Dataset`` through pandas.

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
    if engine not in _ENGINES and not engine.startswith("adbc-"):
        raise ValueError(f"GEOBENCH_ENGINE={engine!r}; expected {_ENGINES}")
    return engine


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
        if isinstance(dtype, pd.DatetimeTZDtype):
            # A zone-labeled result (ClickHouse, timestamptz) is compared
            # with the reference's plain UTC times.
            pdf[col] = pdf[col].dt.tz_convert("UTC").dt.tz_localize(None)
        if pd.api.types.is_datetime64_any_dtype(pdf[col].dtype):
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
            name = engine or engine_name()
            cls = _ADBC if name.startswith("adbc-") else _IMPLS[name]
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

        tables = xql.arrow_datasets(
            ds, name, chunks=chunks, table_names=table_names
        )
        for flat, dataset in tables.items():
            if flat != name:
                # `era5_surface` here is `era5.surface` in the case's SQL,
                # which was written against XarrayContext's schema split.
                group = flat[len(name) + 1 :]
                self._renames[f"{name}.{group}"] = flat
            self._register(flat, dataset)

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


def _as_timedelta(value):
    """A duration a database returned as an interval or text, else as is.

    DuckDB and PostgreSQL return intervals (pandas ``DateOffset``), MySQL
    and Trino text such as ``'21600s'``; the other engines hand back
    ``timedelta`` values the reference compares with directly.
    """
    if isinstance(value, pd.DateOffset):
        parts = value.kwds
        return pd.Timedelta(
            days=parts.get("days", 0),
            microseconds=parts.get("microseconds", 0),
            nanoseconds=parts.get("nanoseconds", 0),
        )
    if isinstance(value, str):
        try:
            return pd.Timedelta(value)
        except ValueError:
            return value
    return value


class _ADBC(EngineContext):
    """A database behind an ADBC driver, via ``xql.register``.

    ADBC registration copies data into the database, and the cases open
    the whole ARCO-ERA5 archive, so nothing is ingested at registration.
    When a query runs, each Dataset is cut to the variables the SQL names
    and to the window its parameters bound (the same inclusive bounds as
    the SQL ``WHERE``, which still applies), and that window is ingested.

    The case SQL is written for DataFusion. The few places it differs
    from another database's SQL are rewritten explicitly in
    :meth:`_dialect`, since xarray-sql translates data, not queries.
    """

    _BOUND_PARAMS = _Polars._BOUND_PARAMS

    _HOUR = {
        "sqlite": "CAST(strftime('%H', {}) AS INTEGER)",
        "mssql": "DATEPART(hour, {})",
        "trino": "hour({})",
        "mysql": "HOUR({})",
        "mariadb": "HOUR({})",
        "clickhouse": "toHour({})",
        "chdb": "toHour({})",
    }
    """How each database extracts the hour, where not ``date_part``."""

    _DOUBLE = {"postgresql": "DOUBLE PRECISION", "mssql": "FLOAT"}

    def _connect(self):
        import sys
        from pathlib import Path

        from _harness import CaseSkipped

        # The ADBC tests' backend table: one source of truth for which
        # databases exist and how to reach them.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tests._adbc import BACKENDS, Database

        backend_name = self.engine.removeprefix("adbc-")
        backends = {b.name: b for b in BACKENDS}
        if backend_name not in backends:
            raise ValueError(
                f"GEOBENCH_ENGINE={self.engine!r}; ADBC backends are "
                f"{sorted('adbc-' + name for name in backends)}"
            )
        backend = backends[backend_name]
        try:
            con = backend.connect()
        except BaseException as exc:  # pytest.skip outside pytest
            if type(exc).__name__ == "Skipped":
                raise CaseSkipped(str(exc)) from None
            raise
        self._db = Database(backend, con)
        self._datasets: dict[str, tuple] = {}
        self.flavor = f"adbc ({backend_name})"

    def from_dataset(self, name, ds, *, chunks=None, table_names=None):
        self._datasets[name] = (ds, chunks, table_names)

    def _window(self, ds: xr.Dataset, params: dict) -> xr.Dataset:
        for dim, low, high in self._BOUND_PARAMS:
            if dim in ds.dims and low in params and high in params:
                index = ds.indexes[dim]
                keep = (index >= params[low]) & (index <= params[high])
                ds = ds.isel({dim: np.flatnonzero(keep)})
        return ds

    def _ingest(self, sql: str, params: dict) -> str:
        """Ingests what *sql* reads; returns *sql* naming those tables."""
        import xarray_sql as xql
        from xarray_sql.df import group_vars_by_dims, resolve_table_names

        mentioned = {a or b for a, b in re.findall(r'"([^"]+)"|(\w+)', sql)}
        for name, (ds, chunks, table_names) in self._datasets.items():
            groups = group_vars_by_dims(ds)
            names = resolve_table_names(ds, table_names)
            for dims, var_names in groups.items():
                wanted = [v for v in var_names if v in mentioned]
                table = f"{name}.{names[dims]}" if len(groups) > 1 else name
                if not wanted or not re.search(rf"\b{re.escape(table)}\b", sql):
                    continue
                window = self._window(ds[wanted], params)
                flat = table.replace(".", "_")
                xql.register(
                    self._db.con, flat, window, chunks=chunks, mode="replace"
                )
                sql = re.sub(rf"\b{re.escape(table)}\b", flat, sql)
        return sql

    def _literal(self, value) -> str:
        if isinstance(value, (datetime.datetime, pd.Timestamp, np.datetime64)):
            text = pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            if self._db.backend.name == "trino":
                return f"TIMESTAMP '{text}'"
            return f"'{text}'"
        return _literal(value)

    def _dialect(self, sql: str) -> str:
        backend = self._db.backend
        if backend.quote != '"':
            sql = re.sub(
                r'"([^"]*)"',
                lambda m: f"{backend.quote}{m[1]}{backend.quote}",
                sql,
            )
        hour = self._HOUR.get(backend.name)
        if hour:
            sql = re.sub(
                r"date_part\('hour',\s*([\w.]+)\)",
                lambda m: hour.format(m[1]),
                sql,
            )
        double = self._DOUBLE.get(backend.name)
        if double:
            sql = re.sub(r"\bAS DOUBLE\b", f"AS {double}", sql)
        return sql

    def sql_to_dataset(self, sql, *, dims, param_values=None):
        params = param_values or {}
        sql = self._ingest(sql, params)
        for key, value in params.items():
            sql = re.sub(rf"\${key}\b", self._literal(value), sql)
        cur = self._db.query(self._dialect(sql))
        pdf = cur.fetch_arrow_table().to_pandas()
        for dim in dims:
            if pdf[dim].dtype == object:
                pdf[dim] = pdf[dim].map(_as_timedelta)
        return _pandas_to_dataset(pdf, dims)


_IMPLS = {
    "datafusion": _DataFusionNative,
    "datafusion-arrow": _DataFusionArrow,
    "duckdb": _DuckDB,
    "polars": _Polars,
}
_ENGINES = tuple(_IMPLS)
