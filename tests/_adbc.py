"""The databases the ADBC tests run against, shared by their modules.

``db`` (defined in ``conftest.py``) runs a test once per backend in
``BACKENDS``. SQLite and DuckDB always run. The others run when
available:

- ``chdb`` and ``datafusion`` run in-process once their drivers are
  installed (``dbc install chdb datafusion``).
- ``clickhouse``, ``postgresql``, ``mysql``, ``mariadb``, ``trino``,
  ``mssql``, and ``flightsql`` need a server: set
  ``XARRAY_SQL_TEST_<NAME>_URI`` to its URI (and
  ``XARRAY_SQL_TEST_CLICKHOUSE_DRIVER`` for a ClickHouse driver that
  ``dbc`` did not install; ``XARRAY_SQL_TEST_FLIGHTSQL_USERNAME`` and
  ``_PASSWORD`` for a Flight SQL server such as GizmoSQL).
"""

import dataclasses
import importlib.util
import os
import uuid

import pytest

try:
    from adbc_driver_manager import dbapi
except ImportError:  # the fixture skips; see Backend.connect
    dbapi = None


def _module_driver(module: str) -> str | None:
    """The driver library an ``adbc_driver_*`` Python package ships."""
    try:
        return str(importlib.import_module(module)._driver_path())
    except ImportError:
        return None


def _duckdb_driver() -> str | None:
    """The shared library holding DuckDB's ADBC entrypoint."""
    for module in ("_duckdb", "duckdb.duckdb"):
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            continue
        if spec is not None and spec.origin:
            return spec.origin
    return None


def _env(name: str) -> str | None:
    return os.environ.get(f"XARRAY_SQL_TEST_{name.upper()}_URI")


@dataclasses.dataclass(frozen=True)
class Backend:
    """A database to run the contract tests against."""

    name: str
    driver: str | None
    uri: str | None = None
    entrypoint: str | None = None
    options: tuple[tuple[str, str], ...] = ()
    """Further database options, e.g. credentials."""
    schemas: bool = True
    """Whether mixed-dimension Datasets register as ``name.group``."""
    temporary: bool = True
    """Whether ``temporary=True`` is supported."""
    temporary_prefix: str = ""
    """How a temporary table's name is prefixed in queries."""
    drop_schema: str = "DROP SCHEMA IF EXISTS {} CASCADE"
    quote: str = '"'
    time_literal: str = "'{}'"
    """How a timestamp literal is written in a comparison."""
    microseconds: bool = False
    """Whether the database stores times only to the microsecond."""
    folds: bool = False
    """Whether unquoted names fold, so mixed-case ones need quotes."""
    needs_uri: bool = False

    def connect(self):
        if dbapi is None:
            pytest.skip("adbc-driver-manager is not installed")
        if self.driver is None or (self.needs_uri and not self.uri):
            pytest.skip(f"{self.name} is not available; see module docstring")
        db_kwargs = dict(self.options)
        if self.uri:
            db_kwargs["uri"] = self.uri
        kwargs: dict = {"db_kwargs": db_kwargs} if db_kwargs else {}
        if self.entrypoint:
            kwargs["entrypoint"] = self.entrypoint
        try:
            return dbapi.connect(driver=self.driver, **kwargs)
        except dbapi.Error as exc:
            if self.needs_uri:
                raise
            pytest.skip(f"{self.name} driver is not installed ({exc})")


def _credentials(name: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (option, os.environ[f"XARRAY_SQL_TEST_{name.upper()}_{option.upper()}"])
        for option in ("username", "password")
        if f"XARRAY_SQL_TEST_{name.upper()}_{option.upper()}" in os.environ
    )


BACKENDS = [
    Backend("sqlite", _module_driver("adbc_driver_sqlite"), schemas=False),
    Backend("duckdb", _duckdb_driver(), entrypoint="duckdb_adbc_init"),
    Backend(
        "chdb",
        "chdb",
        uri="chdb://",
        drop_schema="DROP DATABASE IF EXISTS {}",
    ),
    Backend("datafusion", "datafusion", temporary=False, folds=True),
    Backend(
        "clickhouse",
        os.environ.get("XARRAY_SQL_TEST_CLICKHOUSE_DRIVER", "clickhouse"),
        uri=_env("clickhouse"),
        drop_schema="DROP DATABASE IF EXISTS {}",
        needs_uri=True,
    ),
    Backend(
        "postgresql",
        _module_driver("adbc_driver_postgresql") or "postgresql",
        uri=_env("postgresql"),
        microseconds=True,
        folds=True,
        needs_uri=True,
    ),
    Backend(
        "mysql",
        "mysql",
        uri=_env("mysql"),
        drop_schema="DROP DATABASE IF EXISTS {}",
        quote="`",
        microseconds=True,
        needs_uri=True,
    ),
    Backend(
        "mariadb",
        "mysql",
        uri=_env("mariadb"),
        drop_schema="DROP DATABASE IF EXISTS {}",
        quote="`",
        microseconds=True,
        needs_uri=True,
    ),
    Backend(
        "trino",
        "trino",
        uri=_env("trino"),
        temporary=False,
        time_literal="TIMESTAMP '{}'",
        needs_uri=True,
    ),
    Backend(
        "mssql",
        "mssql",
        uri=_env("mssql"),
        drop_schema="DROP SCHEMA IF EXISTS {}",
        temporary_prefix="#",
        microseconds=True,
        needs_uri=True,
    ),
    Backend(
        "flightsql",
        _module_driver("adbc_driver_flightsql") or "flightsql",
        uri=_env("flightsql"),
        options=_credentials("flightsql"),
        needs_uri=True,
    ),
]


class Database:
    """A connection plus the unique names a test creates, dropped after."""

    def __init__(self, backend: Backend, con) -> None:
        self.backend = backend
        self.con = con
        self._created: list[str] = []

    def name(self, base: str) -> str:
        """A fresh table (or schema) name, dropped when the test ends."""
        name = f"{base}_{uuid.uuid4().hex[:8]}"
        self._created.append(name)
        return name

    def quoted(self, identifier: str) -> str:
        quote = self.backend.quote
        return f"{quote}{identifier.replace(quote, quote * 2)}{quote}"

    def query(self, sql: str):
        cur = self.con.cursor()
        cur.execute(sql)
        return cur

    def cleanup(self) -> None:
        postgresql = self.backend.name == "postgresql"
        if postgresql:
            self.con.rollback()
        for name in self._created:
            quoted = f"{self.backend.quote}{name}{self.backend.quote}"
            for statement in (
                f"DROP TABLE IF EXISTS {quoted}",
                self.backend.drop_schema.format(quoted),
            ):
                try:
                    self.query(statement).close()
                except dbapi.Error:
                    if postgresql:
                        self.con.rollback()
        if postgresql:
            self.con.commit()
