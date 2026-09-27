"""Tests for the ADBC engine adapter, run against every available database.

``xql.register`` ingests a Dataset into any database with an ADBC driver,
and ``xql.to_dataset`` rebuilds a labeled Dataset from the driver's Arrow
cursor. The contract tests below run once per backend in ``BACKENDS``;
tests of one database's specifics follow them.

SQLite and DuckDB always run. The others run when available:

- ``chdb`` and ``datafusion`` run in-process once their drivers are
  installed (``dbc install chdb datafusion``).
- ``clickhouse``, ``postgresql``, ``mysql``, ``mariadb``, ``trino``, and
  ``mssql`` need a server: set ``XARRAY_SQL_TEST_<NAME>_URI`` to its URI (and
  ``XARRAY_SQL_TEST_CLICKHOUSE_DRIVER`` for a ClickHouse driver that
  ``dbc`` did not install).
"""

import dataclasses
import importlib.util
import os
import uuid

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xarray_sql as xql

dbapi = pytest.importorskip("adbc_driver_manager.dbapi")


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
    schemas: bool = True
    """Whether mixed-dimension Datasets register as ``name.group``."""
    temporary: bool = True
    """Whether ``temporary=True`` is supported."""
    temporary_prefix: str = ""
    """How a temporary table's name is prefixed in queries."""
    drop_schema: str = "DROP SCHEMA IF EXISTS {} CASCADE"
    quote: str = '"'
    needs_uri: bool = False

    def connect(self):
        if self.driver is None or (self.needs_uri and not self.uri):
            pytest.skip(f"{self.name} is not available; see module docstring")
        kwargs = {"db_kwargs": {"uri": self.uri}} if self.uri else {}
        if self.entrypoint:
            kwargs["entrypoint"] = self.entrypoint
        try:
            return dbapi.connect(driver=self.driver, **kwargs)
        except dbapi.Error as exc:
            if self.needs_uri:
                raise
            pytest.skip(f"{self.name} driver is not installed ({exc})")


BACKENDS = [
    Backend("sqlite", _module_driver("adbc_driver_sqlite"), schemas=False),
    Backend("duckdb", _duckdb_driver(), entrypoint="duckdb_adbc_init"),
    Backend(
        "chdb",
        "chdb",
        uri="chdb://",
        drop_schema="DROP DATABASE IF EXISTS {}",
    ),
    Backend("datafusion", "datafusion", temporary=False),
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
        needs_uri=True,
    ),
    Backend(
        "mysql",
        "mysql",
        uri=_env("mysql"),
        drop_schema="DROP DATABASE IF EXISTS {}",
        quote="`",
        needs_uri=True,
    ),
    Backend(
        "mariadb",
        "mysql",
        uri=_env("mariadb"),
        drop_schema="DROP DATABASE IF EXISTS {}",
        quote="`",
        needs_uri=True,
    ),
    Backend(
        "trino",
        "trino",
        uri=_env("trino"),
        temporary=False,
        needs_uri=True,
    ),
    Backend(
        "mssql",
        "mssql",
        uri=_env("mssql"),
        drop_schema="DROP SCHEMA IF EXISTS {}",
        temporary_prefix="#",
        needs_uri=True,
    ),
]

NAMES = {
    ("time", "lat", "lon"): "surface",
    ("time", "level", "lat", "lon"): "atmosphere",
}


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


@pytest.fixture(params=BACKENDS, ids=[b.name for b in BACKENDS])
def db(request):
    backend = request.param
    database = Database(backend, backend.connect())
    yield database
    database.cleanup()
    database.con.close()


@pytest.fixture
def ds() -> xr.Dataset:
    rng = np.random.default_rng(3)
    weather = xr.Dataset(
        data_vars=dict(
            temperature=(
                ["time", "lat", "lon"],
                rng.standard_normal((8, 5, 6)),
            ),
            count=(["time", "lat", "lon"], rng.integers(0, 100, (8, 5, 6))),
            sst=(
                ["time", "lat", "lon"],
                rng.random((8, 5, 6)).astype("float32"),
            ),
            land=(["time", "lat", "lon"], rng.random((8, 5, 6)) > 0.5),
        ),
        coords=dict(
            time=pd.date_range("2021-01-01", periods=8, freq="h"),
            lat=np.linspace(-10.0, 10.0, 5),
            lon=np.linspace(0.0, 40.0, 6),
        ),
        attrs=dict(description="Synthetic weather."),
    ).chunk({"time": 4})
    weather["temperature"][0, 0, 0] = np.nan
    return weather


@pytest.fixture
def mixed_ds() -> xr.Dataset:
    rng = np.random.default_rng(11)
    return xr.Dataset(
        {
            "t2m": (["time", "lat", "lon"], rng.random((6, 3, 4))),
            "temperature": (
                ["time", "level", "lat", "lon"],
                rng.random((6, 2, 3, 4)),
            ),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=6, freq="D"),
            "lat": np.linspace(-90, 90, 3),
            "lon": np.linspace(-180, 180, 4),
            "level": [500, 1000],
        },
    ).chunk({"time": 2})


@pytest.fixture
def forecast() -> xr.Dataset:
    rng = np.random.default_rng(7)
    return xr.Dataset(
        {"t2m": (["step", "lat"], rng.random((4, 2)))},
        coords={
            "step": pd.to_timedelta([0, 6, 12, 18], unit="h"),
            "lat": [1.0, 2.0],
        },
    ).chunk({"step": 2})


def _select_all(db, table: str):
    return db.query(
        f"SELECT time, lat, lon, temperature, count, sst, land FROM {table} "
        "ORDER BY time, lat, lon"
    )


# The contract, on every backend -------------------------------------------


def test_round_trip_keeps_values_and_dtypes(db, ds):
    table = db.name("weather")
    xql.register(db.con, table, ds)

    out = xql.to_dataset(_select_all(db, table), template=ds)

    # NaN survives as missing, and bool/float32 come back as themselves
    # even where the database widens them (SQLite, MySQL).
    xr.testing.assert_identical(out, ds.compute())


def test_aggregates_skip_missing_values(db, ds):
    table = db.name("weather")
    xql.register(db.con, table, ds)

    cur = db.query(
        f"SELECT lat, lon, AVG(temperature) AS temperature FROM {table} "
        "GROUP BY lat, lon ORDER BY lat, lon"
    )
    out = xql.to_dataset(cur, template=ds)

    assert out.temperature.dims == ("lat", "lon")
    xr.testing.assert_allclose(
        out.temperature, ds.temperature.mean("time").compute()
    )


def test_existing_table_is_not_overwritten_by_default(db, ds):
    table = db.name("weather")
    xql.register(db.con, table, ds)

    with pytest.raises(dbapi.Error):
        xql.register(db.con, table, ds)


def test_replace_then_append(db, ds):
    table = db.name("weather")
    xql.register(db.con, table, ds)
    xql.register(db.con, table, ds.isel(time=slice(0, 4)), mode="replace")
    xql.register(db.con, table, ds.isel(time=slice(4, 8)), mode="append")

    out = xql.to_dataset(_select_all(db, table), template=ds)

    xr.testing.assert_identical(out, ds.compute())


def test_create_append_creates_then_appends(db, ds):
    table = db.name("weather")
    xql.register(db.con, table, ds, mode="create_append")
    xql.register(db.con, table, ds, mode="create_append")

    count = db.query(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    assert count == 2 * 8 * 5 * 6


def test_temporary_tables(db, ds):
    table = db.name("weather")
    if not db.backend.temporary:
        with pytest.raises(ValueError, match="temporary"):
            xql.register(db.con, table, ds, temporary=True)
        return

    xql.register(db.con, table, ds, temporary=True)

    temporary = f"{db.backend.temporary_prefix}{table}"
    count = db.query(f"SELECT COUNT(*) FROM {temporary}").fetchone()[0]
    assert count == 8 * 5 * 6


def test_mixed_dimensions_are_named_like_every_engine(db, mixed_ds):
    name = db.name("era5")
    if db.backend.schemas:
        xql.register(db.con, name, mixed_ds, table_names=NAMES)
        table = f"{name}.atmosphere"
    else:
        with pytest.warns(RuntimeWarning, match="flat"):
            xql.register(db.con, name, mixed_ds, table_names=NAMES)
        table = f"{name}_atmosphere"

    cur = db.query(
        f"SELECT time, level, lat, lon, temperature FROM {table} "
        "ORDER BY time, level, lat, lon"
    )
    out = xql.to_dataset(cur, template=mixed_ds[["temperature"]])

    xr.testing.assert_allclose(out.temperature, mixed_ds.temperature.compute())


@pytest.mark.parametrize(
    "chunks", [None, {"step": 2}], ids=["eager", "chunked"]
)
def test_timedelta_coordinates_round_trip(db, forecast, chunks):
    # Stored as a duration, an integer count (SQLite, ClickHouse), an
    # interval (DuckDB, PostgreSQL), or text (MySQL, Trino).
    table = db.name("forecast")
    xql.register(db.con, table, forecast)

    cur = db.query(f"SELECT step, lat, t2m FROM {table} ORDER BY step, lat")
    out = xql.to_dataset(
        cur, template=forecast, chunks=chunks, spill=chunks is not None
    )

    xr.testing.assert_allclose(out.compute(), forecast.compute())
    assert out.step.dtype == forecast.step.dtype


def test_chunked_round_trip_spills_the_cursor(db, ds):
    table = db.name("weather")
    xql.register(db.con, table, ds)

    cur = db.query(
        f"SELECT time, lat, lon, temperature FROM {table} "
        "ORDER BY time, lat, lon"
    )
    out = xql.to_dataset(cur, template=ds, chunks={"time": 2}, spill=True)

    assert out.temperature.chunks is not None
    xr.testing.assert_allclose(
        out.temperature.compute(), ds.temperature.compute()
    )


# One database's specifics ---------------------------------------------------


def _only(db, *names: str) -> None:
    if db.backend.name not in names:
        pytest.skip(f"specific to {', '.join(names)}")


def test_ingest_options_reach_the_driver(db, ds):
    _only(db, "sqlite")

    with pytest.raises(dbapi.Error, match="not.an.option"):
        xql.register(
            db.con,
            db.name("weather"),
            ds,
            ingest_options={"not.an.option": "x"},
        )


def test_postgresql_schema_failure_explains_the_aborted_transaction(
    db, mixed_ds
):
    # PostgreSQL rejects schema names starting with `pg_`, so CREATE SCHEMA
    # fails here for any user, and a failed statement aborts the
    # transaction: no fallback ingest could run after it.
    _only(db, "postgresql")

    with pytest.raises(RuntimeError, match="rollback"):
        xql.register(db.con, db.name("pg_era5"), mixed_ds, table_names=NAMES)


def test_postgresql_uses_an_existing_schema(db, mixed_ds):
    _only(db, "postgresql")
    name = db.name("era5")
    db.query(f'CREATE SCHEMA "{name}"').close()

    xql.register(db.con, name, mixed_ds, table_names=NAMES)

    count = db.query(f"SELECT COUNT(*) FROM {name}.surface").fetchone()[0]
    assert count == 6 * 3 * 4


def test_clickhouse_time_literals_mean_utc(db, ds):
    _only(db, "clickhouse", "chdb")
    table = db.name("weather")
    xql.register(db.con, table, ds)

    cur = db.query(
        f"SELECT time, lat, lon, temperature FROM {table} "
        "WHERE time >= '2021-01-01 04:00:00' ORDER BY time, lat, lon"
    )
    out = xql.to_dataset(cur, template=ds)

    expected = ds.temperature.isel(time=slice(4, None))
    xr.testing.assert_allclose(out.temperature, expected.compute())


def test_mysql_keeps_the_default_database(db, mixed_ds):
    # The MySQL driver ignores the target schema, so the adapter switches
    # the default database for the ingest and must switch it back.
    _only(db, "mysql", "mariadb")
    before = db.query("SELECT DATABASE()").fetchone()[0]

    xql.register(db.con, db.name("era5"), mixed_ds, table_names=NAMES)

    assert db.query("SELECT DATABASE()").fetchone()[0] == before
