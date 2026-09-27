"""Tests for the ADBC engine adapter, run against every available database.

``xql.register`` ingests a Dataset into any database with an ADBC driver,
and ``xql.to_dataset`` rebuilds a labeled Dataset from the driver's Arrow
cursor. The contract tests below run once per backend (the ``db``
fixture; which backends run, and how to enable more, is in
``tests/_adbc.py``); tests of one database's specifics follow them.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xarray_sql as xql

dbapi = pytest.importorskip("adbc_driver_manager.dbapi")


NAMES = {
    ("time", "lat", "lon"): "surface",
    ("time", "level", "lat", "lon"): "atmosphere",
}


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
    weather["sst"][0, 0, 1] = np.nan
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


def test_time_filters_select_the_right_rows(db, ds):
    # A literal means UTC everywhere, and SQLite's text times compare
    # with it correctly, including at an inclusive bound.
    table = db.name("weather")
    xql.register(db.con, table, ds)
    start, end = (
        db.backend.time_literal.format(f"2021-01-01 0{hour}:00:00")
        for hour in (4, 6)
    )

    cur = db.query(
        f"SELECT time, lat, lon, temperature FROM {table} "
        f"WHERE time BETWEEN {start} AND {end} ORDER BY time, lat, lon"
    )
    out = xql.to_dataset(cur, template=ds)

    expected = ds.temperature.isel(time=slice(4, 7))
    xr.testing.assert_allclose(out.temperature, expected.compute())


def test_awkward_variable_names_round_trip(db):
    ds = xr.Dataset(
        {
            "select": ("x", np.arange(3.0)),
            "wind speed": ("x", np.arange(3.0) * 2),
            "Order": ("x", np.arange(3.0) * 3),
        },
        coords={"x": [10, 20, 30]},
    ).chunk({"x": 3})
    table = db.name("awkward")
    xql.register(db.con, table, ds)

    columns = ", ".join(
        db.quoted(n) for n in ["x", "select", "wind speed", "Order"]
    )
    cur = db.query(f"SELECT {columns} FROM {table} ORDER BY {db.quoted('x')}")
    out = xql.to_dataset(cur, template=ds)

    xr.testing.assert_identical(out, ds.compute())


def test_text_coordinates_round_trip(db):
    ds = xr.Dataset(
        {"count": ("station", np.arange(4))},
        coords={"station": ["O'Hare", 'say "hi"', "東京", "Zürich"]},
    ).chunk({"station": 4})
    table = db.name("stations")
    xql.register(db.con, table, ds)

    cur = db.query(f"SELECT station, count FROM {table}")
    out = xql.to_dataset(cur, template=ds)

    xr.testing.assert_identical(
        out.sortby("station"), ds.compute().sortby("station")
    )


def test_integer_extremes_round_trip(db):
    info64 = np.iinfo(np.int64)
    ds = xr.Dataset(
        {
            "i64": ("x", np.array([info64.min, 0, info64.max])),
            "u8": ("x", np.array([0, 1, 255], dtype=np.uint8)),
            "u32": ("x", np.array([0, 1, 2**32 - 1], dtype=np.uint32)),
            "u64": ("x", np.array([0, 1, 2**62], dtype=np.uint64)),
        },
        coords={"x": [0, 1, 2]},
    ).chunk({"x": 3})
    table = db.name("extremes")
    xql.register(db.con, table, ds)

    cur = db.query(f"SELECT x, i64, u8, u32, u64 FROM {table} ORDER BY x")
    out = xql.to_dataset(cur, template=ds)

    xr.testing.assert_identical(out, ds.compute())


def test_uint64_beyond_int64_is_never_silently_wrong(db):
    ds = xr.Dataset(
        {"u64": ("x", np.array([2**63, 2**64 - 1], dtype=np.uint64))},
        coords={"x": [0, 1]},
    ).chunk({"x": 2})
    table = db.name("huge")
    try:
        xql.register(db.con, table, ds)
    except (ValueError, dbapi.Error):
        return  # refused loudly: acceptable

    cur = db.query(f"SELECT x, u64 FROM {table} ORDER BY x")
    try:
        out = xql.to_dataset(cur, template=ds)
    except (ValueError, TypeError, dbapi.Error):
        return  # refused loudly on the way back: acceptable
    xr.testing.assert_identical(out, ds.compute())


def test_nanosecond_times_are_kept_or_truncation_is_reported(db):
    times = pd.to_datetime(["2021-01-01", "2021-01-01"]) + pd.to_timedelta(
        [1, 2], unit="ns"
    )
    ds = xr.Dataset({"v": ("time", [1.0, 2.0])}, coords={"time": times}).chunk(
        {"time": 2}
    )
    table = db.name("nanos")
    if db.backend.microseconds:
        with pytest.warns(RuntimeWarning, match="microsecond"):
            xql.register(db.con, table, ds)
        return

    xql.register(db.con, table, ds)

    cur = db.query(f"SELECT time, v FROM {table} ORDER BY time")
    xr.testing.assert_identical(xql.to_dataset(cur, template=ds), ds.compute())


def test_many_chunks_ingest(db):
    ds = xr.Dataset(
        {"v": (["time", "x"], np.arange(200_000.0).reshape(1000, 200))},
        coords={"time": np.arange(1000), "x": np.arange(200)},
    ).chunk({"time": 100})
    table = db.name("big")
    xql.register(db.con, table, ds)

    count, total = db.query(f"SELECT COUNT(*), SUM(v) FROM {table}").fetchone()
    assert (count, float(total)) == (200_000, float(ds.v.sum()))


def test_empty_result_round_trips(db, ds):
    table = db.name("weather")
    xql.register(db.con, table, ds)

    cur = db.query(
        f"SELECT time, lat, lon, temperature FROM {table} WHERE lat > 1000"
    )
    out = xql.to_dataset(cur, template=ds)

    assert out.temperature.size == 0


def test_long_table_name(db, ds):
    table = db.name("t" * 51)  # 60 characters with the unique suffix
    assert len(table) == 60
    xql.register(db.con, table, ds)

    count = db.query(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    assert count == 8 * 5 * 6


def test_mixed_case_names_are_found_quoted(db, ds):
    table = db.name("Weather")
    if db.backend.folds:
        with pytest.warns(RuntimeWarning, match="quote"):
            xql.register(db.con, table, ds)
    else:
        xql.register(db.con, table, ds)

    count = db.query(f"SELECT COUNT(*) FROM {db.quoted(table)}").fetchone()[0]
    assert count == 8 * 5 * 6


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


def test_mysql_keeps_the_default_database(db, mixed_ds):
    # The MySQL driver ignores the target schema, so the adapter switches
    # the default database for the ingest and must switch it back.
    _only(db, "mysql", "mariadb")
    before = db.query("SELECT DATABASE()").fetchone()[0]

    xql.register(db.con, db.name("era5"), mixed_ds, table_names=NAMES)

    assert db.query("SELECT DATABASE()").fetchone()[0] == before
