"""Tests for serving Datasets over Arrow Flight SQL.

A ``FlightSQLServer`` hosts lazily registered Datasets; the client here
is ADBC's Flight SQL driver, the same one remote users would connect
with, so the tests exercise the real wire protocol end to end.

How served data round-trips (types, missing values, names, mixed
dimensions, the ARCO-ERA5 queries) is tested by the ADBC contract, where
the server is the ``served`` backend (``tests/_adbc.py``). The tests here
cover what only a server has: serving, discovery, read-only SQL, plain
Arrow Flight, and other databases as its clients.
"""

import os
import threading
import urllib.request

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.flight as flight
import pytest
import xarray as xr

import xarray_sql as xql

flight_sql = pytest.importorskip("adbc_driver_flightsql.dbapi")


@pytest.fixture
def ds() -> xr.Dataset:
    np.random.seed(5)
    return xr.Dataset(
        data_vars=dict(
            temperature=(["time", "lat", "lon"], np.random.randn(8, 5, 6)),
            precipitation=(["time", "lat", "lon"], np.random.rand(8, 5, 6)),
        ),
        coords=dict(
            time=pd.date_range("2021-01-01", periods=8, freq="h"),
            lat=np.linspace(-10.0, 10.0, 5),
            lon=np.linspace(0.0, 40.0, 6),
        ),
        attrs=dict(description="Synthetic weather."),
    ).chunk({"time": 4})


@pytest.fixture
def server(ds):
    with xql.serve({"weather": ds}) as running:
        yield running


@pytest.fixture
def con(server):
    connection = flight_sql.connect(server.uri)
    yield connection
    connection.close()


def _query(con, sql: str):
    cur = con.cursor()
    cur.execute(sql)
    return cur


def test_datasets_registered_while_serving_are_visible(server, con, ds):
    server.register("later", ds[["precipitation"]])

    count = _query(con, "SELECT COUNT(*) FROM later").fetchone()[0]
    assert count == 8 * 5 * 6


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE EXTERNAL TABLE leak STORED AS CSV LOCATION '/etc/hosts'",
        "COPY (SELECT 1) TO '/tmp/xarray-sql-flight-leak.csv'",
        "CREATE TABLE copy AS SELECT * FROM weather",
        "SET datafusion.execution.batch_size = 1",
        # Read-only, but a file outside the served Datasets.
        "SELECT * FROM '/etc/hosts'",
    ],
)
def test_clients_cannot_reach_beyond_the_datasets(con, statement):
    with pytest.raises(flight_sql.Error):
        _query(con, statement).fetchall()


def test_a_malformed_descriptor_is_the_clients_error(server):
    client = flight.FlightClient(server.uri)
    with pytest.raises(pa.ArrowInvalid):
        client.get_schema(flight.FlightDescriptor.for_command(b"not a command"))


def test_memory_limit_fails_the_query_not_the_server():
    ds = xr.Dataset(
        {"v": (["time", "x"], np.random.rand(1_000, 200))},
        coords={"time": np.arange(1_000), "x": np.arange(200)},
    ).chunk({"time": 100})
    with xql.serve({"big": ds}, memory_limit=2**20) as server:
        con = flight_sql.connect(server.uri)
        with pytest.raises(flight_sql.Error, match="Resources exhausted"):
            _query(
                con,
                "SELECT COUNT(*) FROM big a JOIN big b "
                "ON a.time = b.time AND a.x = b.x",
            ).fetchall()
        count = _query(con, "SELECT COUNT(*) FROM big").fetchone()[0]
        con.close()

    assert count == 1_000 * 200


def test_a_failed_mixed_dimension_registration_adds_nothing():
    ds = xr.Dataset(
        {
            "t2m": (["time", "lat"], np.random.rand(4, 3)),
            "temperature": (["time", "level", "lat"], np.random.rand(4, 2, 3)),
        },
        coords={"time": np.arange(4), "lat": [0.0, 1.0, 2.0], "level": [1, 2]},
    ).chunk({"time": 2})
    surface, atmosphere = ("time", "lat"), ("time", "level", "lat")
    server = xql.FlightSQLServer()
    server.register(
        "era5", ds, table_names={surface: "old", atmosphere: "atmosphere"}
    )

    # `surface` is new but `atmosphere` is taken: neither is added.
    with pytest.raises(RuntimeError, match="atmosphere"):
        server.register(
            "era5",
            ds,
            table_names={surface: "surface", atmosphere: "atmosphere"},
        )
    with server.serve():
        con = flight_sql.connect(server.uri)
        with pytest.raises(flight_sql.Error, match="surface"):
            _query(con, "SELECT * FROM era5.surface").fetchall()
        con.close()


def test_tables_are_discoverable(con):
    objects = con.adbc_get_objects(depth="tables").read_all().to_pylist()
    tables = {
        table["table_name"]
        for catalog in objects
        for schema in catalog["catalog_db_schemas"] or []
        for table in schema["db_schema_tables"] or []
    }

    assert "weather" in tables


def test_shutdown_stops_the_server(ds):
    with xql.serve({"weather": ds}) as server:
        assert server.is_running
        assert server.uri == f"grpc://127.0.0.1:{server.port}"

    assert not server.is_running


def test_shutdown_does_not_wait_forever_on_an_unread_result():
    big = xr.Dataset(
        {"v": (["time", "x"], np.zeros((2_000, 1_000)))},
        coords={"time": np.arange(2_000), "x": np.arange(1_000)},
    ).chunk({"time": 100})
    server = xql.serve({"big": big})
    con = flight_sql.connect(server.uri)
    cur = _query(con, "SELECT * FROM big")
    cur.fetchone()  # leave the rest of the stream unread

    done = threading.Event()
    stopper = threading.Thread(
        target=lambda: (server.shutdown(timeout=0.5), done.set()),
        daemon=True,
    )
    stopper.start()
    stopper.join(timeout=30)

    assert done.is_set()
    assert not server.is_running


def _read_path(server, *path):
    client = flight.FlightClient(server.uri)
    info = client.get_flight_info(flight.FlightDescriptor.for_path(*path))
    return client.do_get(info.endpoints[0].ticket).read_all()


def test_plain_flight_path_names_a_table(server, ds):
    table = _read_path(server, "weather")
    out = xql.to_dataset(table.sort_by([("time", "ascending")]), template=ds)

    xr.testing.assert_allclose(out, ds.compute())


def test_plain_flight_path_finds_a_mixed_case_name(ds):
    # ClickHouse's arrowFlight sends the name as given, unquoted.
    server = xql.FlightSQLServer()
    with pytest.warns(RuntimeWarning, match="quote"):
        server.register("Weather", ds)
    with server.serve():
        table = _read_path(server, "Weather")

    assert table.num_rows == 8 * 5 * 6


def test_plain_flight_path_can_be_schema_qualified(ds):
    upper = ds.temperature.expand_dims(level=[500, 850]).rename("upper")
    server = xql.FlightSQLServer()
    server.register(
        "era5",
        ds.assign(upper=upper),
        table_names={("time", "lat", "lon"): "surface"},
    )
    with server.serve():
        table = _read_path(server, "era5.surface")

    assert table.num_rows == 8 * 5 * 6


@pytest.mark.parametrize("via", ["server", "xql"])
def test_mixed_case_warning_points_at_the_caller(ds, via):
    server = xql.FlightSQLServer()
    with pytest.warns(RuntimeWarning, match="quote") as record:
        if via == "server":
            server.register("Weather", ds)
        else:
            xql.register(server, "Weather", ds)

    assert record[0].filename == __file__


def test_plain_flight_path_can_be_a_query(server, ds):
    table = _read_path(
        server,
        "SELECT lat, lon, AVG(temperature) AS temperature FROM weather "
        "GROUP BY lat, lon ORDER BY lat, lon",
    )
    out = xql.to_dataset(table, template=ds)

    expected = ds.temperature.mean("time")
    xr.testing.assert_allclose(out.temperature, expected.compute())


def test_plain_flight_schema_of_a_path(server):
    client = flight.FlightClient(server.uri)
    result = client.get_schema(flight.FlightDescriptor.for_path("weather"))

    assert result.schema.names == [
        "time",
        "lat",
        "lon",
        "temperature",
        "precipitation",
    ]


def test_clickhouse_reads_through_arrow_flight(ds):
    uri = os.environ.get("XARRAY_SQL_TEST_CLICKHOUSE_URI", "")
    if not uri.startswith(("http://", "https://")):
        # chDB, which the other ClickHouse tests can use, is built without
        # the arrowFlight table function.
        pytest.skip(
            "set XARRAY_SQL_TEST_CLICKHOUSE_URI to a ClickHouse server's "
            "HTTP address to run against ClickHouse"
        )
    # A ClickHouse in a container reaches this process through the Docker
    # host's address, so the server listens on every interface then.
    host = os.environ.get("XARRAY_SQL_TEST_CLICKHOUSE_FLIGHT_HOST")
    bind = "0.0.0.0" if host else "127.0.0.1"
    with xql.serve({"weather": ds}, host=bind) as server:
        sql = (
            "SELECT round(avg(temperature), 9) FROM "
            f"arrowFlight('{host or '127.0.0.1'}:{server.port}', 'weather')"
        )
        request = urllib.request.Request(uri, data=sql.encode())
        with urllib.request.urlopen(request, timeout=60) as response:
            avg = float(response.read().decode())

    assert avg == pytest.approx(float(ds.temperature.mean()))


@pytest.fixture(scope="module")
def spark():
    jar = os.environ.get("XARRAY_SQL_TEST_FLIGHT_SQL_JDBC_JAR")
    if not jar:
        pytest.skip(
            "set XARRAY_SQL_TEST_FLIGHT_SQL_JDBC_JAR to the Arrow Flight SQL "
            "JDBC driver jar to run against Spark"
        )
    pyspark_sql = pytest.importorskip("pyspark.sql")
    # The JVM's zone applies to timestamps read over JDBC, so it must be
    # UTC for xarray's (UTC) times to arrive unshifted.
    session = (
        pyspark_sql.SparkSession.builder.master("local[1]")
        .config("spark.jars", jar)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def _spark_read(spark, server, dbtable):
    return (
        spark.read.format("jdbc")
        .option(
            "url",
            f"jdbc:arrow-flight-sql://127.0.0.1:{server.port}"
            "/?useEncryption=false",
        )
        .option("driver", "org.apache.arrow.driver.jdbc.ArrowFlightJdbcDriver")
        .option("dbtable", dbtable)
        .load()
    )


def test_spark_reads_over_jdbc(spark, server, ds):
    frame = _spark_read(spark, server, "weather").orderBy("time", "lat", "lon")
    out = xql.to_dataset(frame.toArrow(), template=ds)

    xr.testing.assert_allclose(out, ds.compute())


def test_spark_pushes_filters_to_the_server(spark, server, ds):
    frame = (
        _spark_read(spark, server, "weather")
        .where("time >= TIMESTAMP '2021-01-01 04:00:00' AND lat > 0")
        .select("time", "lat", "lon", "temperature")
        .orderBy("time", "lat", "lon")
    )
    plan = frame._jdf.queryExecution().executedPlan().toString()
    out = xql.to_dataset(frame.toArrow(), template=ds)

    assert "*GreaterThanOrEqual(time," in plan
    expected = ds.temperature.isel(time=slice(4, None)).where(
        ds.lat > 0, drop=True
    )
    xr.testing.assert_allclose(out.temperature, expected.compute())
