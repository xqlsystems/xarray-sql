"""Serve lazy xarray Datasets over Arrow Flight SQL.

The other adapters bring an engine to the data: the Dataset becomes a
table on a connection in this process. A
[FlightSQLServer][xarray_sql.backends.flight.FlightSQLServer] turns that
around and brings the data to remote clients. It hosts a DataFusion
session over the same lazy tables
[XarrayContext][xarray_sql.XarrayContext] uses and speaks
[Flight SQL](https://arrow.apache.org/docs/format/FlightSql.html), so any
Flight SQL client — ADBC's Flight SQL driver in Python, R, Go, or Java,
the Flight SQL JDBC and ODBC drivers, and the SQL tools built on them —
can query the Dataset without a copy::

    server = xql.serve({"era5": ds}, port=8815)

    # elsewhere
    import adbc_driver_flightsql.dbapi as flight_sql
    con = flight_sql.connect("grpc://server-host:8815")
    cur = con.cursor()
    cur.execute("SELECT AVG(t2m) FROM era5 WHERE time >= '2020-01-01'")

Queries keep partition pruning on dimension predicates and projection
pushdown: only the chunks and variables a query touches are read from
the source, only while it runs, and results stream back as Arrow record
batches.

The server has no authentication or TLS, and binds to ``127.0.0.1``
unless told otherwise. Its SQL is read-only: DDL, DML, and other
statements are rejected, so clients cannot read or write the server's
filesystem through it.
"""

from __future__ import annotations

import time
from typing import Any, TypeGuard

import xarray as xr

from .._native import FlightSqlServer as _NativeFlightSqlServer
from ..df import (
    Chunks,
    TableNames,
    group_vars_by_dims,
    resolve_table_names,
    shared_coord_arrays,
)
from ..reader import read_xarray_table
from .base import register_adapter

__all__ = ["FlightSQLServer", "serve"]

_WILDCARD_HOSTS = ("", "0.0.0.0", "::")


class FlightSQLServer:
    """A Flight SQL endpoint over lazily registered xarray Datasets.

    Register Datasets with
    [register][xarray_sql.backends.flight.FlightSQLServer.register] (or
    [xarray_sql.register][], which dispatches here), then start it with
    [serve][xarray_sql.backends.flight.FlightSQLServer.serve]. Datasets
    registered while the server runs are visible to the next query.

    The server runs on a background thread; the Python process must stay
    alive while it serves. In a script, block with
    [wait][xarray_sql.backends.flight.FlightSQLServer.wait]. It is also a
    context manager that shuts down on exit.
    """

    def __init__(self) -> None:
        self._native = _NativeFlightSqlServer()
        self._host: str | None = None
        self._port: int | None = None

    def register(
        self,
        name: str,
        ds: xr.Dataset,
        *,
        chunks: Chunks = None,
        table_names: TableNames = None,
    ) -> FlightSQLServer:
        """Register ``ds`` as a table named ``name``.

        A Dataset whose variables sit on different dimensions is split
        into one table per dimension group, in a SQL schema named
        ``name`` (``name.group``), exactly as
        [XarrayContext.from_dataset][xarray_sql.XarrayContext.from_dataset]
        registers it.

        Args:
            name: The table name, or the schema name for a
                mixed-dimension Dataset.
            ds: An xarray Dataset.
            chunks: Xarray-like chunks specification controlling partition
                granularity. Defaults to the Dataset's existing chunks.
            table_names: Maps a dimension group's exact dim tuple to the
                name its table takes.

        Returns:
            The server, to allow chaining.
        """
        groups = group_vars_by_dims(ds)
        names = resolve_table_names(ds, table_names)
        if len(groups) <= 1:
            self._native.register_table(name, read_xarray_table(ds, chunks))
            return self

        coord_arrays = shared_coord_arrays(ds)
        for dims, var_names in groups.items():
            table = read_xarray_table(
                ds[var_names], chunks, coord_arrays=coord_arrays
            )
            self._native.register_table(names[dims], table, schema=name)
        return self

    def serve(self, host: str = "127.0.0.1", port: int = 0) -> FlightSQLServer:
        """Start accepting connections on a background thread.

        Args:
            host: The interface to bind. The default accepts only local
                connections; ``"0.0.0.0"`` accepts them from anywhere the
                network allows (the server has no authentication).
            port: The TCP port. ``0`` (default) picks a free one; read it
                back from [port][xarray_sql.backends.flight.FlightSQLServer.port].

        Returns:
            The server, to allow chaining.
        """
        self._port = self._native.serve(host, port)
        self._host = host
        return self

    @property
    def port(self) -> int:
        """The TCP port the server is bound to."""
        if self._port is None:
            raise RuntimeError("the server has not been started")
        return self._port

    @property
    def uri(self) -> str:
        """A ``grpc://`` URI a local client can connect to."""
        host = self._host if self._host not in _WILDCARD_HOSTS else "localhost"
        if host is not None and ":" in host:
            host = f"[{host}]"
        return f"grpc://{host}:{self.port}"

    @property
    def is_running(self) -> bool:
        """Whether the server is accepting connections."""
        return self._native.is_running()

    def wait(self, poll_interval: float = 0.5) -> None:
        """Block until the server stops; Ctrl+C shuts it down."""
        try:
            while self._native.is_running():
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self) -> None:
        """Stop accepting connections and let in-flight queries finish."""
        self._native.shutdown()

    def __enter__(self) -> FlightSQLServer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        state = f"serving on {self.uri}" if self.is_running else "stopped"
        return f"FlightSQLServer({state})"


def serve(
    datasets: dict[str, xr.Dataset],
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    chunks: Chunks = None,
    table_names: TableNames = None,
) -> FlightSQLServer:
    """Serve Datasets over Arrow Flight SQL, without copying them.

    Starts a [FlightSQLServer][xarray_sql.backends.flight.FlightSQLServer]
    on a background thread with each Dataset registered under its key::

        server = xql.serve({"era5": ds}, port=8815)
        print(server.uri)   # grpc://127.0.0.1:8815
        server.wait()       # in a script: block until Ctrl+C

    Any Flight SQL client can then query the Datasets; with ADBC,
    ``xql.to_dataset(cursor, template=ds)`` turns a result back into a
    labeled Dataset on the client.

    Args:
        datasets: Table name to Dataset.
        host: The interface to bind; ``127.0.0.1`` by default.
        port: The TCP port; ``0`` (default) picks a free one.
        chunks: Chunks specification applied to every Dataset.
        table_names: Dimension-group naming applied to every Dataset.

    Returns:
        The running server.
    """
    server = FlightSQLServer()
    for name, ds in datasets.items():
        server.register(name, ds, chunks=chunks, table_names=table_names)
    return server.serve(host, port)


@register_adapter
class FlightSQLAdapter:
    """Registers Datasets on a [FlightSQLServer][xarray_sql.backends.flight.FlightSQLServer]."""

    @staticmethod
    def matches(con: object) -> TypeGuard[FlightSQLServer]:
        return isinstance(con, FlightSQLServer)

    @staticmethod
    def register(
        con: FlightSQLServer,
        name: str,
        ds: xr.Dataset,
        *,
        chunks: Chunks = None,
        table_names: TableNames = None,
        **kwargs: Any,
    ) -> FlightSQLServer:
        if kwargs:
            raise TypeError(
                f"unexpected options for a FlightSQLServer: {sorted(kwargs)}"
            )
        return con.register(name, ds, chunks=chunks, table_names=table_names)
