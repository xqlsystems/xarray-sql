"""How the databases behind ADBC drivers differ, in one table.

ADBC gives every database the same bulk-ingest call, but the SQL around
it and the drivers' support for it vary: how identifiers are quoted,
whether the database has schemas, whether its driver can create tables
or honor ``temporary=True``, and which Arrow types it can store. Each
[Dialect][xarray_sql.backends._adbc_dialects.Dialect] records those
facts for one database, so the adapter has one code path and adding a
database means adding a row.

A database missing from the table gets the defaults, which are what
ANSI SQL and the ADBC specification prescribe.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import pyarrow as pa

if TYPE_CHECKING:
    from adbc_driver_manager import dbapi

IngestMode = Literal["create", "append", "replace", "create_append"]

TableDDL = Callable[..., list[str]]
"""Builds the statements that create a table before an append-only ingest."""


@dataclasses.dataclass(frozen=True)
class Dialect:
    """What the ADBC adapter needs to know about one database."""

    name: str
    """The database's name, as used in error messages."""

    quote: str = '"'
    """The identifier quote: ``"`` in ANSI SQL, a backtick in MySQL-family
    and Hive-family SQL, where ``"era5"`` is a string."""

    schemas: bool = True
    """Whether a mixed-dimension Dataset can go in a schema of its own
    (``era5.surface``); otherwise its tables are flat (``era5_surface``)."""

    schema_kind: str = "SCHEMA"
    """The object ``CREATE ... IF NOT EXISTS`` makes to hold them."""

    target_schema: Literal["option", "default_database"] = "option"
    """How an ingest reaches a table in that schema: ADBC's target-schema
    option, or, where the driver ignores that option, by making the
    schema the connection's default database for the ingest."""

    temporary_tables: bool = True
    """Whether the driver honors ``temporary=True``."""

    durations: bool = True
    """Whether the database stores Arrow durations; if not, timedeltas are
    ingested as integer counts of their unit."""

    table_ddl: TableDDL | None = None
    """For drivers that can only append: creates each table beforehand."""

    def quote_identifier(self, identifier: str) -> str:
        """*identifier* as a quoted identifier in this database's SQL."""
        escaped = identifier.replace(self.quote, self.quote * 2)
        return f"{self.quote}{escaped}{self.quote}"


_CLICKHOUSE_TYPES = {
    pa.bool_(): "Bool",
    pa.int8(): "Int8",
    pa.int16(): "Int16",
    pa.int32(): "Int32",
    pa.int64(): "Int64",
    pa.uint8(): "UInt8",
    pa.uint16(): "UInt16",
    pa.uint32(): "UInt32",
    pa.uint64(): "UInt64",
    pa.float16(): "Float32",
    pa.float32(): "Float32",
    pa.float64(): "Float64",
    pa.string(): "String",
    pa.large_string(): "String",
    pa.binary(): "String",
    pa.large_binary(): "String",
    pa.date32(): "Date32",
}

_TIMESTAMP_PRECISION = {"s": 0, "ms": 3, "us": 6, "ns": 9}


def _clickhouse_type(field: pa.Field, key: bool) -> str:
    """The ClickHouse column type for an Arrow field.

    Timestamps without a zone are declared UTC, which is what their
    values mean; ClickHouse also parses string literals compared with a
    column in that column's zone, so ``time >= '2020-01-01'`` means UTC
    rather than the server's local time. Sort-key columns are not
    ``Nullable``. Floats are: the scan writes NaN as null so aggregates
    skip missing values, and a plain ``Float64`` column would store that
    null as 0.
    """
    arrow_type = field.type
    if pa.types.is_timestamp(arrow_type):
        precision = _TIMESTAMP_PRECISION[arrow_type.unit]
        zone = arrow_type.tz or "UTC"
        name = f"DateTime64({precision}, '{zone}')"
    elif arrow_type in _CLICKHOUSE_TYPES:
        name = _CLICKHOUSE_TYPES[arrow_type]
    else:
        raise TypeError(
            f"no ClickHouse column type for {field.name!r} of Arrow type "
            f"{arrow_type}; create the table yourself and register with "
            f'mode="append"'
        )
    if key or not field.nullable:
        return name
    return f"Nullable({name})"


def _clickhouse_ddl(
    table: str,
    schema: pa.Schema,
    dims: tuple[str, ...],
    *,
    mode: IngestMode,
    temporary: bool,
    database: str | None,
) -> list[str]:
    """Statements that prepare *table* for an append-mode ingest.

    ClickHouse's ADBC driver only appends, so the table is created here
    for every other mode. Tables are sorted by their dimensions, so
    ClickHouse's primary index skips data on dimension predicates the
    way chunk pruning does in the other engines.
    """
    if mode == "append":
        return []
    quote = CLICKHOUSE.quote_identifier
    target = quote(table)
    if database is not None:
        target = f"{quote(database)}.{target}"
    columns = ", ".join(
        f"{quote(field.name)} {_clickhouse_type(field, field.name in dims)}"
        for field in schema
    )
    kind = "TEMPORARY TABLE" if temporary else "TABLE"
    if temporary:
        engine = "ENGINE = Memory"
    else:
        order = ", ".join(quote(dim) for dim in dims) or "tuple()"
        engine = f"ENGINE = MergeTree ORDER BY ({order})"
    statements = []
    if mode == "replace":
        statements.append(f"DROP {kind} IF EXISTS {target}")
    exists = " IF NOT EXISTS" if mode == "create_append" else ""
    statements.append(f"CREATE {kind}{exists} {target} ({columns}) {engine}")
    return statements


CLICKHOUSE = Dialect(
    "ClickHouse",
    schema_kind="DATABASE",
    durations=False,
    table_ddl=_clickhouse_ddl,
)

DIALECTS: dict[str, Dialect] = {
    # Exercised by the test suite, against a live database or driver.
    "sqlite": Dialect("SQLite", schemas=False, durations=False),
    "duckdb": Dialect("DuckDB"),
    "postgresql": Dialect("PostgreSQL"),
    # MariaDB's server reports itself as MySQL.
    "mysql": Dialect(
        "MySQL",
        quote="`",
        schema_kind="DATABASE",
        target_schema="default_database",
    ),
    "clickhouse": CLICKHOUSE,
    "datafusion": Dialect("DataFusion", temporary_tables=False),
    # Trino's driver ignores temporary=True and creates a permanent table.
    "trino": Dialect("Trino", temporary_tables=False),
    # From the drivers' published feature tables and the databases' SQL
    # references; not exercised by the test suite.
    "spark": Dialect("Spark", quote="`", schemas=False, temporary_tables=False),
    "bigquery": Dialect("BigQuery", quote="`", temporary_tables=False),
    "databricks": Dialect("Databricks", quote="`", temporary_tables=False),
    "snowflake": Dialect("Snowflake", temporary_tables=False),
}
"""Known databases, keyed by a name their drivers' vendor names contain."""


def dialect_for(con: dbapi.Connection) -> Dialect:
    """The [Dialect][xarray_sql.backends._adbc_dialects.Dialect] of *con*.

    Drivers name their database in ``adbc_get_info``. The ClickHouse
    driver does not implement it, so only a driver without it is probed
    with a query against ClickHouse's ``system.one`` table.
    """
    try:
        info = con.adbc_get_info()
    except Exception:  # noqa: BLE001 — unimplemented; probe instead
        pass
    else:
        vendor = str(info.get("vendor_name") or "").lower()
        for key, dialect in DIALECTS.items():
            if key in vendor:
                return dialect
        return Dialect(str(info.get("vendor_name") or "the database"))
    try:
        with con.cursor() as cur:
            cur.execute("SELECT 1 FROM system.one")
            cur.fetchall()
    except Exception:  # noqa: BLE001
        return Dialect("the database")
    return CLICKHOUSE


def durations_as_integers(
    reader: pa.RecordBatchReader,
) -> pa.RecordBatchReader:
    """*reader* with duration columns as integer counts of their unit.

    The template's ``timedelta64`` unit matches the Arrow unit the scan
    derived from it, so [xarray_sql.to_dataset][] reads the counts back
    as the original durations.
    """
    fields = [
        pa.field(f.name, pa.int64(), f.nullable, f.metadata)
        if pa.types.is_duration(f.type)
        else f
        for f in reader.schema
    ]
    if all(f.type == g.type for f, g in zip(fields, reader.schema)):
        return reader
    schema = pa.schema(fields, metadata=reader.schema.metadata)
    return pa.RecordBatchReader.from_batches(
        schema, (batch.cast(schema) for batch in reader)
    )
