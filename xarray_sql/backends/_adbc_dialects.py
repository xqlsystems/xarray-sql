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

import numpy as np
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

    unsigned: bool = True
    """Whether the database stores unsigned integers. If not, they are
    widened to the next signed width, and a ``uint64`` above the int64
    range is refused rather than wrapped."""

    timestamps_as_text: bool = False
    """For databases without a time type: write times as text like
    ``2021-01-01 04:00:00.000000000``, which compares correctly with a
    literal such as ``'2021-01-01 04:00:00'`` (SQLite's own format)."""

    timestamp_unit: Literal["ns", "us"] = "ns"
    """The finest time resolution the database stores."""

    folds: Literal["lower", "upper"] | None = None
    """How the database folds unquoted identifiers; names it would fold
    must be quoted in queries."""

    analyze: bool = False
    """Whether to ``ANALYZE`` a table after ingest. PostgreSQL gathers
    statistics only in the background, after a commit; until then its
    planner guesses, and a join over a freshly registered table can pick
    a nested loop that runs for hours instead of seconds."""

    table_ddl: TableDDL | None = None
    """For drivers that can only append: creates each table beforehand."""

    schema_ddl: Callable[[str], str] | None = None
    """For SQL without ``CREATE ... IF NOT EXISTS``: the statement that
    creates the schema of a given name unless it exists."""

    def quote_identifier(self, identifier: str) -> str:
        """*identifier* as a quoted identifier in this database's SQL."""
        escaped = identifier.replace(self.quote, self.quote * 2)
        return f"{self.quote}{escaped}{self.quote}"

    def create_schema_sql(self, name: str) -> str:
        """The statement that creates schema *name* unless it exists."""
        if self.schema_ddl is not None:
            return self.schema_ddl(name)
        target = self.quote_identifier(name)
        return f"CREATE {self.schema_kind} IF NOT EXISTS {target}"


def _tsql_literal(text: str) -> str:
    """*text* as a T-SQL Unicode string literal."""
    return "N'" + text.replace("'", "''") + "'"


def _sql_server_schema_ddl(name: str) -> str:
    """Creates schema *name* unless it exists, in T-SQL.

    T-SQL has no ``CREATE SCHEMA IF NOT EXISTS``, and ``CREATE SCHEMA``
    must be alone in its batch, hence the dynamic ``EXEC``.
    """
    create = f"CREATE SCHEMA {SQL_SERVER.quote_identifier(name)}"
    return (
        f"IF SCHEMA_ID({_tsql_literal(name)}) IS NULL "
        f"EXEC({_tsql_literal(create)})"
    )


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

SQL_SERVER = Dialect(
    "SQL Server",
    durations=False,
    unsigned=False,
    timestamp_unit="us",
    schema_ddl=_sql_server_schema_ddl,
)

DIALECTS: dict[str, Dialect] = {
    # Exercised by the test suite, against a live database or driver.
    "sqlite": Dialect(
        "SQLite",
        schemas=False,
        durations=False,
        unsigned=False,
        timestamps_as_text=True,
    ),
    "duckdb": Dialect("DuckDB"),
    # PostgreSQL wraps a uint64 above the int64 range without an error.
    "postgresql": Dialect(
        "PostgreSQL",
        unsigned=False,
        timestamp_unit="us",
        folds="lower",
        analyze=True,
    ),
    # MariaDB's server reports itself as MySQL.
    "mysql": Dialect(
        "MySQL",
        quote="`",
        schema_kind="DATABASE",
        target_schema="default_database",
        unsigned=False,
        timestamp_unit="us",
    ),
    "clickhouse": CLICKHOUSE,
    "datafusion": Dialect("DataFusion", temporary_tables=False, folds="lower"),
    # Trino's driver ignores temporary=True and creates a permanent table.
    "trino": Dialect("Trino", temporary_tables=False, unsigned=False),
    # Temporary tables are queried as #name.
    "sql server": SQL_SERVER,
    # From the drivers' published feature tables and the databases' SQL
    # references; not exercised by the test suite.
    "spark": Dialect("Spark", quote="`", schemas=False, temporary_tables=False),
    "bigquery": Dialect(
        "BigQuery",
        quote="`",
        temporary_tables=False,
        durations=False,
        unsigned=False,
    ),
    "databricks": Dialect("Databricks", quote="`", temporary_tables=False),
    "snowflake": Dialect("Snowflake", temporary_tables=False, folds="upper"),
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


_SIGNED_WIDTH = {
    pa.uint8(): pa.int16(),
    pa.uint16(): pa.int32(),
    pa.uint32(): pa.int64(),
    pa.uint64(): pa.int64(),
}


def _timestamps_as_text(array: pa.Array) -> pa.Array:
    """Times as space-separated ISO text.

    Whole seconds are written exactly as SQLite's ``datetime()`` writes
    them (``2021-01-01 04:00:00``), so equality and inclusive bounds
    against such a literal hold; a fraction of a second is kept where
    there is one, which still orders correctly.
    """
    values = array.to_numpy(zero_copy_only=False)
    whole = np.datetime_as_string(values, unit="s")
    exact = np.datetime_as_string(values)
    fractional = values != values.astype("datetime64[s]")
    text = np.where(fractional, exact, whole)
    text = np.char.replace(text, "T", " ", count=1)
    return pa.array(text, pa.string(), mask=np.asarray(array.is_null()))


def _conversion(field: pa.Field, dialect: Dialect) -> pa.DataType | None:
    """The type *field* must be ingested as, or ``None`` to keep it."""
    arrow_type = field.type
    if pa.types.is_duration(arrow_type) and not dialect.durations:
        return pa.int64()
    if pa.types.is_unsigned_integer(arrow_type) and not dialect.unsigned:
        return _SIGNED_WIDTH[arrow_type]
    if pa.types.is_timestamp(arrow_type) and dialect.timestamps_as_text:
        return pa.string()
    return None


def _convert(
    array: pa.Array, target: pa.DataType, dialect: Dialect
) -> pa.Array:
    if pa.types.is_string(target):
        return _timestamps_as_text(array)
    try:
        return array.cast(target)
    except pa.ArrowInvalid as exc:
        raise ValueError(
            f"{dialect.name} cannot store {array.type} values above the "
            f"{target} range, which it would otherwise wrap around; convert "
            f"the variable (e.g. to float64) before registering it."
        ) from exc


def ingestable(
    reader: pa.RecordBatchReader, dialect: Dialect
) -> pa.RecordBatchReader:
    """*reader* with every column in a type *dialect*'s database stores.

    Durations become integer counts of their unit (the template's
    ``timedelta64`` unit matches it, so [xarray_sql.to_dataset][] reads
    them back exactly), unsigned integers widen to the next signed width,
    and times become text where there is no time type.
    """
    targets = {i: _conversion(f, dialect) for i, f in enumerate(reader.schema)}
    targets = {i: t for i, t in targets.items() if t is not None}
    if not targets:
        return reader
    schema = reader.schema
    for i, target in targets.items():
        schema = schema.set(i, schema.field(i).with_type(target))

    def batches():
        for batch in reader:
            columns = list(batch.columns)
            for i, target in targets.items():
                columns[i] = _convert(columns[i], target, dialect)
            yield pa.RecordBatch.from_arrays(columns, schema=schema)

    return pa.RecordBatchReader.from_batches(schema, batches())
