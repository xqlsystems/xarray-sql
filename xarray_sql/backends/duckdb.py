"""DuckDB engine adapter.

Registers a lazy ``xarray.Dataset`` on a ``duckdb.DuckDBPyConnection``
through the Arrow PyCapsule interface: DuckDB requests an Arrow C stream
each time the table is scanned, and :class:`XarrayArrowStream` builds a
fresh lazy reader per request, so the registered table is both lazy
(no data is read until a query executes) and re-queryable.

This adapter never imports the ``duckdb`` package — detection is by
connection type, and registration is a method call on the connection —
so DuckDB stays a purely optional dependency
(``pip install xarray-sql[duckdb]``).

Compared to the DataFusion adapter, this first version does no
projection or filter pushdown: every scan streams all columns of every
partition and DuckDB filters after the fact. Zarr-native scanning with
pushdown is what the `duckdb-zarr <https://github.com/xqlsystems/duckdb-zarr>`_
extension provides; this adapter instead covers everything xarray can
open (NetCDF, GRIB, Xee, CF decoding, in-memory) and pairs with
:func:`xarray_sql.to_dataset` for the labeled round-trip.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import xarray as xr

from ..df import Block, Chunks, DEFAULT_BATCH_SIZE, _ensure_default_indexes
from ..reader import XarrayRecordBatchReader
from .base import register_adapter


class XarrayArrowStream:
    """A re-scannable Arrow C-stream view over a lazy xarray Dataset.

    Arrow PyCapsule consumers (DuckDB among them) call
    ``__arrow_c_stream__`` once per scan. Each call constructs a fresh
    :class:`~xarray_sql.reader.XarrayRecordBatchReader` over the same
    lazy Dataset, so — unlike registering a ``pyarrow.RecordBatchReader``
    directly, which is exhausted after one query — the same registered
    table supports any number of queries, and data is only read while a
    query is executing.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        chunks: Chunks = None,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        _iteration_callback: (
            Callable[[Block, list[str] | None], None] | None
        ) = None,
    ):
        # Validate eagerly (same checks XarrayRecordBatchReader runs) so
        # registration fails fast instead of erroring mid-query.
        probe = XarrayRecordBatchReader(ds, chunks, batch_size=batch_size)
        self._ds = ds
        self._chunks = chunks
        self._batch_size = batch_size
        self._schema = probe.schema
        self._iteration_callback = _iteration_callback

    def __arrow_c_stream__(
        self, requested_schema: object | None = None
    ) -> object:
        reader = XarrayRecordBatchReader(
            self._ds,
            self._chunks,
            batch_size=self._batch_size,
            _iteration_callback=self._iteration_callback,
        )
        return reader.__arrow_c_stream__(requested_schema)

    def __arrow_c_schema__(self) -> object:
        return self._schema.__arrow_c_schema__()


@register_adapter
class DuckDBAdapter:
    """Registers Datasets on ``duckdb.DuckDBPyConnection`` connections."""

    @staticmethod
    def matches(con: Any) -> bool:
        # The connection class lives in ``duckdb`` or, in newer releases,
        # the ``_duckdb`` C-extension module.
        root = type(con).__module__.split(".")[0]
        return root in ("duckdb", "_duckdb")

    @staticmethod
    def register(
        con: Any,
        name: str,
        ds: xr.Dataset,
        *,
        chunks: Chunks = None,
    ) -> Any:
        ds = _ensure_default_indexes(ds)
        con.register(name, XarrayArrowStream(ds, chunks))
        return con
