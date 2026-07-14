"""Engine-agnostic round-trip: Arrow query results → labeled ``xr.Dataset``.

The second seam of xarray-sql. Any engine's result — a DuckDB relation,
a ``pyarrow.Table``, a ``pyarrow.RecordBatchReader``, or any object
implementing the Arrow PyCapsule stream protocol — plus the registered
Dataset as a *template* is enough to rebuild a labeled, metadata-carrying
Dataset. Nothing here is engine-specific: results arrive as Arrow record
batches regardless of which engine executed the SQL.

Reconstruction is eager by default (the result is materialized once
into a dense in-memory Dataset). Passing ``chunks=`` selects the
lazy/chunked path instead: data variables are reconstructed on access,
window by window, by re-executing the engine's query narrowed to each
chunk's coordinate range. That requires the result to be
*re-executable* — a DuckDB relation, a Polars LazyFrame (or eager
DataFrame), or a DataFusion DataFrame — not a one-shot Arrow stream;
see :mod:`xarray_sql.lazyscan`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import xarray as xr

from .ds import (
    Sparsity,
    XarrayDataFrame,
    _apply_template,
    _build_lazy_scan,
    _dataset_from_batches,
    _ds_var_dims,
    _finish_dataset,
)
from .lazyscan import resolve_lazy_handle


def _result_to_batches(result: Any) -> tuple[pa.Schema, list[pa.RecordBatch]]:
    """Normalize an engine result into ``(schema, record batches)``.

    Accepts, in probe order:

    1. ``pyarrow.Table`` / ``pyarrow.RecordBatch``
    2. ``pyarrow.RecordBatchReader``
    3. Any object implementing ``__arrow_c_stream__`` (the Arrow
       PyCapsule protocol) — DuckDB relations qualify on duckdb >= 1.1.
    4. Objects with a ``fetch_record_batch()`` method (DuckDB relations
       on older versions).
    5. Objects with a ``to_arrow_table()`` method (DataFusion DataFrames
       and the :class:`~xarray_sql.ds.XarrayDataFrame` wrapper).
    """
    if isinstance(result, pa.RecordBatch):
        return result.schema, [result]
    if isinstance(result, pa.Table):
        return result.schema, result.to_batches()
    if isinstance(result, pa.RecordBatchReader):
        return result.schema, list(result)
    if hasattr(result, "__arrow_c_stream__"):
        reader = pa.RecordBatchReader.from_stream(result)
        return reader.schema, list(reader)
    if hasattr(result, "fetch_record_batch"):
        reader = result.fetch_record_batch()
        return reader.schema, list(reader)
    if hasattr(result, "to_arrow_table"):
        table = result.to_arrow_table()
        return table.schema, table.to_batches()
    handle = resolve_lazy_handle(result)
    if handle is not None:
        # Re-executable results without a stream protocol (a Polars
        # LazyFrame): execute once through the handle, unnarrowed.
        schema = handle.schema()
        return schema, handle.fetch({}, list(schema.names))
    raise TypeError(
        f"Cannot read an Arrow stream from {type(result).__qualname__}; "
        "expected a pyarrow Table/RecordBatch/RecordBatchReader, an object "
        "implementing __arrow_c_stream__, or an engine result exposing "
        "fetch_record_batch()/to_arrow_table()."
    )


def to_dataset(
    result: Any,
    dims: list[str] | None = None,
    template: xr.Dataset | None = None,
    sparsity: Sparsity = "result",
    fill_value: Any = np.nan,
    chunks: Mapping[str, int] | str | None = None,
    coords: Literal["discover", "template"] = "discover",
) -> xr.Dataset:
    """Convert an engine's Arrow result into a labeled ``xr.Dataset``.

    The engine-agnostic counterpart of
    :meth:`XarrayDataFrame.to_dataset`: SQL in, array out, for engines
    xarray-sql does not wrap in a session of its own.

    Example (DuckDB)::

        con = duckdb.connect()
        xql.register(con, "era5", ds)
        rel = con.sql(
            "SELECT time, lat, lon, AVG(t2m) AS t2m FROM era5 "
            "GROUP BY time, lat, lon"
        )
        out = xql.to_dataset(rel, template=ds)

    Args:
        result: The engine's query result: a ``pyarrow.Table``,
            ``RecordBatch`` or ``RecordBatchReader``, any object
            implementing ``__arrow_c_stream__`` (DuckDB relations), or an
            object with ``fetch_record_batch()`` / ``to_arrow_table()``.
            The result is consumed once.
        dims: Result columns to use as Dataset dimensions. When ``None``,
            defaults to the ``template``'s dimensions that survive into
            the result columns (so aggregations that drop dims round-trip
            on the remaining ones). Either ``dims`` or ``template`` must
            be given.
        template: The source Dataset registered with the engine. Recovers
            metadata the tabular pivot strips (attrs, encoding, non-dim
            coordinates, dim-coord dtype) and provides the ``dims``
            default.
        sparsity: ``"result"`` (default) keeps only dim values present in
            the result. ``"template"`` reindexes to the template's full
            coord ranges, filling absent cells with ``fill_value``.
        fill_value: Fill for ``sparsity="template"``. Defaults to NaN.
        chunks: ``None`` (default) materializes eagerly. A mapping
            (e.g. ``{"time": 100}``), ``"auto"``, or ``"inherit"``
            selects the lazy/chunked path: data variables are
            reconstructed window by window on access, each window
            re-executing the engine's query narrowed to its coordinate
            range (over a table registered through xarray-sql, that
            filter flows back into chunk pruning at the source).
            Requires a re-executable ``result`` — a DuckDB relation, a
            Polars LazyFrame/DataFrame, or a DataFusion DataFrame.
        coords: How the lazy path learns each dimension's coordinate
            values. ``"discover"`` (default) runs one ``DISTINCT`` query
            per dim — correct for any query. ``"template"`` trusts the
            template's coord arrays instead, skipping discovery; only
            valid when the result spans the template's full extent (an
            unfiltered scan), and requires ``template=``.

    Returns:
        An ``xr.Dataset`` with ``dims`` as dimensions and the remaining
        result columns as data variables — dense and in-memory by
        default, lazily chunked when ``chunks`` is given.

    Raises:
        ValueError: When neither ``dims`` nor ``template`` resolves the
            dimension columns, a requested dim is missing from the result,
            or ``sparsity="template"`` is used without a template.
        TypeError: When ``result`` exposes no readable Arrow stream, or
            ``chunks`` is requested for a one-shot stream that cannot be
            re-executed.
    """
    if sparsity not in ("result", "template"):
        raise ValueError(
            f"sparsity must be 'result' or 'template', got {sparsity!r}"
        )
    if sparsity == "template" and template is None:
        raise ValueError("sparsity='template' requires template= to be given")
    if coords not in ("discover", "template"):
        raise ValueError(
            f"coords must be 'discover' or 'template', got {coords!r}"
        )
    if coords == "template" and template is None:
        raise ValueError("coords='template' requires template= to be given")

    if chunks is not None:
        return _to_dataset_lazy(
            result, dims, template, sparsity, fill_value, chunks, coords
        )

    schema, batches = _result_to_batches(result)
    field_names = [f.name for f in schema]
    field_types = {f.name: f.type for f in schema}

    dims = _resolve_dims(dims, template, field_names)

    ds = _dataset_from_batches(batches, dims, field_names, field_types)
    return _finish_dataset(
        ds, dims, template, sparsity, fill_value, None, field_types
    )


def _resolve_dims(
    dims: list[str] | None,
    template: xr.Dataset | None,
    field_names: list[str],
) -> list[str]:
    """Dimension columns, inferred from the template when not given."""
    if dims is None:
        if template is None:
            raise ValueError(
                "dims cannot be inferred without a template; pass "
                "dims=[...] or template=<the registered Dataset>."
            )
        dims = [d for d in _ds_var_dims(template) if d in field_names]
        if not dims:
            raise ValueError(
                "dims cannot be inferred: no template dimension survives "
                "in the result columns. Pass dims=[...] explicitly."
            )
    missing = [d for d in dims if d not in field_names]
    if missing:
        raise ValueError(
            f"dims {missing} are not columns of the result {field_names}."
        )
    return dims


def _to_dataset_lazy(
    result: Any,
    dims: list[str] | None,
    template: xr.Dataset | None,
    sparsity: Sparsity,
    fill_value: Any,
    chunks: Mapping[str, int] | str,
    coords: Literal["discover", "template"],
) -> xr.Dataset:
    """The chunked reconstruction behind ``to_dataset(chunks=...)``."""
    handle = resolve_lazy_handle(result)
    if handle is None:
        raise TypeError(
            "chunks= requires a re-executable engine result (a Polars "
            "LazyFrame/DataFrame or a DataFusion DataFrame); got "
            f"{type(result).__qualname__}, which is a one-shot stream. "
            "Pass the engine's lazy handle instead of a materialized "
            "result, or use chunks=None."
        )
    schema = handle.schema()
    field_names = [f.name for f in schema]
    field_types = {f.name: f.type for f in schema}
    dims = _resolve_dims(dims, template, field_names)

    coord_arrays = None
    if coords == "template":
        assert template is not None
        missing = [d for d in dims if d not in template.coords]
        if missing:
            raise ValueError(
                f"coords='template' requires the template to carry coords "
                f"for every dim; missing {missing}."
            )
        coord_arrays = {
            d: np.asarray(template.coords[d].values) for d in dims
        }

    resolved = XarrayDataFrame._resolve_chunks(chunks, template, dims)
    if resolved is None:
        # "inherit" with no chunked source dimension to inherit from:
        # eager is the right execution, exactly as on the wrapper path.
        batches = handle.fetch({}, field_names)
        ds = _dataset_from_batches(batches, dims, field_names, field_types)
        return _finish_dataset(
            ds, dims, template, sparsity, fill_value, None, field_types
        )
    if not getattr(handle, "supports_chunked", True):
        raise NotImplementedError(
            "Chunked reconstruction is not supported for "
            f"{type(result).__qualname__}: re-executing a DuckDB "
            "relation from worker threads intermittently deadlocks in "
            "duckdb-python when the query scans a Python-backed table "
            "(see xarray_sql.lazyscan.DuckDBHandle.supports_chunked). "
            "Use chunks=None (eager), or run the query through Polars "
            "(pl.scan_pyarrow_dataset(xql.arrow_dataset(ds))) or a "
            "DataFusion context, which support chunked round-trips."
        )
    ds = _build_lazy_scan(
        handle, dims, field_names, field_types, coord_arrays=coord_arrays
    )
    return _finish_dataset(
        ds, dims, template, sparsity, fill_value, resolved, field_types
    )
