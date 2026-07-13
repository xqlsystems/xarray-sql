"""Engine-agnostic round-trip: Arrow query results → labeled ``xr.Dataset``.

The second seam of xarray-sql. Any engine's result — a DuckDB relation,
a ``pyarrow.Table``, a ``pyarrow.RecordBatchReader``, or any object
implementing the Arrow PyCapsule stream protocol — plus the registered
Dataset as a *template* is enough to rebuild a labeled, metadata-carrying
Dataset. Nothing here is engine-specific: results arrive as Arrow record
batches regardless of which engine executed the SQL.

This module implements the eager path only: the result is materialized
once into a dense in-memory Dataset. For DataFusion, the richer
lazy/chunked reconstruction lives on
:meth:`~xarray_sql.ds.XarrayDataFrame.to_dataset`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import xarray as xr

from .ds import (
    Sparsity,
    _apply_template,
    _dataset_from_batches,
    _ds_var_dims,
)


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

    Returns:
        A dense in-memory ``xr.Dataset`` with ``dims`` as dimensions and
        the remaining result columns as data variables.

    Raises:
        ValueError: When neither ``dims`` nor ``template`` resolves the
            dimension columns, a requested dim is missing from the result,
            or ``sparsity="template"`` is used without a template.
        TypeError: When ``result`` exposes no readable Arrow stream.
    """
    if sparsity not in ("result", "template"):
        raise ValueError(
            f"sparsity must be 'result' or 'template', got {sparsity!r}"
        )
    if sparsity == "template" and template is None:
        raise ValueError("sparsity='template' requires template= to be given")

    schema, batches = _result_to_batches(result)
    field_names = [f.name for f in schema]
    field_types = {f.name: f.type for f in schema}

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

    ds = _dataset_from_batches(batches, dims, field_names, field_types)

    if sparsity == "template":
        assert template is not None
        indexers = {
            d: template.coords[d].values
            for d in dims
            if d in template.coords and d in template.dims
        }
        if indexers:
            ds = ds.reindex(indexers, fill_value=fill_value)

    if template is not None:
        ds = _apply_template(ds, template)
    return ds
