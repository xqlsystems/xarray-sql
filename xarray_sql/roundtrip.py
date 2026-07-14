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

import os
import tempfile
import weakref
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
from .lazyscan import LazyResultHandle, PolarsHandle, resolve_lazy_handle


def _guarded(
    batches: Any, schema: pa.Schema, max_bytes: int | None
) -> list[pa.RecordBatch]:
    """Collect a batch iterable, erroring cleanly past ``max_bytes``.

    The bounded-memory ladder's middle rung: a result that would blow
    past the budget raises with the running size instead of exhausting
    memory, before the (larger) dense reconstruction is even attempted.
    """
    if max_bytes is None:
        return list(batches)
    out: list[pa.RecordBatch] = []
    total = 0
    for batch in batches:
        total += batch.nbytes
        if total > max_bytes:
            raise ValueError(
                f"result exceeded max_result_bytes={max_bytes:,} while "
                f"materializing (>= {total:,} bytes after "
                f"{sum(b.num_rows for b in out) + batch.num_rows:,} rows). "
                "Aggregate further, or reconstruct lazily with chunks=."
            )
        out.append(batch)
    return out


def _result_to_batches(
    result: Any, max_bytes: int | None = None
) -> tuple[pa.Schema, list[pa.RecordBatch]]:
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
        return result.schema, _guarded(result, result.schema, max_bytes)
    if hasattr(result, "__arrow_c_stream__"):
        reader = pa.RecordBatchReader.from_stream(result)
        return reader.schema, _guarded(reader, reader.schema, max_bytes)
    if hasattr(result, "fetch_record_batch"):
        reader = result.fetch_record_batch()
        return reader.schema, _guarded(reader, reader.schema, max_bytes)
    if hasattr(result, "to_arrow_table"):
        table = result.to_arrow_table()
        return table.schema, table.to_batches()
    handle = resolve_lazy_handle(result)
    if handle is not None:
        # Re-executable results without a stream protocol (a Polars
        # LazyFrame): execute once through the handle, unnarrowed.
        schema = handle.schema()
        batches = handle.fetch({}, list(schema.names))
        return schema, _guarded(batches, schema, max_bytes)
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
    max_result_bytes: int | None = None,
    spill: bool | str | os.PathLike = False,
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
        max_result_bytes: Optional budget for the eager path. Raises a
            clean ``ValueError`` (with the running size) as soon as the
            materializing result exceeds it — both while collecting the
            Arrow stream and before allocating the dense arrays —
            instead of exhausting memory. ``None`` (default) means
            unlimited.
        spill: Chunked reconstruction from a one-pass on-disk spill
            instead of per-window re-execution: the result is streamed
            *once* (bounded memory) into a temporary Parquet file, and
            windows re-execute against that file. This serves the two
            results the re-execution path cannot — DuckDB relations and
            one-shot Arrow streams — and trades per-window narrowness
            for a single full pass plus temporary disk. ``True`` spills
            to the system temp dir; a path spills into that directory.
            The file is removed when the returned Dataset is garbage
            collected. Requires Polars; only valid with ``chunks=``.

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
    if spill and chunks is None:
        raise ValueError(
            "spill= only applies to chunked reconstruction; pass chunks=."
        )

    if chunks is not None:
        if spill:
            return _to_dataset_spilled(
                result, dims, template, sparsity, fill_value, chunks,
                coords, spill,
            )
        return _to_dataset_lazy(
            result, dims, template, sparsity, fill_value, chunks, coords
        )

    schema, batches = _result_to_batches(result, max_result_bytes)
    field_names = [f.name for f in schema]
    field_types = {f.name: f.type for f in schema}

    dims = _resolve_dims(dims, template, field_names)

    if max_result_bytes is not None:
        _check_dense_size(
            batches, dims, field_names, field_types, max_result_bytes
        )
    ds = _dataset_from_batches(batches, dims, field_names, field_types)
    return _finish_dataset(
        ds, dims, template, sparsity, fill_value, None, field_types
    )


def _check_dense_size(
    batches: list[pa.RecordBatch],
    dims: list[str],
    field_names: list[str],
    field_types: dict[str, Any],
    max_bytes: int,
) -> None:
    """Error before allocating dense arrays larger than the budget.

    The dense grid is the coordinate product, which for sparse results
    can dwarf the Arrow input; check it against the same budget before
    a single output array is allocated.
    """
    sizes = []
    for d in dims:
        values = set()
        for batch in batches:
            column = batch.column(batch.schema.names.index(d))
            values.update(column.to_pylist())
        sizes.append(len(values))
    cells = int(np.prod(sizes)) if sizes else 0
    total = sum(
        cells * np.dtype(field_types[n].to_pandas_dtype()).itemsize
        for n in field_names
        if n not in dims
    )
    if total > max_bytes:
        raise ValueError(
            f"dense reconstruction needs {total:,} bytes "
            f"({cells:,} grid cells), over max_result_bytes="
            f"{max_bytes:,}. Aggregate further, or reconstruct lazily "
            "with chunks=."
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
    _handle: LazyResultHandle | None = None,
) -> xr.Dataset:
    """The chunked reconstruction behind ``to_dataset(chunks=...)``."""
    handle = _handle if _handle is not None else resolve_lazy_handle(result)
    if handle is None:
        raise TypeError(
            "chunks= requires a re-executable engine result (a Polars "
            "LazyFrame/DataFrame or a DataFusion DataFrame); got "
            f"{type(result).__qualname__}, which is a one-shot stream. "
            "Pass the engine's lazy handle instead of a materialized "
            "result, add spill=True to reconstruct from a one-pass "
            "on-disk spill, or use chunks=None."
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
            "Add spill=True to reconstruct from a one-pass on-disk "
            "spill, use chunks=None (eager), or run the query through "
            "Polars (pl.scan_pyarrow_dataset(xql.arrow_dataset(ds))) "
            "or a DataFusion context."
        )
    ds = _build_lazy_scan(
        handle, dims, field_names, field_types, coord_arrays=coord_arrays
    )
    return _finish_dataset(
        ds, dims, template, sparsity, fill_value, resolved, field_types
    )


def _to_dataset_spilled(
    result: Any,
    dims: list[str] | None,
    template: xr.Dataset | None,
    sparsity: Sparsity,
    fill_value: Any,
    chunks: Mapping[str, int] | str,
    coords: Literal["discover", "template"],
    spill: bool | str | os.PathLike,
) -> xr.Dataset:
    """Chunked reconstruction from a one-pass temporary Parquet spill.

    The result is streamed exactly once with bounded memory — through
    the engine handle where one exists (DuckDB spills on its dedicated
    engine thread; Polars uses its streaming sink), or straight from
    the Arrow stream for one-shot results — and the ordinary lazy
    reconstruction then runs against a Polars scan of the file, whose
    per-window predicates enjoy Parquet row-group pruning. The file is
    removed when the reconstruction handle is garbage collected.
    """
    import polars as pl

    directory = None if spill is True else os.fspath(spill)
    fd, path = tempfile.mkstemp(suffix=".parquet", dir=directory)
    os.close(fd)
    try:
        handle = resolve_lazy_handle(result)
        if handle is not None:
            handle.spill_parquet(path)
        else:
            _stream_to_parquet(result, path)
    except BaseException:
        os.unlink(path)
        raise
    spilled = PolarsHandle(pl.scan_parquet(path))
    weakref.finalize(spilled, _unlink_quietly, path)
    return _to_dataset_lazy(
        result, dims, template, sparsity, fill_value, chunks, coords,
        _handle=spilled,
    )


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _stream_to_parquet(result: Any, path: str) -> None:
    """Write a one-shot Arrow result to Parquet, batch by batch."""
    import pyarrow.parquet as pq

    if isinstance(result, pa.RecordBatch):
        schema, batches = result.schema, iter([result])
    elif isinstance(result, pa.Table):
        schema, batches = result.schema, iter(result.to_batches())
    elif isinstance(result, pa.RecordBatchReader):
        schema, batches = result.schema, result
    elif hasattr(result, "__arrow_c_stream__"):
        reader = pa.RecordBatchReader.from_stream(result)
        schema, batches = reader.schema, reader
    elif hasattr(result, "fetch_record_batch"):
        reader = result.fetch_record_batch()
        schema, batches = reader.schema, reader
    else:
        raise TypeError(
            f"cannot spill {type(result).__qualname__}: no readable "
            "Arrow stream."
        )
    with pq.ParquetWriter(path, schema) as writer:
        for batch in batches:
            writer.write_batch(batch)
