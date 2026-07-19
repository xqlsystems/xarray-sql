"""Bridge between cftime calendars and Arrow/DataFusion types.

cftime (https://unidata.github.io/cftime/) provides datetime objects for
calendars used in climate science — noleap, 360-day, all-leap, julian, etc.
Arrow and DataFusion have no native concept of non-Gregorian calendars, so
this module handles the conversion in two tiers:

* **Gregorian-like calendars** (standard, gregorian, proleptic_gregorian,
  noleap/365_day, all_leap/366_day): mapped to ``pa.timestamp('us')`` so
  that string-based SQL filters like ``WHERE time > '1980-01-01'`` work
  naturally.  Microsecond resolution avoids the 1678–2262 overflow of
  nanoseconds while preserving sub-second precision.

* **Non-Gregorian calendars** (360_day, julian): mapped to ``pa.int64()``
  with ``xarray:units`` and ``xarray:calendar`` metadata on the Arrow field.
  This preserves the original CF-convention encoding losslessly.  A
  ``cftime()`` DataFusion UDF (registered automatically by
  ``XarrayContext.from_dataset``) provides ergonomic SQL filtering.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import xarray as xr


# ---------------------------------------------------------------------------
# Calendar classification
# ---------------------------------------------------------------------------

#: Calendars close enough to proleptic Gregorian for ``pa.timestamp('us')``.
GREGORIAN_LIKE_CALENDARS: frozenset[str] = frozenset(
    {
        "standard",
        "gregorian",
        "proleptic_gregorian",
        "noleap",
        "365_day",
        "all_leap",
        "366_day",
    }
)

#: Default CF-convention units when no encoding is available on the coordinate.
#: Microseconds give sub-second precision and fit int64 for ±292 k years.
DEFAULT_UNITS: str = "microseconds since 1970-01-01T00:00:00"


def is_gregorian_like(calendar: str) -> bool:
    """Return True if *calendar* is close enough to Gregorian for ``pa.timestamp``."""
    return calendar in GREGORIAN_LIKE_CALENDARS


# ---------------------------------------------------------------------------
# Detection helpers (avoid materializing Dask/Zarr data where possible)
# ---------------------------------------------------------------------------


def is_cftime(values: np.ndarray) -> bool:
    """Check if a numpy array contains cftime datetime objects."""
    try:
        import cftime

        if values.dtype == np.dtype("O") and len(values) > 0:
            sample = values.ravel()[0]
            return isinstance(sample, cftime.datetime)
    except ImportError:
        pass
    return False


def is_cftime_index(ds: xr.Dataset, coord_name: str) -> bool:
    """Check if a coordinate uses a ``CFTimeIndex`` without materializing data."""
    try:
        idx = ds.indexes.get(coord_name)
        if idx is not None:
            from xarray import CFTimeIndex

            return isinstance(idx, CFTimeIndex)
    except (ImportError, AttributeError):
        pass
    return False


def calendar(ds: xr.Dataset, coord_name: str) -> str | None:
    """Return the calendar name for a cftime coordinate, or ``None``.

    Checks the xarray index first (no data materialization), then falls
    back to inspecting element 0 of the coordinate values.
    """
    try:
        idx = ds.indexes.get(coord_name)
        if idx is not None:
            from xarray import CFTimeIndex

            if isinstance(idx, CFTimeIndex):
                return str(idx.calendar)  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass
    try:
        values = ds.coords[coord_name].values
        if is_cftime(values):
            return str(values.ravel()[0].calendar)
    except (AttributeError, KeyError):
        pass
    return None


def encoding(ds: xr.Dataset, coord_name: str) -> tuple[str, str]:
    """Return ``(units, calendar)`` for a cftime coordinate.

    Reads xarray ``.encoding`` metadata (from the originating NetCDF file)
    first, falling back to :data:`DEFAULT_UNITS`.
    """
    cal = calendar(ds, coord_name) or "standard"
    enc = ds.coords[coord_name].encoding
    units = enc.get("units", DEFAULT_UNITS)
    return units, cal


# ---------------------------------------------------------------------------
# Numeric conversion
# ---------------------------------------------------------------------------


def to_microseconds(values) -> np.ndarray:
    """Convert cftime objects to int64 microseconds since Unix epoch.

    Used for Gregorian-like calendars.  Vectorised via ``cftime.date2num``
    (implemented in C).
    """
    import cftime as _cftime

    us = _cftime.date2num(
        values.ravel(),
        units=DEFAULT_UNITS,
        calendar=values.ravel()[0].calendar,
    )
    return np.asarray(us, dtype=np.float64).astype(np.int64)


def to_offsets(values, units: str, cal: str) -> np.ndarray:
    """Convert cftime objects to int64 offsets in the given *units*/*calendar*.

    Used for non-Gregorian calendars where data is stored as ``pa.int64()``.
    """
    import cftime as _cftime

    raw = _cftime.date2num(values.ravel(), units=units, calendar=cal)
    return np.asarray(raw, dtype=np.float64).astype(np.int64)


def convert_for_field(values, field: pa.Field) -> np.ndarray:
    """Convert cftime values to the numeric type dictated by *field*.

    Reads ``xarray:calendar`` and ``xarray:units`` from the field's Arrow
    metadata to choose between the timestamp path and the integer-offset path.
    """
    meta = field.metadata or {}
    cal = meta.get(b"xarray:calendar", b"standard").decode()
    units = meta.get(b"xarray:units", DEFAULT_UNITS.encode()).decode()
    if is_gregorian_like(cal):
        return to_microseconds(values)
    return to_offsets(values, units, cal)


# ---------------------------------------------------------------------------
# Partition pruning helpers
# ---------------------------------------------------------------------------


def partition_bounds(
    values,
) -> tuple[int, int, str] | None:
    """Return ``(min, max, dtype_tag)`` for a cftime coordinate slice.

    Gregorian-like calendars return nanosecond bounds tagged
    ``"timestamp_ns"`` (compatible with ``ScalarBound::TimestampNanos``
    in the Rust pruning layer).  Non-Gregorian calendars return int64
    offsets tagged ``"int64"``.

    Returns ``None`` when the nanosecond bound falls outside the int64 range
    (e.g. paleoclimate dates before ~1678), signalling the caller to skip
    pruning for that dimension rather than emit a bound the Rust layer would
    reject.
    """
    cal = values.ravel()[0].calendar
    if is_gregorian_like(cal):
        us = to_microseconds(values)
        lo, hi = int(us.min()) * 1_000, int(us.max()) * 1_000
        int64 = np.iinfo(np.int64)
        if lo < int64.min or hi > int64.max:
            return None
        return lo, hi, "timestamp_ns"
    offsets = to_offsets(values, DEFAULT_UNITS, cal)
    return int(offsets.min()), int(offsets.max()), "int64"


# ---------------------------------------------------------------------------
# Arrow schema helpers
# ---------------------------------------------------------------------------


def arrow_field(name: str, units: str, cal: str) -> pa.Field:
    """Build a ``pa.Field`` for a cftime coordinate.

    Gregorian-like → ``pa.timestamp('us')``; non-Gregorian → ``pa.int64()``.
    Both carry ``xarray:calendar`` and ``xarray:units`` metadata for
    round-trip fidelity.
    """
    meta = {
        b"xarray:calendar": cal.encode(),
        b"xarray:units": units.encode(),
    }
    if is_gregorian_like(cal):
        return pa.field(name, pa.timestamp("us"), metadata=meta)
    return pa.field(name, pa.int64(), metadata=meta)


# ---------------------------------------------------------------------------
# DataFusion UDF
# ---------------------------------------------------------------------------


def make_cftime_udf(units: str, calendar: str):
    """Create a DataFusion scalar UDF that converts date strings to int64 offsets.

    This enables ergonomic SQL filtering on non-Gregorian cftime columns::

        SELECT * FROM ds360 WHERE time > cftime('0500-01-01')

    The UDF parses the input string as a cftime datetime in the given
    calendar system and returns the corresponding int64 offset in the
    specified units.
    """
    import cftime as _cftime
    from datafusion import udf

    def _cftime_scalar(date_strings: pa.Array) -> pa.Array:
        results: list[int | None] = []
        for s in date_strings.to_pylist():
            if s is None:
                results.append(None)
                continue
            dt = _cftime.datetime.strptime(s, "%Y-%m-%d", calendar=calendar)
            val = _cftime.date2num(dt, units=units, calendar=calendar)
            results.append(int(val))
        return pa.array(results, type=pa.int64())

    return udf(
        _cftime_scalar,
        [pa.utf8()],
        pa.int64(),
        "immutable",
        "cftime",
    )
