"""Unit tests for the cftime module (cftime ↔ Arrow bridge)."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

from xarray_sql import cftime as cft
from xarray_sql.df import _parse_schema


# -- Fixtures ---------------------------------------------------------------


@pytest.fixture
def ds_360day():
    """Synthetic 360-day calendar dataset."""
    import cftime

    times = [cftime.Datetime360Day(2000, m, 1) for m in range(1, 13)]
    return xr.Dataset(
        {"temp": ("time", np.arange(12, dtype=np.float32))},
        coords={"time": times},
    )


# -- Detection helpers ------------------------------------------------------


class TestDetection:
    def test_is_cftime_detects_cftime_array(self, rasm_ds):
        assert cft.is_cftime(rasm_ds.coords["time"].values)

    def test_is_cftime_rejects_datetime64(self):
        assert not cft.is_cftime(pd.date_range("2020-01-01", periods=10).values)

    def test_is_cftime_rejects_float(self):
        assert not cft.is_cftime(np.array([1.0, 2.0, 3.0]))

    def test_is_cftime_index_detects_cftime(self, rasm_ds):
        assert cft.is_cftime_index(rasm_ds, "time")

    def test_is_cftime_index_rejects_datetime64(self):
        ds = xr.tutorial.open_dataset("air_temperature")
        assert not cft.is_cftime_index(ds, "time")

    def test_is_cftime_index_rejects_nonexistent(self, rasm_ds):
        assert not cft.is_cftime_index(rasm_ds, "nonexistent")


# -- Calendar classification ------------------------------------------------


class TestCalendarClassification:
    def test_calendar_returns_noleap(self, rasm_ds):
        assert cft.calendar(rasm_ds, "time") == "noleap"

    def test_calendar_returns_360_day(self, ds_360day):
        assert cft.calendar(ds_360day, "time") == "360_day"

    def test_calendar_returns_none_for_datetime64(self):
        ds = xr.tutorial.open_dataset("air_temperature")
        assert cft.calendar(ds, "time") is None

    def test_noleap_is_gregorian_like(self):
        assert cft.is_gregorian_like("noleap")
        assert cft.is_gregorian_like("standard")
        assert cft.is_gregorian_like("proleptic_gregorian")
        assert cft.is_gregorian_like("all_leap")

    def test_360_day_is_not_gregorian_like(self):
        assert not cft.is_gregorian_like("360_day")
        assert not cft.is_gregorian_like("julian")


# -- Numeric conversion -----------------------------------------------------


class TestConversion:
    def test_to_microseconds_returns_int64(self, rasm_ds):
        us = cft.to_microseconds(rasm_ds.coords["time"].values)
        assert us.dtype == np.int64

    def test_to_microseconds_is_monotonic(self, rasm_ds):
        us = cft.to_microseconds(rasm_ds.coords["time"].values)
        assert np.all(np.diff(us) > 0)

    def test_to_microseconds_length_matches(self, rasm_ds):
        values = rasm_ds.coords["time"].values
        assert len(cft.to_microseconds(values)) == len(values)

    def test_to_offsets_returns_int64(self, ds_360day):
        values = ds_360day.coords["time"].values
        offsets = cft.to_offsets(values, cft.DEFAULT_UNITS, "360_day")
        assert offsets.dtype == np.int64

    def test_to_offsets_is_monotonic(self, ds_360day):
        values = ds_360day.coords["time"].values
        offsets = cft.to_offsets(values, cft.DEFAULT_UNITS, "360_day")
        assert np.all(np.diff(offsets) > 0)

    def test_convert_for_field_gregorian_like(self, rasm_ds):
        field = cft.arrow_field("time", cft.DEFAULT_UNITS, "noleap")
        result = cft.convert_for_field(rasm_ds.coords["time"].values, field)
        assert result.dtype == np.int64
        assert np.all(np.diff(result) > 0)

    def test_convert_for_field_non_gregorian(self, ds_360day):
        field = cft.arrow_field("time", cft.DEFAULT_UNITS, "360_day")
        result = cft.convert_for_field(ds_360day.coords["time"].values, field)
        assert result.dtype == np.int64
        assert np.all(np.diff(result) > 0)


# -- Arrow schema helpers ---------------------------------------------------


class TestArrowField:
    def test_gregorian_like_produces_timestamp_us(self):
        field = cft.arrow_field("time", cft.DEFAULT_UNITS, "noleap")
        assert field.type == pa.timestamp("us")
        assert field.metadata[b"xarray:calendar"] == b"noleap"
        assert field.metadata[b"xarray:units"] == cft.DEFAULT_UNITS.encode()

    def test_non_gregorian_produces_int64(self):
        field = cft.arrow_field("time", cft.DEFAULT_UNITS, "360_day")
        assert field.type == pa.int64()
        assert field.metadata[b"xarray:calendar"] == b"360_day"


# -- Partition bounds -------------------------------------------------------


class TestPartitionBounds:
    def test_gregorian_like_returns_timestamp_ns_tag(self, rasm_ds):
        values = rasm_ds.coords["time"].values[:10]
        lo, hi, tag = cft.partition_bounds(values)
        assert tag == "timestamp_ns"
        assert lo < hi

    def test_non_gregorian_returns_int64_tag(self, ds_360day):
        values = ds_360day.coords["time"].values
        lo, hi, tag = cft.partition_bounds(values)
        assert tag == "int64"
        assert lo < hi

    def test_out_of_int64_range_returns_none(self):
        # Year-1 gregorian dates exceed the int64 nanosecond range, so no
        # pruning bound can be reported; the caller skips the dimension.
        values = xr.date_range(
            "0001-01-01", periods=3, freq="100YS", use_cftime=True
        ).values
        assert cft.partition_bounds(values) is None


# -- Integration with _parse_schema ----------------------------------------


class TestParseSchemaIntegration:
    def test_noleap_produces_timestamp_us(self, rasm_ds):
        schema = _parse_schema(rasm_ds[["Tair"]])
        time_field = schema.field("time")
        assert time_field.type == pa.timestamp("us")
        assert time_field.metadata[b"xarray:calendar"] == b"noleap"

    def test_360day_produces_int64(self, ds_360day):
        schema = _parse_schema(ds_360day)
        time_field = schema.field("time")
        assert time_field.type == pa.int64()
        assert time_field.metadata[b"xarray:calendar"] == b"360_day"

    def test_datetime64_unchanged(self):
        ds = xr.tutorial.open_dataset("air_temperature")
        schema = _parse_schema(ds)
        time_field = schema.field("time")
        assert pa.types.is_timestamp(time_field.type)
        assert time_field.metadata is None  # no xarray: metadata for native
