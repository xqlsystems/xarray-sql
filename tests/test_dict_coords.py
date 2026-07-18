"""Coordinate columns are dictionary-encoded when it is worthwhile and safe.

A dense grid repeats every coordinate value across the whole partition (a chunk
of shape ``(time, lat, lon)`` carries each latitude ``time × lon`` times).
Encoding a coordinate as an Arrow dictionary keeps only the distinct values plus
an int32 index, shrinking the bytes the engine moves. We encode only when the
int32 key is strictly narrower than the value (8-byte float64/int64/timestamp
coordinates, and variable-width strings) and leave 4-byte float32/int32
coordinates dense — a narrower key would be needed to win there but overflows
under DataFusion's cross-batch dictionary concatenation. These tests pin which
coordinates get encoded, that the values round-trip correctly, and that the
overflow case no longer crashes.
"""

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr

from xarray_sql import XarrayContext
from xarray_sql.df import (
    _coord_index_type,
    _parse_schema,
    block_slices,
    iter_record_batches,
)


@pytest.mark.parametrize(
    "n_values, expected",
    [
        (1, pa.int32()),
        (2**31, pa.int32()),  # int32 holds indices 0..2**31-1
        (2**31 + 1, pa.int64()),  # fallback keeps huge axes representable
    ],
)
def test_coord_index_type_boundaries(n_values, expected):
    # int32 is the narrowest key we use: a narrower one can overflow when
    # DataFusion concatenates per-batch dictionaries across a streaming
    # aggregate (see test_float32_groupby_many_partitions_no_overflow).
    assert _coord_index_type(n_values) == expected


def _grid() -> xr.Dataset:
    return xr.Dataset(
        {"v": (("time", "lat", "lon"), np.random.rand(6, 4, 5))},
        coords={
            "time": xr.date_range("2020-01-01", periods=6, freq="D"),
            "lat": np.array([-90.0, -30.0, 30.0, 90.0]),
            "lon": np.array([0.0, 72.0, 144.0, 216.0, 288.0]),
        },
    )


def test_coordinate_fields_are_dictionary_encoded():
    """8-byte dimension coordinates are dictionary-typed; data variables are not.

    ``_grid`` uses float64/datetime coordinates (8 bytes), which are wider than
    the int32 key, so they are encoded.
    """
    schema = _parse_schema(_grid())
    for dim in ("time", "lat", "lon"):
        assert pa.types.is_dictionary(schema.field(dim).type), dim
    assert not pa.types.is_dictionary(schema.field("v").type)


def test_float32_coordinates_stay_dense():
    """4-byte coordinates are left dense: an int32 key is no narrower than the
    value, so a dictionary would be pure overhead (and a narrower key is unsafe).
    """
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.zeros((3, 4), dtype="float32"))},
        coords={
            "lat": np.array([-90.0, 0.0, 90.0], dtype="float32"),
            "lon": np.arange(4, dtype="int32"),
        },
    )
    schema = _parse_schema(ds)
    assert schema.field("lat").type == pa.float32()
    assert schema.field("lon").type == pa.int32()


def test_float32_groupby_many_partitions_no_overflow():
    """Regression: GROUP BY on float32 coordinates over many partitions must not
    overflow a dictionary key.

    With narrow keys, DataFusion concatenating the per-partition coordinate
    dictionaries across the aggregate overflowed the key type ("Dictionary key
    bigger than the key type"). float32 coordinates are now dense, so there is no
    coordinate dictionary to concatenate, and the aggregate matches xarray.
    """
    ds = xr.Dataset(
        {
            "air": (
                ("time", "lat", "lon"),
                np.random.rand(600, 20, 30).astype("float32"),
            )
        },
        coords={
            "time": np.arange(600),
            "lat": np.linspace(-90, 90, 20).astype("float32"),
            "lon": np.linspace(0, 359, 30).astype("float32"),
        },
    )
    ctx = XarrayContext()
    ctx.from_dataset("air", ds, chunks={"time": 6})  # 100 partitions
    got = ctx.sql(
        "SELECT lat, lon, AVG(air) AS m FROM air GROUP BY lat, lon"
    ).to_dataset(dims=["lat", "lon"])
    ref = ds["air"].mean("time")
    xr.testing.assert_allclose(
        got.m, ref.reindex_like(got.m), rtol=1e-5, atol=1e-6
    )


def test_iter_record_batches_emits_dictionary_coords():
    """A coordinate column arrives as a DictionaryArray with correct values."""
    ds = _grid()
    schema = _parse_schema(ds)
    block = next(block_slices(ds, chunks={"time": 6}))
    batch = next(iter_record_batches(ds.isel(block), schema, batch_size=1024))

    lat = batch.column(batch.schema.names.index("lat"))
    assert isinstance(lat, pa.DictionaryArray)
    # The dictionary holds only the distinct latitudes; decoding reproduces the
    # per-row values in row-major order.
    assert lat.dictionary.to_pylist() == [-90.0, -30.0, 30.0, 90.0]
    decoded = lat.to_numpy(zero_copy_only=False)
    assert decoded.shape == (batch.num_rows,)
    # First lon×... rows share lat=-90 (lat varies slower than lon in C order).
    assert decoded[0] == -90.0

    # Data variables stay plain (not dictionary-encoded).
    v = batch.column(batch.schema.names.index("v"))
    assert not isinstance(v, pa.DictionaryArray)


def test_groupby_coordinate_roundtrips_through_dictionary():
    """GROUP BY on a dictionary coordinate returns the same numbers as xarray."""
    ds = _grid()
    ctx = XarrayContext()
    ctx.from_dataset("grid", ds, chunks={"time": 3})

    got = ctx.sql(
        "SELECT lat, AVG(v) AS m FROM grid GROUP BY lat ORDER BY lat"
    ).to_dataset(dims=["lat"])
    ref = ds["v"].mean(["time", "lon"])

    xr.testing.assert_allclose(
        got.m, ref.reindex_like(got.m), rtol=1e-6, atol=1e-9
    )
