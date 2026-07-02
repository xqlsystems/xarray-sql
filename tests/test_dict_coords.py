"""Coordinate columns are dictionary-encoded.

A dense grid repeats every coordinate value across the whole partition (a chunk
of shape ``(time, lat, lon)`` carries each latitude ``time × lon`` times).
Encoding coordinate columns as Arrow dictionaries keeps only the distinct values
plus small integer indices, which shrinks the bytes the engine moves and lets
``GROUP BY`` / equality ``JOIN`` on coordinates compare integer keys. These
tests pin that the coordinates are dictionary-encoded end to end and that the
values still round-trip correctly.
"""

import numpy as np
import pyarrow as pa
import xarray as xr

from xarray_sql import XarrayContext
from xarray_sql.df import _parse_schema, block_slices, iter_record_batches


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
    """Dimension coordinates are dictionary-typed; data variables are not."""
    schema = _parse_schema(_grid())
    for dim in ("time", "lat", "lon"):
        assert pa.types.is_dictionary(schema.field(dim).type), dim
    assert not pa.types.is_dictionary(schema.field("v").type)


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
