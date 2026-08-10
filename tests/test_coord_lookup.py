"""Coordinate-value -> array-position lookup used by the reverse pivot.

``_scatter_batches_to_ndarray`` places each result row into a dense N-D
array by resolving its dim-coord values to integer positions through
``_CoordLookup``: an affine formula for uniformly spaced axes, a hash
table (``pd.Index``, built once, probed per batch) for irregular unique
axes, and an ``argsort``/``searchsorted`` fallback for axes with
duplicate values. These tests pin the correctness of each strategy on
out-of-order (shuffled) input -- the row order a parallel engine
produces -- plus the error contract for values absent from the axis.
"""

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr

from xarray_sql import to_dataset
from xarray_sql.ds import _CoordLookup, _scatter_batches_to_ndarray


def _irregular_axis(n: int, seed: int = 0) -> np.ndarray:
    """n unique, strictly increasing, non-uniformly spaced float64 values."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.exponential(scale=1.0, size=n) + 1e-6)


def _shuffled_batches(
    time: np.ndarray, station: np.ndarray, values: np.ndarray, batch_size: int
) -> list[pa.RecordBatch]:
    """The full (time, station) grid as shuffled row batches."""
    tt, ss = np.meshgrid(time, station, indexing="ij")
    flat_t, flat_s, flat_v = tt.ravel(), ss.ravel(), values.ravel()
    rng = np.random.default_rng(1)
    order = rng.permutation(flat_v.shape[0])
    flat_t, flat_s, flat_v = flat_t[order], flat_s[order], flat_v[order]
    schema = pa.schema(
        [
            ("time", pa.from_numpy_dtype(time.dtype)),
            ("station", pa.from_numpy_dtype(station.dtype)),
            ("v", pa.from_numpy_dtype(values.dtype)),
        ]
    )
    return [
        pa.RecordBatch.from_arrays(
            [
                pa.array(flat_t[i : i + batch_size]),
                pa.array(flat_s[i : i + batch_size]),
                pa.array(flat_v[i : i + batch_size]),
            ],
            schema=schema,
        )
        for i in range(0, flat_v.shape[0], batch_size)
    ]


def test_irregular_axis_scatter_matches_reference():
    """Shuffled rows over an irregular (hash-path) axis land correctly."""
    time = np.arange(6, dtype="int64")
    station = _irregular_axis(50)
    values = np.random.default_rng(2).standard_normal((6, 50)).astype("f4")
    batches = _shuffled_batches(time, station, values, batch_size=37)

    out = _scatter_batches_to_ndarray(
        batches=batches,
        dimension_columns=["time", "station"],
        requested={"time": time, "station": station},
        var_name="v",
        out_shape=(6, 50),
        dtype=np.dtype("float32"),
        drop_axes=[],
    )
    np.testing.assert_array_equal(out, values)


def test_descending_affine_axis_unchanged():
    """A descending uniformly spaced axis stays on the affine path."""
    lat = np.linspace(90.0, -90.0, 19)  # descending, uniform
    lookup = _CoordLookup(lat)
    assert lookup._affine is not None
    pos = lookup.positions_for(np.array([90.0, 0.0, -90.0]), dim="lat")
    np.testing.assert_array_equal(pos, [0, 9, 18])


def test_missing_value_raises_value_error():
    """A result value absent from the axis is a coordinate-discovery bug;
    it must fail loudly instead of scattering to a wrong cell."""
    station = _irregular_axis(10)
    lookup = _CoordLookup(station)
    assert lookup._hash_index is not None
    with pytest.raises(ValueError, match="dimension 'station'"):
        lookup.positions_for(np.array([-1.0]), dim="station")


def test_duplicate_axis_values_fall_back_to_search():
    """An axis with duplicate values cannot key a unique hash table; the
    sorted-search fallback keeps every value resolving to a position that
    holds it (which of the duplicate positions is returned is
    unspecified)."""
    axis = np.array([3.0, 1.0, 2.0, 1.0])  # 1.0 appears twice
    lookup = _CoordLookup(axis)
    assert lookup._hash_index is None and lookup._sorted_req is not None
    pos = lookup.positions_for(np.array([1.0, 2.0, 3.0]), dim="x")
    assert axis[pos[0]] == 1.0
    assert axis[pos[1]] == 2.0
    assert axis[pos[2]] == 3.0


def test_nan_in_irregular_axis_resolves():
    """A NaN dim value in the result resolves to the axis's NaN position
    (pandas index lookups treat NaN as equal to NaN)."""
    axis = np.array([2.0, np.nan, 5.0, 1.0])  # non-affine (NaN breaks it)
    lookup = _CoordLookup(axis)
    assert lookup._hash_index is not None
    pos = lookup.positions_for(np.array([np.nan, 1.0]), dim="x")
    np.testing.assert_array_equal(pos, [1, 3])


def test_to_dataset_roundtrips_shuffled_irregular_result():
    """End to end through the engine-agnostic ``to_dataset``: a shuffled
    Arrow result over an irregular axis reconstructs the exact Dataset."""
    time = np.arange(4, dtype="int64")
    station = _irregular_axis(30, seed=3)
    values = np.random.default_rng(4).standard_normal((4, 30)).astype("f8")
    template = xr.Dataset(
        {"v": (("time", "station"), values)},
        coords={"time": time, "station": station},
    )

    tt, ss = np.meshgrid(time, station, indexing="ij")
    rng = np.random.default_rng(5)
    order = rng.permutation(values.size)
    table = pa.table(
        {
            "time": tt.ravel()[order],
            "station": ss.ravel()[order],
            "v": values.ravel()[order],
        }
    )

    out = to_dataset(table, dims=["time", "station"], template=template)
    # Coordinate order follows first appearance in the (shuffled) result --
    # the documented behavior that lets an ORDER BY direction carry through
    # -- so compare on a common sort.
    xr.testing.assert_allclose(out.sortby(["time", "station"]), template)


def test_hash_index_reused_across_batches():
    """The pandas hash index is built once per reconstruction, not per
    batch -- the property the speedup rests on."""
    station = _irregular_axis(100)
    lookup = _CoordLookup(station)
    first = lookup._hash_index
    lookup.positions_for(station[:10], dim="station")
    lookup.positions_for(station[50:60], dim="station")
    assert lookup._hash_index is first
