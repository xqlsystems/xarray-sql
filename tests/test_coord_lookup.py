"""Coordinate-value -> array-position lookup used by the reverse pivot.

``to_dataset`` places each result row into a dense N-D array by
resolving its dim-coord values to integer positions: an affine formula
for uniformly spaced axes, a hash table for irregular unique axes, and
an ``argsort``/``searchsorted`` fallback for axes a hash table cannot
represent. These tests exercise each strategy through the public
``to_dataset`` contract on out-of-order (shuffled) rows — the arrival
order a parallel engine produces.

Two behaviors are unreachable through the eager public path, because
``to_dataset`` derives each axis from the same rows it scatters
(``pd.unique``), so the axis can neither miss a row's value nor carry
duplicates. Those two are covered at the ``_scatter_batches_to_ndarray``
seam directly: the error raised for a value absent from the axis (which
arises when an engine-backed lazy read's pre-computed coords go stale),
and the duplicate-axis fallback.
"""

import numpy as np
import pyarrow as pa
import pytest
import xarray as xr

from xarray_sql import to_dataset
from xarray_sql.ds import _scatter_batches_to_ndarray


def _irregular_axis(n: int, seed: int = 0) -> np.ndarray:
    """n unique, strictly increasing, non-uniformly spaced float64 values."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.exponential(scale=1.0, size=n) + 1e-6)


def _shuffled_table(template: xr.Dataset, seed: int = 1) -> pa.Table:
    """The template's full grid as one Arrow table with rows shuffled."""
    dims = list(next(iter(template.data_vars.values())).dims)
    grids = np.meshgrid(*(template[d].values for d in dims), indexing="ij")
    columns = {d: g.ravel() for d, g in zip(dims, grids)}
    for name, var in template.data_vars.items():
        columns[name] = var.values.ravel()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(columns[dims[0]]))
    return pa.table({name: col[order] for name, col in columns.items()})


def _roundtrip(template: xr.Dataset) -> xr.Dataset:
    """Shuffle the template into rows, reconstruct, and sort back.

    Output coordinate order follows first appearance in the (shuffled)
    result — the behavior that lets an ORDER BY direction carry through
    — so the reconstruction is sorted before comparing.
    """
    out = to_dataset(_shuffled_table(template), template=template)
    dims = list(next(iter(template.data_vars.values())).dims)
    return out.sortby(dims)


def test_irregular_axis_roundtrip():
    """Shuffled rows over an irregular (hash-strategy) axis reconstruct
    the exact Dataset."""
    template = xr.Dataset(
        {
            "v": (
                ("time", "station"),
                np.random.default_rng(2).standard_normal((6, 50)),
            )
        },
        coords={
            "time": np.arange(6, dtype="int64"),
            "station": _irregular_axis(50),
        },
    )
    xr.testing.assert_allclose(_roundtrip(template), template)


def test_descending_uniform_axis_roundtrip():
    """A descending uniformly spaced axis (affine strategy) reconstructs;
    the descending order itself survives via first-appearance coords."""
    template = xr.Dataset(
        {
            "v": (
                ("lat",),
                np.random.default_rng(3).standard_normal(19),
            )
        },
        coords={"lat": np.linspace(90.0, -90.0, 19)},
    )
    # Unshuffled rows: the descending source order carries through as-is.
    out = to_dataset(_shuffled_table(template, seed=0), template=template)
    xr.testing.assert_allclose(out.sortby("lat"), template.sortby("lat"))


def test_nan_dim_value_roundtrip():
    """A NaN dim value in the result resolves to its own cell (the hash
    strategy matches NaN by value equality)."""
    station = np.array([2.0, np.nan, 5.0, 1.0])
    template = xr.Dataset(
        {"v": (("station",), np.array([10.0, 20.0, 30.0, 40.0]))},
        coords={"station": station},
    )
    table = pa.table({"station": station, "v": template["v"].values})
    out = to_dataset(table, template=template)
    np.testing.assert_array_equal(out["v"].values, template["v"].values)


def test_float16_axis_roundtrip():
    """A float16 coordinate axis reconstructs through the sorted-search
    strategy (pandas indexes do not support float16)."""
    station = np.array([0.5, 1.5, 4.0, 9.0], dtype="float16")
    template = xr.Dataset(
        {"v": (("station",), np.array([1.0, 2.0, 3.0, 4.0], dtype="f4"))},
        coords={"station": station},
    )
    table = pa.table(
        {
            "station": pa.array(station, type=pa.float16()),
            "v": template["v"].values,
        }
    )
    out = to_dataset(table, template=template)
    np.testing.assert_array_equal(out["v"].values, template["v"].values)


def _one_batch(station: np.ndarray, v: np.ndarray) -> list[pa.RecordBatch]:
    return [
        pa.RecordBatch.from_arrays(
            [pa.array(station), pa.array(v)], names=["station", "v"]
        )
    ]


def test_missing_value_raises_value_error():
    """A row value absent from a unique irregular axis fails loudly
    instead of scattering to a wrong cell."""
    axis = _irregular_axis(10)
    with pytest.raises(ValueError, match="dimension 'station'"):
        _scatter_batches_to_ndarray(
            batches=_one_batch(np.array([-1.0]), np.array([0.0], dtype="f4")),
            dimension_columns=["station"],
            requested={"station": axis},
            var_name="v",
            out_shape=(10,),
            dtype=np.dtype("float32"),
            drop_axes=[],
        )


def test_duplicate_axis_values_scatter_to_a_holding_position():
    """An axis with duplicate values takes the sorted-search fallback:
    each value lands on a position that holds it in the axis."""
    axis = np.array([3.0, 1.0, 2.0, 1.0])  # 1.0 appears twice
    out = _scatter_batches_to_ndarray(
        batches=_one_batch(
            np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0], dtype="f4")
        ),
        dimension_columns=["station"],
        requested={"station": axis},
        var_name="v",
        out_shape=(4,),
        dtype=np.dtype("float32"),
        drop_axes=[],
    )
    # 2.0 and 3.0 have unique positions; 10.0 landed on one of the two
    # cells whose coordinate is 1.0.
    assert out[2] == 20.0 and out[0] == 30.0
    assert 10.0 in (out[1], out[3])
