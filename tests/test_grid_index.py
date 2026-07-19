"""Integer grid-position columns (``<dim>_idx``) for exact grid joins.

Regridding and forecast alignment join a source grid to a table on the source
*coordinate*. Joining on the floating-point coordinate value is fragile — any
sub-ULP drift (e.g. a reproject/interp computed in float32) makes the equality
join silently drop rows. The opt-in ``<dim>_idx`` columns give exact integer
grid keys instead. These tests pin that the indices are global (partition
independent) and that an index-keyed regrid stays exact where a float join does
not.
"""

import numpy as np
import pyarrow as pa
import xarray as xr

from xarray_sql import XarrayContext
from xarray_sql.df import _ensure_default_indexes, _parse_schema


# Deliberately not float32-exact, so a float32 round-trip actually perturbs the
# bits (a float32-exact axis like 10.0/20.0 would round-trip unchanged).
_X = np.array([10.1, 20.2, 30.3])
_Y = np.array([1.1, 2.2, 3.3, 4.4])


def _src() -> xr.Dataset:
    return xr.Dataset(
        {"v": (("x", "y"), np.arange(12.0).reshape(3, 4))},
        coords={"x": _X, "y": _Y},
    )


def test_index_columns_are_int32_in_schema():
    schema = _parse_schema(_ensure_default_indexes(_src()), index_columns=True)
    assert schema.field("x_idx").type == pa.int32()
    assert schema.field("y_idx").type == pa.int32()
    # Not present unless requested.
    plain = _parse_schema(_ensure_default_indexes(_src()))
    assert "x_idx" not in plain.names


def test_index_columns_are_global_across_chunks():
    """Indices are absolute axis positions, not per-partition local ones."""
    ctx = XarrayContext()
    # chunk x into 3 single-row partitions: a local index would restart at 0
    # in every partition; the global index must run 0,1,2.
    ctx.from_dataset("g", _src(), chunks={"x": 1}, index_columns=True)
    df = ctx.sql("SELECT x, y, x_idx, y_idx FROM g").to_pandas()

    assert df["x_idx"].dtype == np.int32
    xpos = {v: i for i, v in enumerate(_X)}
    ypos = {v: i for i, v in enumerate(_Y)}
    assert (df["x_idx"] == df["x"].map(xpos)).all()
    assert (df["y_idx"] == df["y"].map(ypos)).all()


def _weights(perturb_f32: bool = False) -> xr.Dataset:
    """A tiny gather 'regrid': dst cell k reads one src cell, weight 1.0."""
    # dst 0..4 map to source cells (x_idx, y_idx):
    sx = np.array([0, 2, 1, 0, 2], dtype=np.int32)
    sy = np.array([0, 3, 1, 3, 0], dtype=np.int32)
    src_x = _X[sx]
    src_y = _Y[sy]
    if perturb_f32:  # coords computed in single precision, as a reproject UDF
        src_x = src_x.astype(np.float32).astype(np.float64)
        src_y = src_y.astype(np.float32).astype(np.float64)
    return xr.Dataset(
        {
            "dst_id": (("pair",), np.arange(5, dtype=np.int32)),
            "src_x_idx": (("pair",), sx),
            "src_y_idx": (("pair",), sy),
            "src_x": (("pair",), src_x),
            "src_y": (("pair",), src_y),
            "weight": (("pair",), np.ones(5)),
        }
    )


INDEX_JOIN = """
    SELECT w.dst_id, SUM(s.v * w.weight) AS out
    FROM weights w JOIN src s
      ON s.x_idx = w.src_x_idx AND s.y_idx = w.src_y_idx
    GROUP BY w.dst_id ORDER BY w.dst_id
"""
FLOAT_JOIN = """
    SELECT w.dst_id, SUM(s.v * w.weight) AS out
    FROM weights w JOIN src s
      ON s.x = w.src_x AND s.y = w.src_y
    GROUP BY w.dst_id ORDER BY w.dst_id
"""


def _ctx(weights: xr.Dataset) -> XarrayContext:
    ctx = XarrayContext()
    ctx.from_dataset("src", _src(), chunks={"x": 1}, index_columns=True)
    ctx.from_dataset("weights", weights, chunks={"pair": 5})
    return ctx


def test_index_join_regrids_exactly():
    """Index-keyed regrid matches a direct numpy gather, across chunks."""
    ctx = _ctx(_weights())
    got = ctx.sql(INDEX_JOIN).to_pandas().sort_values("dst_id")

    v = np.arange(12.0).reshape(3, 4)
    sx = np.array([0, 2, 1, 0, 2])
    sy = np.array([0, 3, 1, 3, 0])
    expected = v[sx, sy]
    np.testing.assert_allclose(got["out"].to_numpy(), expected)


def test_index_join_survives_float32_drift_where_float_join_does_not():
    ctx = _ctx(_weights(perturb_f32=True))
    idx = ctx.sql(INDEX_JOIN).to_pandas()
    flt = ctx.sql(FLOAT_JOIN).to_pandas()
    # Index join keeps every destination cell; the float-equality join drops the
    # cells whose float32-roundtripped coord no longer bit-matches the source.
    assert len(idx) == 5
    assert len(flt) < 5
