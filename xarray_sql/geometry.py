"""GeoArrow point-geometry columns derived from coordinate dimensions.

A regular grid's pivot already materializes per-row x/y coordinate
columns; a point-geometry column is those same values under a GeoArrow
extension annotation. Two encodings:

* ``"wkb"`` (default) — 21-byte WKB points under the ``geoarrow.wkb``
  extension name. DuckDB (>= 1.2, spatial loaded) ingests the column as
  a native ``GEOMETRY`` with the CRS attached, so ``ST_Within(geometry,
  ...)`` works with no ``ST_Point(x, y)`` construction in user SQL.
* ``"point"`` — GeoArrow native points with *separated* coordinates
  (``struct<x: double, y: double>`` under ``geoarrow.point``): the
  child arrays are the coordinate columns themselves, no per-row
  parsing for consumers that execute on native layouts (GeoPandas 1.x,
  geoarrow-rs, lonboard, SedonaDB). DuckDB does not consume this
  encoding.

The CRS rides in the extension metadata (GeoArrow 0.2 allows
authority:code strings alongside PROJJSON). ``OGC:CRS84`` is the
correct tag for plain longitude/latitude grids.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa

GEOMETRY_COLUMN = "geometry"

_ENCODINGS = ("wkb", "point")


def geometry_field(encoding: str, crs: str | None) -> pa.Field:
    """The schema field for the derived geometry column."""
    if encoding not in _ENCODINGS:
        raise ValueError(
            f"geometry_encoding must be one of {_ENCODINGS}, got {encoding!r}"
        )
    metadata = {
        b"ARROW:extension:name": f"geoarrow.{encoding}".encode(),
    }
    if crs is not None:
        metadata[b"ARROW:extension:metadata"] = json.dumps(
            {"crs": crs}
        ).encode()
    storage = (
        pa.binary()
        if encoding == "wkb"
        else pa.struct([("x", pa.float64()), ("y", pa.float64())])
    )
    return pa.field(GEOMETRY_COLUMN, storage, metadata=metadata)


def build_geometry(
    encoding: str, x: pa.Array, y: pa.Array
) -> pa.Array:
    """Point geometries for one batch's x/y coordinate columns."""
    if encoding == "point":
        return pa.StructArray.from_arrays(
            [x.cast(pa.float64()), y.cast(pa.float64())], ["x", "y"]
        )
    return _wkb_points(
        np.ascontiguousarray(x.to_numpy(zero_copy_only=False), "<f8"),
        np.ascontiguousarray(y.to_numpy(zero_copy_only=False), "<f8"),
    )


def _wkb_points(x: np.ndarray, y: np.ndarray) -> pa.Array:
    """Vectorized 21-byte little-endian WKB point encoding."""
    n = len(x)
    buf = np.empty((n, 21), dtype=np.uint8)
    buf[:, 0] = 1  # little-endian byte order mark
    buf[:, 1:5] = np.array([1, 0, 0, 0], dtype=np.uint8)  # type: Point
    buf[:, 5:13] = x.view(np.uint8).reshape(n, 8)
    buf[:, 13:21] = y.view(np.uint8).reshape(n, 8)
    offsets = pa.py_buffer(
        np.arange(0, (n + 1) * 21, 21, dtype=np.int32).tobytes()
    )
    return pa.Array.from_buffers(
        pa.binary(), n, [None, offsets, pa.py_buffer(buf.tobytes())]
    )


def bbox_conjuncts(
    bounds: Any, x: str = "x", y: str = "y", pad: float = 0.0
) -> str:
    """SQL bbox conjuncts for a geometry's envelope — the pruning half.

    Engines do not push ``ST_*`` functions into the scan, so a
    geometry-only predicate reads every chunk; pairing it with range
    conjuncts on the coordinate columns restores pruning. This helper
    renders those conjuncts from a geometry's envelope::

        poly = shapely.from_wkt("POLYGON (...)")
        con.execute(f"""
            SELECT avg(risk) FROM eri
            WHERE {xql.bbox_conjuncts(poly, x="x", y="y")}
              AND ST_Within(geometry, ST_GeomFromText('{poly.wkt}'))
        """)

    Args:
        bounds: ``(xmin, ymin, xmax, ymax)``, or any object with a
            ``.bounds`` attribute in that convention (shapely
            geometries qualify).
        x: The x/longitude column name.
        y: The y/latitude column name.
        pad: Optional margin added on every side (e.g. to be safe
            around ``ST_DWithin``-style predicates).

    Returns:
        A SQL snippet ``"x" BETWEEN a AND b AND "y" BETWEEN c AND d``.
    """
    values = getattr(bounds, "bounds", bounds)
    xmin, ymin, xmax, ymax = (float(v) for v in values)
    return (
        f'"{x}" BETWEEN {xmin - pad!r} AND {xmax + pad!r} '
        f'AND "{y}" BETWEEN {ymin - pad!r} AND {ymax + pad!r}'
    )
