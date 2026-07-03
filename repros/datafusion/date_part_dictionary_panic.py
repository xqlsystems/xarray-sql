#!/usr/bin/env python3
"""Minimal repro: ``date_part`` over a dictionary-encoded timestamp produces a
result whose physical type disagrees with its declared logical type.

    Arrow error: Invalid argument error: column types must match schema types,
    expected Int32 but found Dictionary(Int32, Int32) at column index 0

``date_part('hour', t)`` where ``t`` is ``Dictionary(Int32, Timestamp)``
preserves the dictionary encoding and yields a ``Dictionary(Int32, Int32)``
array, but the expression's declared logical type is plain ``Int32``. The
mismatch surfaces the moment the column is materialized. Downstream operators
make it worse: a ``GROUP BY`` / ``JOIN`` on the same expression panics a worker
thread instead (``arrow-array/src/cast.rs`` ``"primitive array"`` — a failed
downcast to a primitive array), which is the shape climate SQL usually hits
(``GROUP BY date_part('hour', time)`` for a diurnal climatology).

Expected: ``date_part`` on a dictionary timestamp should yield a plain ``Int32``
(decode), or its declared type should match the ``Dictionary(Int32, Int32)`` it
returns — either way, logical and physical types must agree.

Pure ``datafusion`` + ``pyarrow``; no third-party packages, no network.

    pip install "datafusion==54.0.0" pyarrow
    python date_part_dictionary_panic.py    # exits 1 while the bug is present

Observed with datafusion 54.0.0 / pyarrow 23.0.0 (arrow-rs 58.3.0).
"""

import sys

import numpy as np
import pyarrow as pa
from datafusion import SessionContext

N = 240


def main() -> int:
    stamps = (
        np.datetime64("2020-01-01") + np.arange(N) * np.timedelta64(1, "h")
    ).astype("datetime64[us]")
    distinct = np.unique(stamps)
    values = pa.array(distinct, type=pa.timestamp("us"))
    indices = pa.array(np.searchsorted(distinct, stamps).astype(np.int32))
    # Dictionary(Int32, Timestamp) — how xarray-sql dictionary-encodes a time axis.
    time = pa.DictionaryArray.from_arrays(indices, values)

    tbl = pa.table({"t": time, "v": pa.array(np.random.rand(N))})
    ctx = SessionContext()
    ctx.register_record_batches("x", [tbl.to_batches()])

    print("SELECT date_part('hour', t) over a Dictionary(Int32, Timestamp) column ...")
    try:
        ctx.sql("SELECT date_part('hour', t) AS h FROM x").to_arrow_table()
    except Exception as exc:  # noqa: BLE001
        print(f"\nREPRODUCED — date_part over the dictionary timestamp failed:\n  {exc}")
        print(
            "\n(GROUP BY / JOIN on the same expression instead panics a worker "
            'thread: arrow-array cast.rs "primitive array".)'
        )
        return 1

    print("\nOK — date_part over the dictionary timestamp succeeded; bug appears fixed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
