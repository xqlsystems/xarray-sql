#!/usr/bin/env python3
"""Minimal repro: aggregating a dictionary column across partitions overflows
the dictionary key type.

    DataFusion error: Arrow error: Dictionary key bigger than the key type

Each partition is a valid ``Dictionary(Int8, Int64)`` batch — its own dictionary
has at most 128 distinct values, so an int8 key is legal *per batch*. But the
partitions carry *disjoint* values, so when the aggregate combines them the
combined dictionary exceeds 128 entries and the int8 key overflows. Combining
dictionary-typed columns across partitions should not fail on a key type that
was valid for each input batch (widen the key, or decode) — a producer cannot
know, per batch, how large the *combined* dictionary will become downstream.

Pure ``datafusion`` + ``pyarrow``; no third-party packages, no network.

    pip install "datafusion==54.0.0" pyarrow
    python dict_key_overflow.py     # exits 1 while the bug is present

Observed with datafusion 54.0.0 / pyarrow 23.0.0 (arrow-rs 58.3.0).
"""

import sys

import numpy as np
import pyarrow as pa
from datafusion import SessionContext

PER_PARTITION = 100  # <= 128, so int8 keys are valid within each batch
N_PARTITIONS = 50  # combined distinct values = 5000 >> int8 max (127)


def main() -> int:
    partitions = []
    for p in range(N_PARTITIONS):
        base = p * PER_PARTITION
        values = pa.array(
            np.arange(base, base + PER_PARTITION, dtype=np.int64)
        )  # disjoint across partitions -> cannot be unified away
        indices = pa.array(np.arange(PER_PARTITION, dtype=np.int8))
        key = pa.DictionaryArray.from_arrays(indices, values)
        partitions.append(
            pa.record_batch({"k": key, "v": pa.array(np.ones(PER_PARTITION))})
        )

    ctx = SessionContext()
    # one partition per batch
    ctx.register_record_batches("t", [[b] for b in partitions])

    print(
        f"{N_PARTITIONS} partitions x Dictionary(Int8, Int64) of "
        f"{PER_PARTITION} disjoint values -> combined cardinality "
        f"{N_PARTITIONS * PER_PARTITION}"
    )
    try:
        result = ctx.sql("SELECT k, SUM(v) FROM t GROUP BY k").to_arrow_table()
    except Exception as exc:  # noqa: BLE001
        print(f"\nREPRODUCED — GROUP BY over the dictionary column failed:\n  {exc}")
        return 1

    print(f"\nOK — aggregate returned {result.num_rows} groups; bug appears fixed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
