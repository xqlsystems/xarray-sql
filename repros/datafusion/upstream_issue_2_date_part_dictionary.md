<!-- Draft to file at apache/datafusion. Title below; body is everything under it. -->

# Title

`date_part` over a `Dictionary(Timestamp)` returns a type that violates its declared `Int32` (crashes GROUP BY / JOIN)

# Body

> Filed with AI assistance, per the project's AI policy. The reproduction was written and verified by running it against `datafusion 54.0.0` (`pyarrow 23.0.0`, arrow-rs 58.3.0); a human reviewed it before filing.

## Describe the bug

`date_part('hour', t)` where `t` is `Dictionary(Int32, Timestamp)` preserves the dictionary encoding and returns a `Dictionary(Int32, Int32)`, but the expression's **declared logical type is plain `Int32`**. The physical/logical mismatch surfaces as soon as the column is materialized:

```
Arrow error: Invalid argument error: column types must match schema types,
expected Int32 but found Dictionary(Int32, Int32) at column index 0
```

A `GROUP BY` / `JOIN` on the same expression is worse — it **panics a worker thread**:

```
thread 'tokio-rt-worker' panicked at arrow-array-58.3.0/src/cast.rs:849: primitive array
```

(a failed downcast of the dictionary array to a primitive array). `GROUP BY date_part('hour', time)` is the canonical way to write a diurnal climatology, so this is not an edge case for time-series workloads.

## To Reproduce

Pure `datafusion` + `pyarrow`, no other deps, no network:

```python
import numpy as np, pyarrow as pa
from datafusion import SessionContext

N = 240
stamps = (np.datetime64("2020-01-01") + np.arange(N) * np.timedelta64(1, "h")).astype("datetime64[us]")
distinct = np.unique(stamps)
values = pa.array(distinct, type=pa.timestamp("us"))
indices = pa.array(np.searchsorted(distinct, stamps).astype(np.int32))
t = pa.DictionaryArray.from_arrays(indices, values)  # Dictionary(Int32, Timestamp)

ctx = SessionContext()
ctx.register_record_batches("x", [pa.table({"t": t}).to_batches()])
ctx.sql("SELECT date_part('hour', t) AS h FROM x").to_arrow_table()
# -> expected Int32 but found Dictionary(Int32, Int32) at column index 0
# and: SELECT date_part('hour', t), COUNT(*) FROM x GROUP BY 1  -> panics ("primitive array")
```

Rust equivalent (paste into a `datafusion` test):

```rust
use std::sync::Arc;
use arrow::array::{DictionaryArray, Int32Array, TimestampMicrosecondArray};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;

#[tokio::test]
async fn date_part_over_dictionary_timestamp() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();
    let values = TimestampMicrosecondArray::from(vec![0, 3_600_000_000, 7_200_000_000]);
    let keys = Int32Array::from(vec![0, 1, 2, 0, 1, 2]);
    let t = DictionaryArray::new(keys, Arc::new(values));
    ctx.register_batch("x", RecordBatch::try_from_iter([("t", Arc::new(t) as _)])?)?;
    ctx.sql("SELECT date_part('hour', t) AS h FROM x").await?.collect().await?;
    Ok(())
}
```

## Expected behavior

`date_part` (and temporal scalar functions generally) over a dictionary-encoded input should return a plain `Int32` (decode the dictionary), or declare the `Dictionary(Int32, Int32)` it actually produces — logical and physical types must agree, and a downstream `GROUP BY` / `JOIN` must never panic on the result.

## Additional context

Version: datafusion 54.0.0 / pyarrow 23.0.0 (arrow-rs 58.3.0).

Downstream tracking issue with the same repro (xarray-sql): xqlsystems/xarray-sql#221. It blocks dictionary-encoding the time coordinate there; because time is where `date_part`/`date_trunc`/`extract` are applied most, encoding it breaks climatology and anomaly queries.
