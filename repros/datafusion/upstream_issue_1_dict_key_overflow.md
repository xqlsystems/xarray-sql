<!-- Draft to file at apache/datafusion. Title below; body is everything under it. -->

# Title

Aggregating a dictionary column across partitions overflows the key type ("Dictionary key bigger than the key type")

# Body

> Filed with AI assistance, per the project's AI policy. The reproduction was written and verified by running it against `datafusion 54.0.0` (`pyarrow 23.0.0`, arrow-rs 58.3.0); a human reviewed it before filing.

## Describe the bug

`GROUP BY` over a dictionary-typed column fails when the *combined* dictionary across partitions exceeds the per-batch key width:

```
DataFusion error: Arrow error: Dictionary key bigger than the key type
```

Each input partition is a **valid** `Dictionary(Int8, Int64)` — its dictionary has ≤128 distinct values, so an `Int8` key is legal *per batch*. But the partitions carry **disjoint** values, so when the aggregate combines them the combined dictionary exceeds 128 entries and the `Int8` key overflows.

The disjointness matters: when every partition shares the *same* dictionary values, they are unified and there is no overflow. That is why this is intermittent in the wild.

## To Reproduce

Pure `datafusion` + `pyarrow`, no other deps, no network:

```python
import numpy as np, pyarrow as pa
from datafusion import SessionContext

PER_PARTITION, N_PARTITIONS = 100, 50  # combined cardinality 5000 >> Int8 max
partitions = []
for p in range(N_PARTITIONS):
    values = pa.array(np.arange(p * PER_PARTITION, p * PER_PARTITION + PER_PARTITION, dtype=np.int64))
    keys = pa.array(np.arange(PER_PARTITION, dtype=np.int8))
    k = pa.DictionaryArray.from_arrays(keys, values)  # valid Dictionary(Int8, Int64) per batch
    partitions.append(pa.record_batch({"k": k, "v": pa.array(np.ones(PER_PARTITION))}))

ctx = SessionContext()
ctx.register_record_batches("t", [[b] for b in partitions])  # one partition per batch
ctx.sql("SELECT k, SUM(v) FROM t GROUP BY k").to_arrow_table()
# -> Arrow error: Dictionary key bigger than the key type
```

Rust equivalent (paste into a `datafusion` test):

```rust
use std::sync::Arc;
use arrow::array::{DictionaryArray, Float64Array, Int8Array, Int64Array};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;

#[tokio::test]
async fn dict_key_overflow() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();
    let mut partitions = Vec::new();
    for p in 0..50i64 {
        let values = Int64Array::from_iter_values((p * 100)..(p * 100 + 100));
        let keys = Int8Array::from_iter_values(0..100i8);
        let k = DictionaryArray::new(keys, Arc::new(values));
        let v = Float64Array::from(vec![1.0; 100]);
        partitions.push(vec![RecordBatch::try_from_iter([
            ("k", Arc::new(k) as _), ("v", Arc::new(v) as _),
        ])?]);
    }
    ctx.register_batches("t", partitions)?;
    ctx.sql("SELECT k, SUM(v) FROM t GROUP BY k").await?.collect().await?;
    Ok(())
}
```

## Expected behavior

Combining dictionary-typed columns across partitions should not fail on a key type that was valid for each input batch. It should widen the key (`Int8` → `Int16` → …) or decode, since a producer cannot know per batch how large the *combined* dictionary will become downstream.

## Additional context

Version: datafusion 54.0.0 / pyarrow 23.0.0 (arrow-rs 58.3.0).

Downstream tracking issue with the same repro (xarray-sql): xqlsystems/xarray-sql#220. It blocks dictionary-encoding coordinate columns there; a chunked coordinate whose partitions hold disjoint slices hits this under streaming aggregation.
