# DataFusion dictionary-column repros

Two minimal, self-contained reproductions of DataFusion bugs that block
dictionary-encoding coordinate columns in xarray-sql (see
[#217](https://github.com/xqlsystems/xarray-sql/pull/217) and the tracking
issues). Both are pure `datafusion` + `pyarrow` — no xarray-sql, no network — so
they can be handed to upstream maintainers directly.

```shell
pip install "datafusion==54.0.0" pyarrow
python dict_key_overflow.py            # exits 1 while the bug is present
python date_part_dictionary_panic.py   # exits 1 while the bug is present
```

Observed with **datafusion 54.0.0 / pyarrow 23.0.0 (arrow-rs 58.3.0)**. Each
script exits non-zero while the bug reproduces and `0` if a DataFusion build has
fixed it, so they double as regression checks.

## 1. `dict_key_overflow.py` — key overflow when combining partitions

`GROUP BY` over a dictionary column fails with:

```
DataFusion error: Arrow error: Dictionary key bigger than the key type
```

Each partition is a valid `Dictionary(Int8, Int64)` (≤128 distinct values, so an
int8 key is legal per batch), but the partitions carry *disjoint* values. When
the aggregate combines them the combined dictionary exceeds 128 entries and the
int8 key overflows. A producer can't know, per batch, how large the *combined*
dictionary will grow, so combining should widen the key (or decode) rather than
error.

## 2. `date_part_dictionary_panic.py` — scalar-fn result type mismatch

`date_part('hour', t)` where `t` is `Dictionary(Int32, Timestamp)` yields a
`Dictionary(Int32, Int32)` whose declared logical type is plain `Int32`:

```
Arrow error: ... expected Int32 but found Dictionary(Int32, Int32) at column index 0
```

A `GROUP BY` / `JOIN` on the same expression instead **panics** a worker thread
(`arrow-array/src/cast.rs` `"primitive array"`, a failed downcast). `date_part`
on a dictionary timestamp should return a plain `Int32` (decode), or declare the
dictionary type it actually produces — logical and physical types must agree.

## Rust equivalents (for a DataFusion test)

The Python above drives the same Rust engine; here are paste-in sketches.

```rust
// 1. key overflow
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
            ("k", Arc::new(k) as _),
            ("v", Arc::new(v) as _),
        ])?]);
    }
    ctx.register_batches("t", partitions)?; // one Vec<RecordBatch> per partition
    // panics/errors: "Dictionary key bigger than the key type"
    ctx.sql("SELECT k, SUM(v) FROM t GROUP BY k").await?.collect().await?;
    Ok(())
}
```

```rust
// 2. date_part over a dictionary timestamp
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
    let batch = RecordBatch::try_from_iter([("t", Arc::new(t) as _)])?;
    ctx.register_batch("x", batch)?;
    // errors: expected Int32 but found Dictionary(Int32, Int32)
    ctx.sql("SELECT date_part('hour', t) AS h FROM x").await?.collect().await?;
    Ok(())
}
```
