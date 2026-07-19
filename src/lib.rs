// Licensed under the Apache License, Version 2.0
//!
//! Lazy Arrow stream table provider for xarray-sql.
//!
//! This module provides `LazyArrowStreamTable`, which wraps a Python object
//! implementing `__arrow_c_stream__` and exposes it as a DataFusion TableProvider
//! via the `__datafusion_table_provider__` protocol.
//!
//! ## Key Features
//!
//! - **Lazy evaluation**: Data is not read from the Python stream until query
//!   execution time (during `collect()`), not at registration time.
//!
//! - **Error propagation**: Errors during batch reading are properly propagated
//!   to DataFusion and surfaced to the user.
//!
//! - **Reusable tables**: The factory pattern allows the same table to be queried
//!   multiple times, with fresh data streams created for each query.
//!
//! ## Streaming Behavior
//!
//! Batches are read lazily one at a time during query execution. The GIL is acquired
//! for each batch read, allowing DataFusion to process and potentially filter batches
//! incrementally. This enables processing of larger-than-memory datasets when combined
//! with DataFusion's streaming execution.
//!
//! ## Parallel Execution
//!
//! Each xarray chunk becomes a separate partition, enabling parallel execution across
//! multiple cores.
//!
//! ## Filter Pushdown (Partition Pruning)
//!
//! When partition metadata is provided, SQL filters on dimension columns (time, lat, lon)
//! automatically prune partitions that can't contain matching rows. For example:
//!
//! ```sql
//! SELECT * FROM air WHERE time > '2020-02-01'
//! ```
//!
//! Will skip loading partitions whose time ranges are entirely before 2020-02-01.
//! Supported operators: `=`, `<`, `>`, `<=`, `>=`, `BETWEEN`, `IN`, `AND`, `OR`.

mod autograd;

use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use async_stream::try_stream;
use async_trait::async_trait;
use datafusion::catalog::streaming::StreamingTable;
use datafusion::catalog::Session;
use datafusion::common::stats::Precision;
use datafusion::common::{
    ColumnStatistics, DFSchema, DataFusionError, Result as DFResult, ScalarValue, Statistics,
};
use datafusion::datasource::TableProvider;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{
    BinaryExpr, Expr, Operator, TableProviderFilterPushDown, TableType,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::PartitionStream;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use datafusion::prelude::SessionContext;
use datafusion::sql::unparser::expr_to_sql;
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};

// ============================================================================
// Partition Metadata Types for Filter Pushdown
// ============================================================================

// TODO(alxmrs, Claude): Support every valid xarray coordinate type.
/// Scalar value for dimension bounds, supporting common xarray coordinate types.
#[derive(Clone, Debug)]
pub enum ScalarBound {
    /// 64-bit integer (for integer coordinates)
    Int64(i64),
    /// 64-bit float (for lat/lon coordinates)
    Float64(f64),
    /// Nanoseconds since Unix epoch (for datetime64[ns] coordinates)
    TimestampNanos(i64),
}

impl ScalarBound {
    /// Compare this bound with a DataFusion ScalarValue.
    /// Returns None if types are incompatible.
    fn compare_to_scalar(&self, scalar: &ScalarValue) -> Option<std::cmp::Ordering> {
        match (self, scalar) {
            // Integer comparisons
            (ScalarBound::Int64(a), ScalarValue::Int64(Some(b))) => Some(a.cmp(b)),
            (ScalarBound::Int64(a), ScalarValue::Int32(Some(b))) => Some(a.cmp(&(*b as i64))),

            // Float comparisons
            (ScalarBound::Float64(a), ScalarValue::Float64(Some(b))) => a.partial_cmp(b),
            (ScalarBound::Float64(a), ScalarValue::Float32(Some(b))) => a.partial_cmp(&(*b as f64)),

            // Timestamp comparisons - convert to nanoseconds.
            // Use checked_mul to avoid silent overflow in release builds;
            // on overflow return None (conservative: include the partition).
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampNanosecond(Some(b), _)) => {
                Some(a.cmp(b))
            }
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampMicrosecond(Some(b), _)) => {
                b.checked_mul(1_000).map(|b_ns| a.cmp(&b_ns))
            }
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampMillisecond(Some(b), _)) => {
                b.checked_mul(1_000_000).map(|b_ns| a.cmp(&b_ns))
            }
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampSecond(Some(b), _)) => {
                b.checked_mul(1_000_000_000).map(|b_ns| a.cmp(&b_ns))
            }

            // Incompatible types
            _ => None,
        }
    }
}

/// Range bounds for one dimension in a partition.
#[derive(Clone, Debug)]
pub struct DimensionRange {
    /// The column name (dimension name from xarray)
    pub column_name: String,
    /// Minimum value (inclusive) - first coordinate value in this partition
    pub min: ScalarBound,
    /// Maximum value (inclusive) - last coordinate value in this partition
    pub max: ScalarBound,
}

/// Metadata for a single partition, used for filter-based pruning.
#[derive(Clone, Debug)]
pub struct PartitionMetadata {
    /// Dimension ranges for this partition, keyed by column name
    pub ranges: HashMap<String, DimensionRange>,
    /// Exact number of rows in this partition (product of the chunk's
    /// per-dimension sizes). Every partition supplies it, so the scan can
    /// report exact `Statistics::num_rows` to the optimizer and cost-based
    /// rules (join build-side selection, broadcast vs. shuffle) have real
    /// cardinalities instead of guesses. xarray knows this exactly — it is
    /// the product of the partition's dimension lengths — so unlike most
    /// table providers these statistics are not estimates.
    pub num_rows: usize,
}

impl PartitionMetadata {
    /// Get the range for a specific dimension column.
    pub fn get_range(&self, column: &str) -> Option<&DimensionRange> {
        self.ranges.get(column)
    }
}

// ============================================================================
// Custom TableProvider with Filter Pushdown
// ============================================================================

/// A table provider that supports partition pruning via filter pushdown.
///
/// This wraps partition streams with their metadata and implements
/// `TableProvider::supports_filters_pushdown` and partition pruning in `scan()`.
struct PrunableStreamingTable {
    schema: SchemaRef,
    /// Partition streams paired with their coordinate range metadata.
    /// Stored behind the `ProjectableStream` trait so `PrunableStreamingTable`
    /// is not coupled to `PyArrowStreamPartition`.
    partitions: Vec<(Arc<dyn ProjectableStream>, PartitionMetadata)>,
    /// Set of column names that are dimension columns (eligible for pruning)
    dimension_columns: HashSet<String>,
}

impl PrunableStreamingTable {
    fn new(
        schema: SchemaRef,
        partitions: Vec<(Arc<dyn ProjectableStream>, PartitionMetadata)>,
    ) -> Self {
        // Collect dimension column names from the first partition that has
        // non-empty metadata. All partitions share the same dimension names,
        // so we only need one representative. Using find_map keeps this O(D)
        // rather than O(N × D) — important when N is in the hundreds of
        // thousands (e.g. hourly chunks of a decades-long climate dataset).
        let dimension_columns: HashSet<String> = partitions
            .iter()
            .find_map(|(_, meta)| {
                if meta.ranges.is_empty() {
                    None
                } else {
                    Some(meta.ranges.keys().cloned().collect())
                }
            })
            .unwrap_or_default();

        Self {
            schema,
            partitions,
            dimension_columns,
        }
    }

    /// Determine which partitions should be included based on filters.
    /// Returns indices of partitions that may contain matching rows.
    fn prune_partitions(&self, filters: &[Expr]) -> Vec<usize> {
        self.partitions
            .iter()
            .enumerate()
            .filter(|(_, (_, meta))| {
                // Include partition unless a filter definitely excludes it
                !filters
                    .iter()
                    .any(|f| self.filter_excludes_partition(f, meta))
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Returns true if this filter definitely excludes the partition.
    /// Conservative: returns false (include) if uncertain.
    fn filter_excludes_partition(&self, expr: &Expr, meta: &PartitionMetadata) -> bool {
        match expr {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                // Handle AND/OR logic
                match op {
                    Operator::And => {
                        // For AND, exclude if either side excludes
                        self.filter_excludes_partition(left, meta)
                            || self.filter_excludes_partition(right, meta)
                    }
                    Operator::Or => {
                        // For OR, exclude only if both sides exclude
                        self.filter_excludes_partition(left, meta)
                            && self.filter_excludes_partition(right, meta)
                    }
                    // Handle comparison operators
                    _ => self.comparison_excludes(left, op, right, meta),
                }
            }
            Expr::Not(_) => {
                // NOT inverts the predicate. We cannot safely derive exclusion
                // from the inner result: if inner returns false (uncertain),
                // !false = true would incorrectly exclude the partition.
                // Example: partition [1,10], NOT (col > 5) ≡ col <= 5 —
                // inner returns false (max=10 > 5), but the partition contains
                // values 1–5 which satisfy col <= 5 and must be included.
                // Be conservative: never exclude on NOT.
                false
            }
            Expr::Between(between) => self.between_excludes(between, meta),
            Expr::InList(in_list) => self.in_list_excludes(in_list, meta),
            // Unknown expression type - be conservative
            _ => false,
        }
    }

    /// Check if a comparison expression excludes this partition.
    fn comparison_excludes(
        &self,
        left: &Expr,
        op: &Operator,
        right: &Expr,
        meta: &PartitionMetadata,
    ) -> bool {
        // Try to extract column and literal from either side
        let (col_name, scalar, flipped) = match (left, right) {
            (Expr::Column(c), Expr::Literal(s, _)) => (c.name.clone(), s, false),
            (Expr::Literal(s, _), Expr::Column(c)) => (c.name.clone(), s, true),
            _ => return false, // Not a simple column-literal comparison
        };

        // Get the dimension range for this column
        let range = match meta.get_range(&col_name) {
            Some(r) => r,
            None => return false, // Not a dimension column, can't prune
        };

        // Flip operator if literal was on left side
        let effective_op = if flipped { flip_operator(op) } else { *op };

        // Determine if partition can be excluded based on operator
        match effective_op {
            // col > literal: exclude if max <= literal
            Operator::Gt => matches!(
                range.max.compare_to_scalar(scalar),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ),
            // col >= literal: exclude if max < literal
            Operator::GtEq => {
                matches!(
                    range.max.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Less)
                )
            }
            // col < literal: exclude if min >= literal
            Operator::Lt => matches!(
                range.min.compare_to_scalar(scalar),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            ),
            // col <= literal: exclude if min > literal
            Operator::LtEq => {
                matches!(
                    range.min.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Greater)
                )
            }
            // col = literal: exclude if literal outside [min, max]
            Operator::Eq => {
                let below_min = matches!(
                    range.min.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Greater)
                );
                let above_max = matches!(
                    range.max.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Less)
                );
                below_min || above_max
            }
            // col != literal: can exclude only if range is a single point equal
            // to the literal — every row has that value, so none satisfy !=.
            Operator::NotEq => {
                let min_eq = matches!(
                    range.min.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Equal)
                );
                let max_eq = matches!(
                    range.max.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Equal)
                );
                min_eq && max_eq
            }
            // Other operators: be conservative
            _ => false,
        }
    }

    /// Check if a BETWEEN expression excludes this partition.
    fn between_excludes(
        &self,
        between: &datafusion::logical_expr::Between,
        meta: &PartitionMetadata,
    ) -> bool {
        if between.negated {
            // NOT BETWEEN is complex, be conservative
            return false;
        }

        // Extract column name
        let col_name = match between.expr.as_ref() {
            Expr::Column(c) => c.name.clone(),
            _ => return false,
        };

        // Get dimension range
        let range = match meta.get_range(&col_name) {
            Some(r) => r,
            None => return false,
        };

        // Extract low and high bounds
        let (low, high) = match (between.low.as_ref(), between.high.as_ref()) {
            (Expr::Literal(l, _), Expr::Literal(h, _)) => (l, h),
            _ => return false,
        };

        // Exclude if partition range doesn't overlap with [low, high]
        // No overlap if: partition.max < low OR partition.min > high
        let max_below_low = matches!(
            range.max.compare_to_scalar(low),
            Some(std::cmp::Ordering::Less)
        );
        let min_above_high = matches!(
            range.min.compare_to_scalar(high),
            Some(std::cmp::Ordering::Greater)
        );

        max_below_low || min_above_high
    }

    /// Check if an IN list expression excludes this partition.
    fn in_list_excludes(&self, in_list: &InList, meta: &PartitionMetadata) -> bool {
        if in_list.negated {
            // NOT IN is complex, be conservative
            return false;
        }

        // Extract column name
        let col_name = match in_list.expr.as_ref() {
            Expr::Column(c) => c.name.clone(),
            _ => return false,
        };

        // Get dimension range
        let range = match meta.get_range(&col_name) {
            Some(r) => r,
            None => return false,
        };

        // Check if any value in the list could be in this partition's range
        let any_in_range = in_list.list.iter().any(|expr| {
            if let Expr::Literal(scalar, _) = expr {
                // Value is in range if: min <= value <= max
                let above_min = matches!(
                    range.min.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                );
                let below_max = matches!(
                    range.max.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                );
                above_min && below_max
            } else {
                // Non-literal in list, be conservative
                true
            }
        });

        // Exclude only if NO values could be in range
        !any_in_range
    }

    /// Check if an expression is a filter on a dimension column.
    fn is_dimension_filter(&self, expr: &Expr) -> bool {
        match expr {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
                Operator::And | Operator::Or => {
                    self.is_dimension_filter(left) || self.is_dimension_filter(right)
                }
                _ => self.expr_references_dimension(left) || self.expr_references_dimension(right),
            },
            Expr::Between(b) => self.expr_references_dimension(&b.expr),
            Expr::InList(i) => self.expr_references_dimension(&i.expr),
            Expr::Not(inner) => self.is_dimension_filter(inner),
            _ => false,
        }
    }

    /// Check if an expression references a dimension column.
    fn expr_references_dimension(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Column(c) => self.dimension_columns.contains(&c.name),
            _ => false,
        }
    }
}

/// Extension trait for partition streams that support column projection.
///
/// Implemented by `PyArrowStreamPartition` so that `PrunableStreamingTable`
/// can push projections to Python factories without coupling to the concrete type.
/// Any new stream implementation (e.g. for non-Python backends) can implement this
/// trait and be used with `PrunableStreamingTable` directly.
trait ProjectableStream: PartitionStream + Debug {
    /// Return a new stream that emits only the specified columns.
    fn clone_with_projection(
        &self,
        projection: Arc<[String]>,
        projected_schema: SchemaRef,
    ) -> Arc<dyn PartitionStream>;

    /// Clone this stream as a generic `PartitionStream` Arc.
    fn clone_as_stream(&self) -> Arc<dyn PartitionStream>;
}

/// Flip a comparison operator (for when literal is on left side).
fn flip_operator(op: &Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => *other,
    }
}

/// Convert a Python object to a ScalarBound using an explicit dtype tag.
fn python_to_scalar_bound(obj: &Bound<'_, PyAny>, dtype_tag: &str) -> PyResult<ScalarBound> {
    match dtype_tag {
        "timestamp_ns" => {
            let val = obj.extract::<i64>()?;
            Ok(ScalarBound::TimestampNanos(val))
        }
        "float64" => {
            let val = obj.extract::<f64>()?;
            Ok(ScalarBound::Float64(val))
        }
        "int64" => {
            let val = obj.extract::<i64>()?;
            Ok(ScalarBound::Int64(val))
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported dtype tag for partition bound: {dtype_tag}"
        ))),
    }
}

/// Convert a bound Python metadata dict into per-dimension coordinate ranges.
///
/// Operates on an already-bound reference so no additional GIL acquisition
/// is needed — this is called from within a `#[pymethods]` context where
/// the GIL is already held. The caller pairs the result with the partition's
/// row count to build a [`PartitionMetadata`].
fn convert_python_ranges_from_bound(
    meta_obj: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, DimensionRange>> {
    type MetaDict = HashMap<String, (Py<PyAny>, Py<PyAny>, String)>;
    let meta_dict: MetaDict = meta_obj.extract()?;
    let py = meta_obj.py();
    let mut ranges = HashMap::new();
    for (dim_name, (min_obj, max_obj, dtype_tag)) in meta_dict {
        let min_bound = python_to_scalar_bound(min_obj.bind(py), &dtype_tag)?;
        let max_bound = python_to_scalar_bound(max_obj.bind(py), &dtype_tag)?;
        ranges.insert(
            dim_name.clone(),
            DimensionRange {
                column_name: dim_name,
                min: min_bound,
                max: max_bound,
            },
        );
    }
    Ok(ranges)
}

impl Debug for PrunableStreamingTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrunableStreamingTable")
            .field("schema", &self.schema)
            .field("num_partitions", &self.partitions.len())
            .field("dimension_columns", &self.dimension_columns)
            .finish()
    }
}

#[async_trait]
impl TableProvider for PrunableStreamingTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // For dimension filters we can do exact pruning at partition level
        // Return Inexact so DataFusion still applies row-level filtering
        Ok(filters
            .iter()
            .map(|expr| {
                if self.is_dimension_filter(expr) {
                    // We can prune partitions but not individual rows within
                    TableProviderFilterPushDown::Inexact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Prune partitions based on filters
        let included_indices = self.prune_partitions(filters);

        // Exact per-partition row counts for the partitions that survive
        // pruning, in scan order. These feed `XarrayScanExec`'s statistics so
        // the optimizer sees real cardinalities.
        let included_metas: Vec<&PartitionMetadata> = included_indices
            .iter()
            .map(|&idx| &self.partitions[idx].1)
            .collect();
        let partition_rows: Vec<Precision<usize>> = included_metas
            .iter()
            .map(|meta| Precision::Exact(meta.num_rows))
            .collect();

        // Handle empty case — all partitions pruned, return empty plan
        if included_indices.is_empty() {
            let empty_table = StreamingTable::try_new(Arc::clone(&self.schema), vec![])?;
            let inner = empty_table.scan(state, projection, filters, limit).await?;
            let stats = build_scan_statistics(inner.schema().as_ref(), &included_metas);
            return Ok(Arc::new(XarrayScanExec::new(inner, stats, partition_rows)));
        }

        // Determine whether to push projection down to the Python factory.
        //
        // We push when the projection includes at least one data variable
        // (non-dimension column), because xarray can selectively load only
        // the requested data arrays while dimension coordinates are always
        // available via xarray's coordinate system.
        //
        // We do NOT push when:
        //   - projection is None (load everything — factory receives None)
        //   - projection is Some([]) (COUNT(*) — let StreamingTable handle)
        //   - projection contains only dimension columns (ds.to_dataframe()
        //     needs at least one data variable; dimensions are always loaded)
        let push_projection = match projection {
            Some(indices) if !indices.is_empty() => indices
                .iter()
                .any(|&i| !self.dimension_columns.contains(self.schema.field(i).name())),
            _ => false,
        };

        if push_projection {
            let indices = projection.unwrap();

            // Build the projected schema (only the requested fields)
            let proj_fields: Vec<_> = indices
                .iter()
                .map(|&i| self.schema.field(i).clone())
                .collect();
            let projected_schema = Arc::new(Schema::new(proj_fields));

            // Collect the requested column names to send to the factory.
            // Stored in an Arc so each clone_with_projection call shares the
            // same allocation via an atomic refcount increment (no N Vec copies).
            let proj_col_names: Arc<[String]> = indices
                .iter()
                .map(|&i| self.schema.field(i).name().to_string())
                .collect::<Vec<_>>()
                .into();

            // Clone each pruned partition with the projection baked in.
            // The factory will receive proj_col_names and load only those vars.
            let projected_partitions: Vec<Arc<dyn PartitionStream>> = included_indices
                .iter()
                .map(|&idx| {
                    self.partitions[idx].0.clone_with_projection(
                        Arc::clone(&proj_col_names),
                        Arc::clone(&projected_schema),
                    )
                })
                .collect();

            // StreamingTable already has the projected schema — pass None for
            // projection so it doesn't wrap the stream in a redundant ProjectionExec.
            let streaming = StreamingTable::try_new(projected_schema, projected_partitions)?;
            let inner = streaming.scan(state, None, filters, limit).await?;
            let stats = build_scan_statistics(inner.schema().as_ref(), &included_metas);
            Ok(Arc::new(XarrayScanExec::new(inner, stats, partition_rows)))
        } else {
            // No projection pushdown — factory is called with None (loads all
            // columns). StreamingTable applies projection via ProjectionExec.
            let included_partitions: Vec<Arc<dyn PartitionStream>> = included_indices
                .iter()
                .map(|&idx| self.partitions[idx].0.clone_as_stream())
                .collect();
            let streaming = StreamingTable::try_new(Arc::clone(&self.schema), included_partitions)?;
            let inner = streaming.scan(state, projection, filters, limit).await?;
            let stats = build_scan_statistics(inner.schema().as_ref(), &included_metas);
            Ok(Arc::new(XarrayScanExec::new(inner, stats, partition_rows)))
        }
    }
}

// ============================================================================
// Exact Statistics + Scan Wrapper
// ============================================================================

/// Sum the exact per-partition row counts into a `Precision<usize>`.
///
/// Every partition carries an exact count, so the total is exact too. The sum
/// is the table's row count and fits `usize`; `saturating_add` is defensive
/// against a pathological overflow rather than an expected case.
fn sum_row_counts<'a>(metas: impl Iterator<Item = &'a PartitionMetadata>) -> Precision<usize> {
    Precision::Exact(metas.fold(0usize, |total, meta| total.saturating_add(meta.num_rows)))
}

/// Fold two same-variant `ScalarBound`s, keeping the smaller (`keep_min`) or
/// larger one. Returns `None` if the variants differ (never expected within a
/// single dimension) so the caller can fall back to unknown.
fn fold_bound(a: &ScalarBound, b: &ScalarBound, keep_min: bool) -> Option<ScalarBound> {
    let ord = match (a, b) {
        (ScalarBound::Int64(x), ScalarBound::Int64(y)) => x.partial_cmp(y),
        (ScalarBound::Float64(x), ScalarBound::Float64(y)) => x.partial_cmp(y),
        (ScalarBound::TimestampNanos(x), ScalarBound::TimestampNanos(y)) => x.partial_cmp(y),
        _ => return None,
    }?;
    let take_a = if keep_min {
        ord != std::cmp::Ordering::Greater
    } else {
        ord != std::cmp::Ordering::Less
    };
    Some(if take_a { a.clone() } else { b.clone() })
}

/// Convert a coordinate bound into a `ScalarValue` matching a column's Arrow
/// type, so the min/max we report line up with the column's own type. Returns
/// `None` for combinations we cannot convert without loss (e.g. a timestamp
/// unit we can't scale exactly), in which case the column is left without
/// min/max rather than risk a wrong value.
fn bound_to_scalar(bound: &ScalarBound, dtype: &DataType) -> Option<ScalarValue> {
    match (bound, dtype) {
        (ScalarBound::Int64(v), DataType::Int64) => Some(ScalarValue::Int64(Some(*v))),
        (ScalarBound::Int64(v), DataType::Int32) => {
            i32::try_from(*v).ok().map(|x| ScalarValue::Int32(Some(x)))
        }
        (ScalarBound::Float64(v), DataType::Float64) => Some(ScalarValue::Float64(Some(*v))),
        (ScalarBound::Float64(v), DataType::Float32) => Some(ScalarValue::Float32(Some(*v as f32))),
        // Datetime coordinates arrive as nanoseconds (see `cftime.partition_bounds`
        // and the datetime64[ns] path in `_block_metadata`). Map them onto the
        // column's own timestamp unit, but only when the scaling is exact so a
        // reported bound is never a rounded value.
        (ScalarBound::TimestampNanos(v), DataType::Timestamp(unit, tz)) => {
            let scaled = match unit {
                TimeUnit::Nanosecond => Some(*v),
                TimeUnit::Microsecond if v % 1_000 == 0 => Some(v / 1_000),
                TimeUnit::Millisecond if v % 1_000_000 == 0 => Some(v / 1_000_000),
                TimeUnit::Second if v % 1_000_000_000 == 0 => Some(v / 1_000_000_000),
                _ => None,
            }?;
            Some(match unit {
                TimeUnit::Nanosecond => ScalarValue::TimestampNanosecond(Some(scaled), tz.clone()),
                TimeUnit::Microsecond => {
                    ScalarValue::TimestampMicrosecond(Some(scaled), tz.clone())
                }
                TimeUnit::Millisecond => {
                    ScalarValue::TimestampMillisecond(Some(scaled), tz.clone())
                }
                TimeUnit::Second => ScalarValue::TimestampSecond(Some(scaled), tz.clone()),
            })
        }
        _ => None,
    }
}

/// Exact in-memory byte size of `num_rows` rows of `schema`, or `Absent` if any
/// column is variable-width (e.g. Utf8) and cannot be sized from the row count
/// alone. Our data model is dense fixed-width grids, so this is normally exact.
fn total_byte_size(schema: &Schema, num_rows: &Precision<usize>) -> Precision<usize> {
    let Precision::Exact(rows) = num_rows else {
        return Precision::Absent;
    };
    let mut row_width = 0usize;
    for field in schema.fields() {
        match field.data_type().primitive_width() {
            Some(w) => row_width += w,
            None => return Precision::Absent,
        }
    }
    Precision::Exact(rows.saturating_mul(row_width))
}

/// Build `Statistics` for a scan over the given partitions.
///
/// Every statistic here is derived from coordinate metadata xarray already
/// knows — none of it scans the data — and each is exact, not an estimate:
///
/// * `num_rows`: summed product of each surviving chunk's dimension sizes.
///   Drives `JoinSelection`'s build-side choice and lets `COUNT(*)` skip the
///   scan entirely.
/// * `total_byte_size`: `num_rows × fixed row width`, for memory-cost rules.
/// * per dimension-coordinate column: exact `min`/`max` (folded coordinate
///   bounds — the join/filter keys) and `null_count = 0` (grid axes are always
///   fully populated). Data variables are left unknown; their bounds would need
///   a scan.
fn build_scan_statistics(output_schema: &Schema, metas: &[&PartitionMetadata]) -> Statistics {
    let mut stats = Statistics::new_unknown(output_schema);
    stats.num_rows = sum_row_counts(metas.iter().copied());
    stats.total_byte_size = total_byte_size(output_schema, &stats.num_rows);

    for (col_idx, field) in output_schema.fields().iter().enumerate() {
        // Fold this column's min/max across every partition that carries a
        // range for it. A column has a range iff it is a dimension coordinate
        // with a representable bound; all such partitions share the same bound
        // variant, so the fold is well-defined.
        let mut folded: Option<(ScalarBound, ScalarBound)> = None;
        for meta in metas {
            if let Some(range) = meta.ranges.get(field.name()) {
                folded = Some(match folded {
                    None => (range.min.clone(), range.max.clone()),
                    Some((lo, hi)) => (
                        fold_bound(&lo, &range.min, true).unwrap_or(lo),
                        fold_bound(&hi, &range.max, false).unwrap_or(hi),
                    ),
                });
            }
        }

        let Some((lo, hi)) = folded else { continue };
        // This column is a coordinate axis: never null, so the null count is
        // exactly zero regardless of whether the bound maps to a ScalarValue.
        let dtype = field.data_type();
        stats.column_statistics[col_idx] = ColumnStatistics {
            null_count: Precision::Exact(0),
            min_value: bound_to_scalar(&lo, dtype)
                .map(Precision::Exact)
                .unwrap_or(Precision::Absent),
            max_value: bound_to_scalar(&hi, dtype)
                .map(Precision::Exact)
                .unwrap_or(Precision::Absent),
            sum_value: Precision::Absent,
            distinct_count: Precision::Absent,
            byte_size: Precision::Absent,
        };
    }

    stats
}

/// A thin scan operator that wraps an inner `StreamingTableExec` and reports
/// exact `Statistics` to the query optimizer.
///
/// Execution, schema, ordering, and partitioning are delegated verbatim to the
/// inner plan (so projection mechanics are reused unchanged); the only thing
/// this node adds is real cardinality. `StreamingTableExec` reports unknown
/// statistics, and the physical `JoinSelection` rule reads statistics from the
/// `ExecutionPlan` (not from `TableProvider::statistics`) — even in DataFusion
/// 54, which forwards `ExecutionPlan` statistics across the FFI boundary — so
/// this wrapper is what carries the exact cardinality through to the optimizer.
#[derive(Debug)]
struct XarrayScanExec {
    inner: Arc<dyn ExecutionPlan>,
    statistics: Statistics,
    /// Exact row count per output partition (parallel to `inner` partitions),
    /// so `partition_statistics(Some(i))` is exact too.
    partition_rows: Vec<Precision<usize>>,
}

impl XarrayScanExec {
    fn new(
        inner: Arc<dyn ExecutionPlan>,
        statistics: Statistics,
        partition_rows: Vec<Precision<usize>>,
    ) -> Self {
        Self {
            inner,
            statistics,
            partition_rows,
        }
    }
}

impl DisplayAs for XarrayScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "XarrayScanExec: rows={:?}", self.statistics.num_rows)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "rows={:?}", self.statistics.num_rows)
            }
        }
    }
}

#[async_trait]
impl ExecutionPlan for XarrayScanExec {
    fn name(&self) -> &str {
        "XarrayScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        // Delegate partitioning + output ordering + boundedness to the inner
        // StreamingTableExec.
        self.inner.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // A scan is a leaf; the inner plan is an execution detail, not a child
        // the optimizer should rewrite.
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        self.inner.execute(partition, ctx)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DFResult<Arc<Statistics>> {
        match partition {
            None => Ok(Arc::new(self.statistics.clone())),
            Some(i) => {
                // Build a fresh, self-consistent per-partition summary rather
                // than reusing the table-level one: the folded column min/max
                // and `total_byte_size` describe the whole scan, and claiming
                // them (as `Exact`) for a single partition could be wrong —
                // partition `i` need not contain the table-wide min/max. We
                // keep only what is exact per partition: its row count and the
                // byte size derived from it. Column bounds are left unknown
                // (we do not retain per-partition bounds here).
                let num_rows = self
                    .partition_rows
                    .get(i)
                    .cloned()
                    .unwrap_or(Precision::Absent);
                let schema = self.inner.schema();
                let mut s = Statistics::new_unknown(&schema);
                s.num_rows = num_rows;
                s.total_byte_size = total_byte_size(&schema, &s.num_rows);
                Ok(Arc::new(s))
            }
        }
    }
}

/// A partition stream that wraps a Python factory function that creates streams.
///
/// The factory is called lazily on each `execute()` invocation, allowing
/// the same table to be queried multiple times.
///
/// When `projection` is set, the factory is called with that list of column
/// names so that xarray only loads the requested data variables rather than
/// materializing every variable in the dataset.
struct PyArrowStreamPartition {
    schema: SchemaRef,
    /// A Python callable (factory) that returns a fresh stream.
    /// Signature: `make_stream(projection_names: Optional[List[str]]) -> RecordBatchReader`
    ///
    /// Wrapped in `Arc` so `ProjectableStream::clone_with_projection` can share
    /// the same Python object across projected partitions without acquiring the
    /// GIL — only an atomic reference-count increment is needed.
    stream_factory: Arc<Py<PyAny>>,
    /// Column names to pass to the factory. `None` means load all columns.
    /// Stored as `Arc<[String]>` so multiple projected clones share one allocation.
    projection: Option<Arc<[String]>>,
}

impl PyArrowStreamPartition {
    fn new(stream_factory: Py<PyAny>, schema: SchemaRef) -> Self {
        Self {
            schema,
            stream_factory: Arc::new(stream_factory),
            projection: None,
        }
    }
}

impl ProjectableStream for PyArrowStreamPartition {
    /// Return a new partition that emits only the given columns.
    ///
    /// Clones the factory `Arc` (atomic refcount increment, no GIL) so the
    /// same Python callable is shared across all projected partitions.
    fn clone_with_projection(
        &self,
        projection: Arc<[String]>,
        projected_schema: SchemaRef,
    ) -> Arc<dyn PartitionStream> {
        Arc::new(Self {
            schema: projected_schema,
            stream_factory: Arc::clone(&self.stream_factory),
            projection: Some(projection),
        })
    }

    fn clone_as_stream(&self) -> Arc<dyn PartitionStream> {
        Arc::new(Self {
            schema: Arc::clone(&self.schema),
            stream_factory: Arc::clone(&self.stream_factory),
            projection: self.projection.clone(),
        })
    }
}

impl Debug for PyArrowStreamPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyArrowStreamPartition")
            .field("schema", &self.schema)
            .finish()
    }
}

impl PartitionStream for PyArrowStreamPartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let schema = Arc::clone(&self.schema);

        // Clone the factory Arc (no GIL needed) and the projection list.
        let factory = Arc::clone(&self.stream_factory);
        let projection = self.projection.clone();

        // Create a lazy stream using try_stream! macro.
        // The GIL is acquired only for the duration of each Python call
        // and released between batches.  After each yielded batch we call
        // yield_now() to explicitly suspend this task, giving the Tokio
        // executor a chance to poll other partition streams (which can then
        // acquire the GIL and make progress in parallel).
        let batch_stream = try_stream! {
            // Call factory with the projection argument.
            // `projection` is either a Python list of column names or None
            // (load all columns).  The factory always receives exactly one arg
            // so it can distinguish "no projection" from "empty projection".
            let reader: Py<PyAny> = Python::attach(|py| {
                let proj_arg = match &projection {
                    Some(cols) => PyList::new(py, cols.iter().map(|s| s.as_str()))
                        .map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Failed to build projection list: {e}"
                            ))
                        })?
                        .into_any(),
                    None => py.None().into_bound(py).into_any(),
                };
                factory.call1(py, (proj_arg,)).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to call stream factory: {e}"))
                })
            })?;

            // Read batches until StopIteration.
            // The GIL is released between iterations; yield_now() ensures
            // other async tasks (i.e., other partitions) are scheduled
            // before this stream is polled again.
            loop {
                let batch_result = Python::attach(|py| {
                    let bound_reader = reader.bind(py);
                    match bound_reader.call_method0("read_next_batch") {
                        Ok(batch_obj) => {
                            match RecordBatch::from_pyarrow_bound(&batch_obj) {
                                Ok(batch) => Ok(Some(batch)),
                                Err(e) => Err(DataFusionError::Execution(format!(
                                    "Failed to convert batch from PyArrow: {e}"
                                ))),
                            }
                        }
                        Err(e) => {
                            if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                                Ok(None) // Stream exhausted normally
                            } else {
                                Err(DataFusionError::Execution(format!(
                                    "Error reading batch from stream: {e}"
                                )))
                            }
                        }
                    }
                });

                match batch_result {
                    Ok(Some(batch)) => {
                        yield batch;
                        // Yield to the executor so that other partition
                        // streams can acquire the GIL and make progress.
                        tokio::task::yield_now().await;
                    }
                    Ok(None) => break,
                    Err(e) => Err(e)?,
                }
            }
        };

        Box::pin(RecordBatchStreamAdapter::new(schema, batch_stream))
    }
}

// ============================================================================
// FFI Helpers
// ============================================================================

/// Extract an `FFI_LogicalExtensionCodec` from a Python session object.
///
/// DataFusion 52 passes the `SessionContext` to `__datafusion_table_provider__`
/// so that the provider can obtain the codec needed for physical-plan
/// serialisation across the FFI boundary.  The session exposes this via
/// `__datafusion_logical_extension_codec__()`, which returns a PyCapsule
/// named `"datafusion_logical_extension_codec"`.
///
/// Mirrors the helper in the official datafusion-python FFI example
/// (`examples/datafusion-ffi-example/src/utils.rs`).
fn ffi_logical_codec_from_pycapsule(
    session: Bound<'_, PyAny>,
) -> PyResult<FFI_LogicalExtensionCodec> {
    let attr = "__datafusion_logical_extension_codec__";
    let capsule = if session.hasattr(attr)? {
        session.getattr(attr)?.call0()?
    } else {
        session
    };

    let capsule = capsule.cast::<PyCapsule>().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "session did not produce a PyCapsule for the logical extension codec: {e}"
        ))
    })?;

    // `pointer_checked` validates the capsule name matches before handing back
    // the pointer, so an unexpectedly-named capsule is rejected here.
    let expected = CString::new("datafusion_logical_extension_codec").unwrap();
    let ptr = capsule
        .pointer_checked(Some(expected.as_c_str()))
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "capsule is not a datafusion_logical_extension_codec: {e}"
            ))
        })?;

    // SAFETY: The capsule was produced by datafusion-python and contains a
    // valid FFI_LogicalExtensionCodec (#[repr(C)] StableAbi struct).
    let codec = unsafe { &*(ptr.as_ptr() as *const FFI_LogicalExtensionCodec) };
    Ok(codec.clone())
}

// ============================================================================
// Python-visible Table Class
// ============================================================================

/// A lazy table provider that wraps Python stream factory functions.
///
/// This class implements the `__datafusion_table_provider__` protocol, allowing
/// it to be registered with DataFusion's `SessionContext.register_table()`.
///
/// Data is NOT read until query execution time - this enables true lazy evaluation.
/// Each partition has its own factory function that is called on query execution
/// to create a fresh stream, enabling true parallelism in DataFusion.
///
/// ## Filter Pushdown
///
/// SQL filters on dimension columns (time, lat, lon, etc.) automatically prune
/// partitions that can't contain matching rows when metadata is supplied.
///
/// # Example
///
/// ```python
/// from datafusion import SessionContext
/// import pyarrow as pa
///
/// schema = pa.schema([("time", pa.int64()), ("air", pa.float32())])
///
/// # Each element is a (factory_callable, metadata_dict, num_rows) tuple.
/// # metadata_dict maps dim name -> (min, max, dtype_str); use {} for no pruning.
/// # num_rows is the exact partition row count.
/// def make_partitions():
///     yield (lambda: pa.RecordBatchReader.from_batches(schema, batches_0),
///            {"time": (0, 1_000_000_000, "int64")}, len(batches_0_rows))
///     yield (lambda: pa.RecordBatchReader.from_batches(schema, batches_1),
///            {"time": (1_000_000_001, 2_000_000_000, "int64")}, len(batches_1_rows))
///
/// table = LazyArrowStreamTable(make_partitions(), schema)
///
/// ctx = SessionContext()
/// ctx.register_table("air", table)
/// result = ctx.sql("SELECT AVG(air) FROM air WHERE time > 500000000").to_arrow_table()
/// ```

#[pyclass(name = "LazyArrowStreamTable")]
struct LazyArrowStreamTable {
    /// The underlying table provider with pruning support
    table: Arc<PrunableStreamingTable>,
}

#[pymethods]
impl LazyArrowStreamTable {
    /// Create a new LazyArrowStreamTable from an iterable of partition pairs.
    ///
    /// Args:
    ///     partitions: Any Python iterable yielding
    ///             ``(factory, metadata_dict, num_rows)`` tuples, where:
    ///             - ``factory`` is a zero-argument callable returning a
    ///               ``pa.RecordBatchReader`` (called lazily at query time).
    ///             - ``metadata_dict`` is a ``dict[str, tuple[Any, Any, str]]``
    ///               mapping dimension name to ``(min, max, dtype_str)``; pass
    ///               ``{}`` to skip pruning for a partition.
    ///             - ``num_rows`` is the exact row count for the partition, so
    ///               the scan reports exact ``Statistics`` to the optimizer.
    ///             Generators are accepted, so partition state can be produced
    ///             one item at a time and released after Rust stores it.
    ///     schema: A PyArrow Schema for the table.
    ///
    /// Raises:
    ///     TypeError: If the schema is not a valid PyArrow Schema.
    ///     ValueError: If the partitions iterable is empty.
    #[new]
    #[pyo3(signature = (partitions, schema))]
    fn new(partitions: &Bound<'_, PyAny>, schema: &Bound<'_, PyAny>) -> PyResult<Self> {
        use arrow::datatypes::Schema;
        use arrow::pyarrow::FromPyArrow;

        let arrow_schema = Schema::from_pyarrow_bound(schema).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Failed to convert schema: {e}"))
        })?;
        let schema_ref = Arc::new(arrow_schema);

        // Consume the Python iterable one item at a time.
        // All GIL-bound work happens here, in a single GIL-held context,
        // eliminating the per-partition Python::attach() calls of the old
        // three-list approach.  Python can release each block dict, factory
        // closure, and metadata dict as soon as Rust has ingested them.
        // Stored as Arc<dyn ProjectableStream> so PrunableStreamingTable
        // is decoupled from PyArrowStreamPartition.
        let mut partition_list: Vec<(Arc<dyn ProjectableStream>, PartitionMetadata)> = Vec::new();
        for item_result in partitions.try_iter()? {
            let item = item_result?;
            // Each partition is a ``(factory, metadata_dict, num_rows)`` tuple;
            // `num_rows` is the exact per-partition row count that feeds the
            // scan's statistics.
            let (factory_obj, meta_obj, num_rows): (Py<PyAny>, Py<PyAny>, usize) =
                item.extract().map_err(|e| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "each partition must be a (factory, metadata_dict, num_rows) tuple: {e}"
                    ))
                })?;
            let ranges = convert_python_ranges_from_bound(meta_obj.bind(partitions.py()))?;
            let meta = PartitionMetadata { ranges, num_rows };
            let partition: Arc<dyn ProjectableStream> =
                Arc::new(PyArrowStreamPartition::new(factory_obj, schema_ref.clone()));
            partition_list.push((partition, meta));
        }

        if partition_list.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "partitions iterable must not be empty",
            ));
        }

        let table = PrunableStreamingTable::new(schema_ref, partition_list);
        Ok(Self {
            table: Arc::new(table),
        })
    }

    /// Returns a PyCapsule implementing the DataFusion TableProvider FFI.
    ///
    /// This method is called by DataFusion's `register_table()` to get a
    /// foreign table provider that can be used in queries.
    ///
    /// In DataFusion 52+, the caller passes `session` (a `SessionContext`)
    /// so that the provider can access task-context and codec information
    /// needed for physical plan serialisation across the FFI boundary.
    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = ffi_logical_codec_from_pycapsule(session)?;

        let provider: Arc<dyn TableProvider + Send> = self.table.clone();

        let ffi_provider = FFI_TableProvider::new_with_ffi_codec(provider, true, None, codec);

        let name = CString::new("datafusion_table_provider").unwrap();
        PyCapsule::new(py, ffi_provider, Some(name))
    }

    /// Get the schema of the table as a PyArrow Schema.
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.table
            .schema()
            .to_pyarrow(py)
            .map(|bound| bound.unbind())
    }

    fn __repr__(&self) -> String {
        format!("LazyArrowStreamTable(schema={:?})", self.table.schema())
    }
}

// ============================================================================
// Autograd: SQL-level grad() rewrite
// ============================================================================

/// Rewrite every `grad`/`jvp`/`vjp` call in a SQL query into its symbolic
/// derivative, returning the rewritten SQL text.
///
/// The autograd engine operates on DataFusion logical `Expr` trees. Rather than
/// round-tripping a whole plan across the cdylib boundary, this rewrites the
/// query as **SQL text** before it is planned: each marker call is parsed,
/// differentiated, and rendered back to SQL in place. Because it runs before
/// planning, it works for any query shape the parser accepts — recursive CTEs,
/// DML, and subqueries — which the plan-level Substrait bridge could not carry.
///
/// Args:
///     query: A SQL query string that may contain `grad`/`jvp`/`vjp` calls.
///
/// Returns:
///     The rewritten SQL string, ready to pass to ``SessionContext.sql``.
#[pyfunction]
fn rewrite_grad_sql(query: &str) -> PyResult<String> {
    autograd::rewrite_grad_in_sql(query).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "rewrite_grad_sql: failed to rewrite grad() calls: {e}"
        ))
    })
}

/// Differentiate a SQL scalar expression symbolically and return the
/// derivative as SQL text.
///
/// Where [`grad_rewrite`] rewrites `grad(...)` calls inside a whole plan, this
/// differentiates a single expression and hands back the result as SQL — the
/// autograd engine acting as a "calculus compiler". It lets a caller obtain an
/// update rule once and embed it in queries the Substrait round-trip can't
/// carry a `grad` marker through, such as a recursive-CTE training loop.
///
/// Args:
///     expr: A SQL scalar expression over `columns` (e.g. `"sin(x) * x"`).
///     wrt: The column name to differentiate with respect to.
///     columns: The column names in scope; all treated as `Float64` (enough to
///         parse and differentiate — types don't affect the symbolic result).
///
/// Returns:
///     The derivative as a SQL string (e.g. `"cos(x) * x + sin(x)"`).
#[pyfunction]
fn differentiate_sql(expr: &str, wrt: &str, columns: Vec<String>) -> PyResult<String> {
    let ctx = SessionContext::new();

    let fields: Vec<Field> = columns
        .iter()
        .map(|name| Field::new(name, DataType::Float64, true))
        .collect();
    let df_schema = DFSchema::try_from(Schema::new(fields)).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "differentiate_sql: failed to build schema: {e}"
        ))
    })?;

    let parsed = ctx.parse_sql_expr(expr, &df_schema).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "differentiate_sql: failed to parse expression '{expr}': {e}"
        ))
    })?;

    let derivative = autograd::differentiate(&parsed, wrt).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "differentiate_sql: failed to differentiate: {e}"
        ))
    })?;

    let sql = expr_to_sql(&derivative).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "differentiate_sql: failed to render derivative to SQL: {e}"
        ))
    })?;

    Ok(sql.to_string())
}

/// Python module initialization
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LazyArrowStreamTable>()?;
    m.add_function(wrap_pyfunction!(rewrite_grad_sql, m)?)?;
    m.add_function(wrap_pyfunction!(differentiate_sql, m)?)?;
    Ok(())
}
