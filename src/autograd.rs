//! Symbolic differentiation of DataFusion logical [`Expr`] trees.
//!
//! This is the autograd engine for xarray-sql. Given an [`Expr`] and the name
//! of a column to differentiate with respect to, [`differentiate`] returns a
//! new [`Expr`] for the (symbolic) partial derivative, built entirely from
//! ordinary DataFusion expressions so the result can be planned and evaluated
//! by DataFusion like any other SQL expression.
//!
//! ## Design
//!
//! The approach mirrors JAX's per-primitive rule registry (`defjvp` and
//! friends in `jax/_src/interpreters/ad.py`): every expression node has a
//! differentiation rule, and the chain rule composes them as the tree is
//! walked. Because each row of a relational table is an independent evaluation
//! point, differentiating a column expression and letting DataFusion evaluate
//! it row-by-row is the moral equivalent of `jax.vmap(jax.grad(f))` — the rows
//! *are* the batch dimension.
//!
//! A small simplifier folds the `0`/`1` constants that differentiation
//! produces in abundance (e.g. `d/dx (c) = 0`, `d/dx (x) = 1`), keeping output
//! expressions compact. This plays the role of JAX's `Zero` tangents and
//! `add_tangents`: a `0` derivative short-circuits products and drops out of
//! sums, and a `1` factor drops out of products.
//!
//! ## Scope (MVP)
//!
//! This first cut implements scalar `grad`: the partial derivative of a single
//! expression with respect to one named column. Forward-/reverse-mode
//! (`jvp`/`vjp`) and multi-input Jacobians are deliberately left for later.

#![allow(dead_code)]

use std::any::Any;
use std::f64::consts::{LN_10, LN_2};

use datafusion::arrow::datatypes::DataType;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::functions::math::expr_fn;
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{
    lit, BinaryExpr, Cast, ColumnarValue, Expr, LogicalPlan, Operator, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, Volatility,
};

// ---------------------------------------------------------------------------
// Constant helpers and the 0/1-folding builders
// ---------------------------------------------------------------------------

/// The constant `0.0`, used as the derivative of anything not depending on the
/// differentiation variable.
fn zero() -> Expr {
    lit(0.0_f64)
}

/// The constant `1.0`, used as the derivative of the differentiation variable.
fn one() -> Expr {
    lit(1.0_f64)
}

/// Interpret a [`ScalarValue`] as `f64` if it is a (non-null) numeric scalar.
fn scalar_as_f64(sv: &ScalarValue) -> Option<f64> {
    match sv {
        ScalarValue::Float64(Some(v)) => Some(*v),
        ScalarValue::Float32(Some(v)) => Some(*v as f64),
        ScalarValue::Int64(Some(v)) => Some(*v as f64),
        ScalarValue::Int32(Some(v)) => Some(*v as f64),
        ScalarValue::Int16(Some(v)) => Some(*v as f64),
        ScalarValue::Int8(Some(v)) => Some(*v as f64),
        ScalarValue::UInt64(Some(v)) => Some(*v as f64),
        ScalarValue::UInt32(Some(v)) => Some(*v as f64),
        ScalarValue::UInt16(Some(v)) => Some(*v as f64),
        ScalarValue::UInt8(Some(v)) => Some(*v as f64),
        _ => None,
    }
}

/// Return the constant `f64` value of a literal expression, if it is one.
fn as_const(e: &Expr) -> Option<f64> {
    match e {
        Expr::Literal(sv, _) => scalar_as_f64(sv),
        _ => None,
    }
}

/// True if the expression is a numeric literal exactly equal to zero.
fn is_zero(e: &Expr) -> bool {
    matches!(as_const(e), Some(v) if v == 0.0)
}

/// True if the expression is a numeric literal exactly equal to one.
fn is_one(e: &Expr) -> bool {
    matches!(as_const(e), Some(v) if v == 1.0)
}

fn binary(left: Expr, op: Operator, right: Expr) -> Expr {
    Expr::BinaryExpr(BinaryExpr::new(Box::new(left), op, Box::new(right)))
}

/// `a + b`, dropping a zero operand.
fn add(a: Expr, b: Expr) -> Expr {
    if is_zero(&a) {
        b
    } else if is_zero(&b) {
        a
    } else {
        binary(a, Operator::Plus, b)
    }
}

/// `a - b`, dropping a zero right operand and turning `0 - b` into `-b`.
fn sub(a: Expr, b: Expr) -> Expr {
    if is_zero(&b) {
        a
    } else if is_zero(&a) {
        neg(b)
    } else {
        binary(a, Operator::Minus, b)
    }
}

/// `a * b`, folding `0 * _ = 0` and `1 * b = b` (and the mirror cases).
fn mul(a: Expr, b: Expr) -> Expr {
    if is_zero(&a) || is_zero(&b) {
        zero()
    } else if is_one(&a) {
        b
    } else if is_one(&b) {
        a
    } else {
        binary(a, Operator::Multiply, b)
    }
}

/// `a / b`, folding `0 / _ = 0` and `a / 1 = a`.
fn div(a: Expr, b: Expr) -> Expr {
    if is_zero(&a) {
        zero()
    } else if is_one(&b) {
        a
    } else {
        binary(a, Operator::Divide, b)
    }
}

/// `-a`, folding `-0 = 0`.
fn neg(a: Expr) -> Expr {
    if is_zero(&a) {
        zero()
    } else {
        Expr::Negative(Box::new(a))
    }
}

/// `e * e`.
fn square(e: Expr) -> Expr {
    mul(e.clone(), e)
}

// ---------------------------------------------------------------------------
// The differentiation rules
// ---------------------------------------------------------------------------

/// Differentiate `expr` with respect to the column named `wrt`.
///
/// Returns a new [`Expr`] for the partial derivative, composed of ordinary
/// DataFusion expressions. Returns a [`DataFusionError::NotImplemented`] for
/// expression nodes or scalar functions without a differentiation rule, so the
/// caller can surface a clear, actionable error rather than silently producing
/// a wrong answer.
pub fn differentiate(expr: &Expr, wrt: &str) -> Result<Expr> {
    match expr {
        // d/dx (x) = 1 ; d/dx (y) = 0 for any other column.
        Expr::Column(c) => Ok(if c.name == wrt { one() } else { zero() }),

        // d/dx (constant) = 0.
        Expr::Literal(_, _) => Ok(zero()),

        // An alias is transparent to differentiation; the surrounding query
        // re-applies any output naming.
        Expr::Alias(a) => differentiate(&a.expr, wrt),

        // A numeric cast is (locally) linear: d/dx cast(u) = cast(du). We keep
        // the cast so the derivative retains the declared output type.
        Expr::Cast(c) => {
            let du = differentiate(&c.expr, wrt)?;
            Ok(Expr::Cast(Cast::new(Box::new(du), c.data_type.clone())))
        }

        // d/dx (-u) = -(du).
        Expr::Negative(inner) => Ok(neg(differentiate(inner, wrt)?)),

        Expr::BinaryExpr(be) => diff_binary(be, wrt),

        Expr::ScalarFunction(sf) => diff_scalar_function(sf, wrt),

        other => Err(DataFusionError::NotImplemented(format!(
            "grad: differentiation is not implemented for this expression: {other}"
        ))),
    }
}

/// Differentiate a binary arithmetic expression via the sum/product/quotient
/// rules.
fn diff_binary(be: &BinaryExpr, wrt: &str) -> Result<Expr> {
    let a = be.left.as_ref();
    let b = be.right.as_ref();
    let da = differentiate(a, wrt)?;
    let db = differentiate(b, wrt)?;

    match be.op {
        // d/dx (a + b) = da + db
        Operator::Plus => Ok(add(da, db)),
        // d/dx (a - b) = da - db
        Operator::Minus => Ok(sub(da, db)),
        // d/dx (a * b) = da*b + a*db   (product rule)
        Operator::Multiply => Ok(add(mul(da, b.clone()), mul(a.clone(), db))),
        // d/dx (a / b) = (da*b - a*db) / b^2   (quotient rule)
        Operator::Divide => {
            let numerator = sub(mul(da, b.clone()), mul(a.clone(), db));
            Ok(div(numerator, square(b.clone())))
        }
        op => Err(DataFusionError::NotImplemented(format!(
            "grad: operator '{op}' is not differentiable"
        ))),
    }
}

/// Differentiate a scalar-function call via the chain rule.
///
/// For a unary primitive `f(u)`, the derivative is `f'(u) * du`. For `power`,
/// which is binary, we handle the constant-exponent and constant-base cases.
fn diff_scalar_function(sf: &ScalarFunction, wrt: &str) -> Result<Expr> {
    let name = sf.func.name();
    let args = &sf.args;

    // `power(base, exponent)` is the one binary primitive we differentiate.
    if name == "power" {
        return diff_power(args, wrt);
    }

    if args.len() != 1 {
        return Err(DataFusionError::NotImplemented(format!(
            "grad: no derivative rule for function '{name}' with {} arguments",
            args.len()
        )));
    }

    let u = &args[0];
    let du = differentiate(u, wrt)?;
    // Chain rule short-circuit: if du is 0, the whole derivative is 0 and we
    // avoid emitting the (dead) outer derivative term entirely.
    if is_zero(&du) {
        return Ok(zero());
    }

    let outer = match name {
        // Trigonometric.
        "sin" => expr_fn::cos(u.clone()),
        "cos" => neg(expr_fn::sin(u.clone())),
        "tan" => div(one(), square(expr_fn::cos(u.clone()))),
        // Inverse trigonometric.
        "asin" => div(one(), expr_fn::sqrt(sub(one(), square(u.clone())))),
        "acos" => neg(div(one(), expr_fn::sqrt(sub(one(), square(u.clone()))))),
        "atan" => div(one(), add(one(), square(u.clone()))),
        // Exponential / logarithmic.
        "exp" => expr_fn::exp(u.clone()),
        "ln" => div(one(), u.clone()),
        "log2" => div(one(), mul(u.clone(), lit(LN_2))),
        "log10" => div(one(), mul(u.clone(), lit(LN_10))),
        "sqrt" => div(one(), mul(lit(2.0_f64), expr_fn::sqrt(u.clone()))),
        // Hyperbolic.
        "sinh" => expr_fn::cosh(u.clone()),
        "cosh" => expr_fn::sinh(u.clone()),
        "tanh" => sub(one(), square(expr_fn::tanh(u.clone()))),
        // Piecewise-linear: derivative is the sign (undefined at 0, like JAX).
        "abs" => expr_fn::signum(u.clone()),
        _ => {
            return Err(DataFusionError::NotImplemented(format!(
                "grad: no derivative rule for function '{name}'"
            )))
        }
    };

    Ok(mul(outer, du))
}

/// Differentiate `power(base, exponent)`.
///
/// * Constant exponent `c`: `d/dx base^c = c * base^(c-1) * d(base)`.
/// * Constant base `a`: `d/dx a^u = a^u * ln(a) * d(u)`.
/// * Both variable (`u^v`): not supported in the MVP.
fn diff_power(args: &[Expr], wrt: &str) -> Result<Expr> {
    if args.len() != 2 {
        return Err(DataFusionError::NotImplemented(
            "grad: power() expects exactly two arguments".to_string(),
        ));
    }
    let base = &args[0];
    let exponent = &args[1];

    match (as_const(base), as_const(exponent)) {
        // Constant exponent (covers the common x^2, x^0.5, ... cases).
        (_, Some(c)) => {
            let dbase = differentiate(base, wrt)?;
            if is_zero(&dbase) {
                return Ok(zero());
            }
            let outer = mul(lit(c), expr_fn::power(base.clone(), lit(c - 1.0)));
            Ok(mul(outer, dbase))
        }
        // Constant base, variable exponent.
        (Some(a), None) => {
            let dexp = differentiate(exponent, wrt)?;
            if is_zero(&dexp) {
                return Ok(zero());
            }
            let outer = mul(expr_fn::power(base.clone(), exponent.clone()), lit(a.ln()));
            Ok(mul(outer, dexp))
        }
        // General u^v requires the exp/log trick; deferred past the MVP.
        (None, None) => Err(DataFusionError::NotImplemented(
            "grad: power(base, exponent) where both depend on the \
             differentiation variable is not yet supported"
                .to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// The `grad` marker UDF and the plan-level rewrite
// ---------------------------------------------------------------------------

/// A no-op placeholder UDF for `grad(expr, column)`.
///
/// `grad` is a *marker*: it carries the differentiation request intact through
/// SQL parsing, logical planning, and Substrait serialization. It is always
/// rewritten away by [`rewrite_grad_calls`] before execution, so its `invoke`
/// is never reached in normal use (and deliberately errors if it somehow is,
/// rather than silently returning a wrong value).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct GradMarker {
    signature: Signature,
}

impl GradMarker {
    pub fn new() -> Self {
        // grad(expr, column): two arguments of any (numeric) type.
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for GradMarker {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for GradMarker {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "grad"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Err(DataFusionError::Execution(
            "grad() marker reached execution without being rewritten; this is \
             an internal xarray-sql autograd error"
                .to_string(),
        ))
    }
}

/// Rewrite every `grad(expr, column)` call anywhere in a logical plan into the
/// symbolic derivative of `expr` with respect to `column`, leaving everything
/// else untouched. The plan's schema is recomputed afterwards because replacing
/// a `grad` call can change an expression's name or type.
pub fn rewrite_grad_calls(plan: LogicalPlan) -> Result<LogicalPlan> {
    let rewritten = plan
        .transform_up(|node| node.map_expressions(rewrite_grad_in_expr))?
        .data;
    rewritten.recompute_schema()
}

/// Replace any `grad(...)` calls nested anywhere inside a single expression.
fn rewrite_grad_in_expr(expr: Expr) -> Result<Transformed<Expr>> {
    expr.transform_up(|e| {
        let Expr::ScalarFunction(sf) = &e else {
            return Ok(Transformed::no(e));
        };
        if sf.func.name() != "grad" {
            return Ok(Transformed::no(e));
        }
        if sf.args.len() != 2 {
            return Err(DataFusionError::Plan(format!(
                "grad() expects two arguments grad(expr, column), got {}",
                sf.args.len()
            )));
        }
        let wrt = match &sf.args[1] {
            Expr::Column(c) => c.name.clone(),
            other => {
                return Err(DataFusionError::Plan(format!(
                    "grad(): the second argument must be a bare column to \
                     differentiate with respect to, got: {other}"
                )))
            }
        };
        let derivative = differentiate(&sf.args[0], &wrt)?;
        Ok(Transformed::yes(derivative))
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::col;

    use super::*;

    #[test]
    fn constant_has_zero_derivative() {
        assert_eq!(differentiate(&lit(3.0_f64), "x").unwrap(), zero());
    }

    #[test]
    fn variable_has_unit_derivative() {
        assert_eq!(differentiate(&col("x"), "x").unwrap(), one());
    }

    #[test]
    fn other_variable_has_zero_derivative() {
        assert_eq!(differentiate(&col("y"), "x").unwrap(), zero());
    }

    #[test]
    fn sum_rule_folds_constants() {
        // d/dx (x + y) = 1 + 0 = 1
        let e = add(col("x"), col("y"));
        assert_eq!(differentiate(&e, "x").unwrap(), one());
    }

    #[test]
    fn product_rule() {
        // d/dx (x * x) = 1*x + x*1 = x + x
        let e = binary(col("x"), Operator::Multiply, col("x"));
        let expected = add(col("x"), col("x"));
        assert_eq!(differentiate(&e, "x").unwrap(), expected);
    }

    #[test]
    fn quotient_rule() {
        // d/dx (x / y) = (1*y - x*0) / (y*y) = y / (y*y)
        let e = binary(col("x"), Operator::Divide, col("y"));
        let expected = div(col("y"), square(col("y")));
        assert_eq!(differentiate(&e, "x").unwrap(), expected);
    }

    #[test]
    fn chain_rule_sin() {
        // d/dx sin(x) = cos(x) * 1 = cos(x)
        let d = differentiate(&expr_fn::sin(col("x")), "x").unwrap();
        assert_eq!(d, expr_fn::cos(col("x")));
        // Readable, precedence-free rendering.
        assert_eq!(d.to_string(), "cos(x)");
    }

    #[test]
    fn composite_sin_times_x() {
        // d/dx (sin(x) * x) = cos(x)*x + sin(x)
        let e = binary(expr_fn::sin(col("x")), Operator::Multiply, col("x"));
        let d = differentiate(&e, "x").unwrap();
        assert_eq!(d.to_string(), "cos(x) * x + sin(x)");
    }

    #[test]
    fn power_constant_exponent() {
        // d/dx power(x, 2) = 2 * power(x, 1) * 1 = 2 * power(x, 1)
        let e = expr_fn::power(col("x"), lit(2.0_f64));
        let expected = mul(lit(2.0_f64), expr_fn::power(col("x"), lit(1.0_f64)));
        assert_eq!(differentiate(&e, "x").unwrap(), expected);
    }

    #[test]
    fn unsupported_operator_errors() {
        let e = binary(col("x"), Operator::Modulo, col("y"));
        assert!(differentiate(&e, "x").is_err());
    }

    #[test]
    fn unsupported_function_errors() {
        // atan2 is binary and has no rule yet.
        let e = expr_fn::atan2(col("x"), col("y"));
        assert!(differentiate(&e, "x").is_err());
    }
}
