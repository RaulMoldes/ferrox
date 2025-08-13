// src/backend/number.rs

use ndarray::{LinalgScalar, ScalarOperand};
use rand_distr::num_traits::{FromPrimitive, One, Zero};
use serde::{Deserialize, Serialize};
use std::cmp::{PartialEq, PartialOrd};
use std::default::Default;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

// Import cudarc traits only when cuda feature is enabled
#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

/// Base trait for all numeric types in FerroxN.
/// Provides common interface for arithmetic operations, comparisons, and conversions.
/// Designed for both integer and floating-point types, excluding unsigned integers
/// due to lack of negative value support required for negation and signum operations.
pub trait FerroxN:
    // Basic arithmetic operations
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
    + Sum<Self> + for<'a> Sum<&'a Self> +
    // Assignment operations
    AddAssign + SubAssign + MulAssign + DivAssign +
    // Remainder operation
    Rem<Output = Self> +
    // Negation
    Neg<Output = Self> +
    // Comparisons
    PartialOrd + PartialEq + Neg +
    // Essential traits
    Clone + Copy + Debug + Display + PartialOrd + PartialEq +
    Default +

    // Only conversions that always work without loss
    From<i8> +
    Sized
    + Zero
    + One
    + FromPrimitive + Serialize + for<'a> Deserialize<'a>
    + LinalgScalar + ScalarOperand + 'static
{
    /// Neutral element for addition (zero)
    fn zero() -> Self;

    /// Neutral element for multiplication (one)
    fn one() -> Self;

    /// Checks if the value is zero
    fn is_zero(&self) -> bool {
        *self == <Self as FerroxN>::zero()
    }

    /// Checks if the value is one
    fn is_one(&self) -> bool {
        *self == <Self as FerroxN>::one()
    }

    /// Absolute value
    fn abs(self) -> Self;

    /// Sign of the number (-1, 0, 1)
    fn signum(self) -> Self;

    /// Power using an integer exponent
    fn powi(self, exp: i32) -> Self;

    fn max(self, other: Self) -> Self{
        if self >= other {
            return self;
        }
        other
    }

    fn min(self, other: Self) -> Self{
        if self <= other {
            return self;
        }
        other
    }

    /// Converts to f64 for operations that require floating point
    fn to_f64(self) -> f64;

    /// Converts to f32 for operations that require floating point
    fn to_f32(self) -> f32;

    /// Converts from f32 (may fail for integer types if there's precision loss)
    fn from_f32(value: f32) -> Option<Self>;

    /// Converts from f64 (may fail for integer types if there's precision loss)
    fn from_f64(value: f64) -> Option<Self>;

    /// Converts from i32 (may fail if there's precision loss)
    fn from_i32(value: i32) -> Option<Self>;

    /// Converts from i16 (may fail if there's precision loss)
    fn from_i16(value: i16) -> Option<Self>;

    /// Converts from i64 (may fail if there's precision loss)
    fn from_i64(value: i64) -> Option<Self>;

    /// Minimum value representable by this type
    fn min_value() -> Self;

    /// Maximum value representable by this type
    fn max_value() -> Self;


}

/// Additional trait for floating-point FerroxN types
/// Now properly extends FerroxN instead of the incorrect CPUNumber trait
pub trait FerroxF: FerroxN {
    /// Provide default implementations using FerroxN trait methods
    fn zero() -> Self {
        <Self as FerroxN>::from_f64(0.0).expect("Failed to convert 0.0 to FerroxF type")
    }

    fn one() -> Self {
        <Self as FerroxN>::from_f64(1.0).expect("Failed to convert 1.0 to FerroxF type")
    }

    /// Square root
    fn sqrt(self) -> Self;

    /// Exponential function (e^x)
    fn exp(self) -> Self;

    /// Natural logarithm
    fn ln(self) -> Self;

    /// Base-10 logarithm
    fn log10(self) -> Self;

    /// Power with floating-point exponent
    fn powf(self, exp: Self) -> Self;

    /// Sine
    fn sin(self) -> Self;

    /// Cosine
    fn cos(self) -> Self;

    /// Tangent
    fn tan(self) -> Self;

    /// Checks if it's NaN
    fn is_nan(self) -> bool;

    /// Checks if it's infinite
    fn is_infinite(self) -> bool;

    /// Checks if it's finite
    fn is_finite(self) -> bool;

    /// Epsilon for floating-point comparisons
    fn epsilon() -> Self;
}

// ============= GPU TRAIT DEFINITIONS =============

/// CUDA-compatible numeric trait that extends FerroxN with GPU-specific requirements
/// This trait is for all numeric types (integers and floats) that can be used on GPU
#[cfg(feature = "cuda")]
pub trait FerroxCudaN: FerroxN + DeviceRepr + ValidAsZeroBits + Unpin + 'static {}

/// When CUDA is not available, FerroxCudaN is just an alias for FerroxN
#[cfg(not(feature = "cuda"))]
pub trait FerroxCudaN: FerroxN {}

/// CUDA-compatible floating-point trait that extends both FerroxF and FerroxCudaN
/// This ensures floating-point GPU types have both float operations and GPU compatibility
#[cfg(feature = "cuda")]
pub trait FerroxCudaF: FerroxF + FerroxCudaN {}

/// When CUDA is not available, FerroxCudaF extends both FerroxF and FerroxCudaN
#[cfg(not(feature = "cuda"))]
pub trait FerroxCudaF: FerroxF + FerroxCudaN {}

// ============= FERROXN IMPLEMENTATIONS =============

// Implementation for f64
impl FerroxN for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn signum(self) -> Self {
        self.signum()
    }

    fn powi(self, exp: i32) -> Self {
        self.powi(exp)
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f32(value: f32) -> Option<Self> {
        Some(value as f64)
    }

    fn from_f64(value: f64) -> Option<Self> {
        Some(value)
    }

    fn from_i32(value: i32) -> Option<Self> {
        Some(value as f64)
    }

    fn from_i16(value: i16) -> Option<Self> {
        Some(value as f64)
    }

    fn from_i64(value: i64) -> Option<Self> {
        Some(value as f64)
    }

    fn min_value() -> Self {
        f64::MIN
    }

    fn max_value() -> Self {
        f64::MAX
    }
}

impl FerroxF for f64 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn log10(self) -> Self {
        self.log10()
    }

    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn is_nan(self) -> bool {
        self.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn epsilon() -> Self {
        f64::EPSILON
    }
}

// Implementation for f32
impl FerroxN for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn signum(self) -> Self {
        self.signum()
    }

    fn powi(self, exp: i32) -> Self {
        self.powi(exp)
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(value: f32) -> Option<Self> {
        Some(value)
    }

    fn from_f64(value: f64) -> Option<Self> {
        if value.is_finite() && value >= f32::MIN as f64 && value <= f32::MAX as f64 {
            Some(value as f32)
        } else {
            None
        }
    }

    fn from_i32(value: i32) -> Option<Self> {
        // i32 values up to 2^24 can be exactly represented in f32
        if value.abs() <= (1 << 24) {
            Some(value as f32)
        } else {
            None
        }
    }

    fn from_i16(value: i16) -> Option<Self> {
        Some(value as f32)
    }

    fn from_i64(value: i64) -> Option<Self> {
        // Only small i64 values can be exactly represented in f32
        if value.abs() <= (1i64 << 24) {
            Some(value as f32)
        } else {
            None
        }
    }

    fn min_value() -> Self {
        f32::MIN
    }

    fn max_value() -> Self {
        f32::MAX
    }
}

impl FerroxF for f32 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn log10(self) -> Self {
        self.log10()
    }

    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn is_nan(self) -> bool {
        self.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn epsilon() -> Self {
        f32::EPSILON
    }
}

// Implementation for i32
impl FerroxN for i32 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn signum(self) -> Self {
        self.signum()
    }

    fn powi(self, exp: i32) -> Self {
        if exp < 0 {
            if (self == 1) || (self == -1 && exp % 2 == 0) {
                1
            } else if self == -1 {
                -1
            } else {
                0
            }
        } else {
            self.saturating_pow(exp as u32)
        }
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f32(value: f32) -> Option<Self> {
        // Check if value fits in i32 and is exact integer
        if value.is_finite() && value.fract() == 0.0 {
            let int_val = value as i64; // Use i64 to avoid temporary overflow
            if int_val >= i32::MIN as i64 && int_val <= i32::MAX as i64 {
                return Some(int_val as i32);
            }
        }
        None
    }

    fn from_f64(value: f64) -> Option<Self> {
        if value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64 {
            Some(value as i32)
        } else {
            None
        }
    }

    fn from_i32(value: i32) -> Option<Self> {
        Some(value)
    }

    fn from_i16(value: i16) -> Option<Self> {
        Some(value as i32)
    }

    fn from_i64(value: i64) -> Option<Self> {
        if value >= i32::MIN as i64 && value <= i32::MAX as i64 {
            Some(value as i32)
        } else {
            None
        }
    }

    fn min_value() -> Self {
        i32::MIN
    }

    fn max_value() -> Self {
        i32::MAX
    }
}

// Implementation for i64
impl FerroxN for i64 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn signum(self) -> Self {
        self.signum()
    }

    fn powi(self, exp: i32) -> Self {
        if exp < 0 {
            if (self == 1) || (self == -1 && exp % 2 == 0) {
                1
            } else if self == -1 {
                -1
            } else {
                0
            }
        } else {
            self.saturating_pow(exp as u32)
        }
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn from_f32(value: f32) -> Option<Self> {
        // Check that it's finite and exact integer
        if value.is_finite() && value.fract() == 0.0 {
            // Use f64 to avoid overflow
            let int_val = value as f64;
            // Check i64 range
            if int_val >= i64::MIN as f64 && int_val <= i64::MAX as f64 {
                return Some(int_val as i64);
            }
        }
        None
    }

    fn from_f64(value: f64) -> Option<Self> {
        if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
            Some(value as i64)
        } else {
            None
        }
    }

    fn from_i32(value: i32) -> Option<Self> {
        Some(value as i64)
    }

    fn from_i16(value: i16) -> Option<Self> {
        Some(value as i64)
    }

    fn from_i64(value: i64) -> Option<Self> {
        Some(value)
    }

    fn min_value() -> Self {
        i64::MIN
    }

    fn max_value() -> Self {
        i64::MAX
    }
}

// ============= CUDA IMPLEMENTATIONS =============

// When CUDA is enabled, implement FerroxCudaN for numeric types that satisfy GPU requirements
#[cfg(feature = "cuda")]
impl FerroxCudaN for f32 {}

#[cfg(feature = "cuda")]
impl FerroxCudaN for f64 {}

#[cfg(feature = "cuda")]
impl FerroxCudaN for i32 {}

#[cfg(feature = "cuda")]
impl FerroxCudaN for i64 {}

// When CUDA is enabled, implement FerroxCudaF for floating-point types
// These automatically get both FerroxF and FerroxCudaN functionality
#[cfg(feature = "cuda")]
impl FerroxCudaF for f32 {}

#[cfg(feature = "cuda")]
impl FerroxCudaF for f64 {}

// ============= NON-CUDA IMPLEMENTATIONS =============

// When CUDA is not available, provide blanket implementations
// All FerroxN types automatically implement FerroxCudaN
#[cfg(not(feature = "cuda"))]
impl<T: FerroxN> FerroxCudaN for T {}

// All FerroxF types automatically implement FerroxCudaF
// Since FerroxCudaF requires both FerroxF and FerroxCudaN, and we have blanket impl for FerroxCudaN
#[cfg(not(feature = "cuda"))]
impl<T: FerroxF> FerroxCudaF for T {}
