use std::cmp::{PartialEq, PartialOrd};
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};


#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};


/// Trait that defines the basic operations and properties for numeric types.
///This trait is designed to be implemented by both integer and floating-point types,
/// providing a common interface for arithmetic operations, comparisons, and conversions.
/// I did not implement it for unsigned integers because they do not support negative values,
/// which is a requirement for some operations like negation and signum.
/// It includes methods for basic arithmetic operations, assignment operations,
/// and conversions between different numeric types.
pub trait Numeric: 
    // Basic arithmetic operations
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> +
    // Assignment operations
    AddAssign + SubAssign + MulAssign + DivAssign +
    // Remainder operation
    Rem<Output = Self> +
    // Negation
    Neg<Output = Self> +
    // Comparisons
    PartialOrd + PartialEq +
    // Essential traits
    Clone + Copy + Debug + Display +
    // Only conversions that always work without loss
    From<i8> +
    // Known size at compile time
    Sized
{
    /// Neutral element for addition (zero)
    fn zero() -> Self;
    
    /// Neutral element for multiplication (one)
    fn one() -> Self;
    
    /// Checks if the value is zero
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
    
    /// Checks if the value is one
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
    
    /// Absolute value
    fn abs(self) -> Self;
    
    /// Sign of the number (-1, 0, 1)
    fn signum(self) -> Self;
    
    /// Power using an integer exponent
    fn powi(self, exp: i32) -> Self;
    
    /// Converts to f64 for operations that require floating point
    fn to_f64(self) -> f64;
    
    /// Converts from f64 (may fail for integer types if there's precision loss)
    fn from_f64(value: f64) -> Option<Self>;
    
    /// Converts from i32 (may fail if there's precision loss)
    fn from_i32(value: i32) -> Option<Self>;
    
    /// Converts from i16 (may fail if there's precision loss)
    fn from_i16(value: i16) -> Option<Self>;

    // Converts from i64 (may fail if there's precision loss)
    fn from_i64(value: i64) -> Option<Self>;
    
    /// Minimum value representable by this type
    fn min_value() -> Self;
    
    /// Maximum value representable by this type
    fn max_value() -> Self;
}

/// Additional trait for floating-point numeric types
pub trait Float: Numeric {
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

// Implementation for f64
impl Numeric for f64 {
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

impl Float for f64 {
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
impl Numeric for f32 {
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
    fn from_f64(value: f64) -> Option<Self> {
        if value.is_finite() && value >= f32::MIN as f64 && value <= f32::MAX as f64 {
            Some(value as f32)
        } else {
            None
        }
    }
    fn from_i32(value: i32) -> Option<Self> {
        // f32 can exactly represent integers up to 2^24
        if value.abs() <= (1 << 24) {
            Some(value as f32)
        } else {
            Some(value as f32) // May have precision loss but is valid
        }
    }
    fn from_i16(value: i16) -> Option<Self> {
        Some(value as f32)
    }
    fn from_i64(value: i64) -> Option<Self> {
        // f32 can exactly represent integers up to 2^24
        if value.abs() <= (1 << 24) {
            Some(value as f32)
        } else {
            Some(value as f32) // May have precision loss but is valid
        }
    }

    fn min_value() -> Self {
        f32::MIN
    }
    fn max_value() -> Self {
        f32::MAX
    }
}

impl Float for f32 {
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

// Implementation for i8
impl Numeric for i8 {
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
            if self == 1 {
                1
            } else if self == -1 && exp % 2 == 0 {
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
    fn from_f64(value: f64) -> Option<Self> {
        if value.fract() == 0.0 && value >= i8::MIN as f64 && value <= i8::MAX as f64 {
            Some(value as i8)
        } else {
            None
        }
    }

    fn from_i64(value: i64) -> Option<Self> {
        if value >= i8::MIN as i64 && value <= i8::MAX as i64 {
            Some(value as i8)
        } else {
            None
        }
    }

    fn from_i32(value: i32) -> Option<Self> {
        if value >= i8::MIN as i32 && value <= i8::MAX as i32 {
            Some(value as i8)
        } else {
            None
        }
    }
    fn from_i16(value: i16) -> Option<Self> {
        if value >= i8::MIN as i16 && value <= i8::MAX as i16 {
            Some(value as i8)
        } else {
            None
        }
    }

    fn min_value() -> Self {
        i8::MIN
    }
    fn max_value() -> Self {
        i8::MAX
    }
}

// Implementation for i16
impl Numeric for i16 {
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
            if self == 1 {
                1
            } else if self == -1 && exp % 2 == 0 {
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
    fn from_f64(value: f64) -> Option<Self> {
        if value.fract() == 0.0 && value >= i16::MIN as f64 && value <= i16::MAX as f64 {
            Some(value as i16)
        } else {
            None
        }
    }

    fn from_i64(value: i64) -> Option<Self> {
        if value >= i16::MIN as i64 && value <= i16::MAX as i64 {
            Some(value as i16)
        } else {
            None
        }
    }
    fn from_i32(value: i32) -> Option<Self> {
        if value >= i16::MIN as i32 && value <= i16::MAX as i32 {
            Some(value as i16)
        } else {
            None
        }
    }
    fn from_i16(value: i16) -> Option<Self> {
        Some(value)
    }

    fn min_value() -> Self {
        i16::MIN
    }
    fn max_value() -> Self {
        i16::MAX
    }
}

// Implementation for i32
impl Numeric for i32 {
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
            if self == 1 {
                1
            } else if self == -1 && exp % 2 == 0 {
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
    fn from_f64(value: f64) -> Option<Self> {
        if value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64 {
            Some(value as i32)
        } else {
            None
        }
    }

    fn from_i64(value: i64) -> Option<Self> {
        if value >= i32::MIN as i64 && value <= i32::MAX as i64 {
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

    fn min_value() -> Self {
        i32::MIN
    }
    fn max_value() -> Self {
        i32::MAX
    }
}

// Implementation for i64
impl Numeric for i64 {
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
            if self == 1 {
                1
            } else if self == -1 && exp % 2 == 0 {
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
    fn from_f64(value: f64) -> Option<Self> {
        if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
            Some(value as i64)
        } else {
            None
        }
    }

    fn from_i64(value: i64) -> Option<Self> {
        if value >= i64::MIN && value <= i64::MAX {
            Some(value)
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

    fn min_value() -> Self {
        i64::MIN
    }
    fn max_value() -> Self {
        i64::MAX
    }
}

/// CUDA-compatible numeric trait that extends Numeric with GPU-specific requirements.
/// This trait is only available when the "cuda" feature is enabled, allowing the
/// same code to compile with or without CUDA support.
#[cfg(feature = "cuda")]
pub trait NumericCuda: Numeric + DeviceRepr + ValidAsZeroBits + Unpin {}

/// When CUDA is not available, NumericCuda is just an alias for Numeric.
/// This allows the same generic bounds to work regardless of CUDA availability.
#[cfg(not(feature = "cuda"))]
pub trait NumericCuda: Numeric {}

// CUDA trait implementations - only compiled when cuda feature is enabled
#[cfg(feature = "cuda")]
impl NumericCuda for f32 {}

#[cfg(feature = "cuda")]
impl NumericCuda for f64 {}

#[cfg(feature = "cuda")]
impl NumericCuda for i32 {}

#[cfg(feature = "cuda")]
impl NumericCuda for i64 {}

// When CUDA is not available, provide blanket implementation for all Numeric types
#[cfg(not(feature = "cuda"))]
impl<T: Numeric> NumericCuda for T {}