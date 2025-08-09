//! # Ferrox
//!
//! Ferrox is a lightweight, CPU-based automatic differentiation engine written in Rust,
//! inspired by PyTorch's autograd and the CMU Deep Learning Systems course.
//!
//! ## Features
//!
//! - Reverse-mode automatic differentiation (backpropagation)
//! - Dynamic computation graph construction
//! - Scalar and tensor support via `ndarray`
//! - Operator overloading for intuitive expressions
//! - Gradient accumulation
//! - Graph visualization (requires GraphViz installed)
//! - High-level neural network modules
//! - Written 100% in safe Rust
//!
#[cfg(feature = "jemalloc")]
use jemallocator::Jemalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[allow(unused_imports)]
pub mod backend;
// Re-export commonly used types for convenience
pub use backend::{cpu, default_device, Device, FerroxCudaF, FerroxF};

pub mod graph;
pub mod nn;
pub mod ops;
