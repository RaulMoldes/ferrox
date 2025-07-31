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
pub mod backend;
// Re-export commonly used types for convenience
pub use backend::{ Device, GPUFloat, CPUFloat, cpu, default_device};
