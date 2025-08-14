# Ferrox

A lightweight, high-performance automatic differentiation engine written in Rust, inspired by PyTorch's autograd and the CMU Deep Learning Systems course.

## Overview

I created Ferrox as a learning project to understand the internals of automatic differentiation engines while building a practical deep learning framework. The project implements reverse-mode automatic differentiation (backpropagation) with dynamic computation graph construction, making it suitable for both research and educational purposes.

## Features

### Core Engine
- **Reverse-mode automatic differentiation**: Full backpropagation support with gradient accumulation
- **Dynamic computation graph**: Build graphs on-the-fly like PyTorch
- **Multi-device support**: CPU and CUDA backends with automatic device management
- **Memory efficient**: Optimized storage backend with optional jemalloc integration

### Tensor Operations
- **Comprehensive operator support**: All fundamental arithmetic, unary, and matrix operations
- **Broadcasting**: Automatic shape broadcasting for element-wise operations
- **Reduction operations**: Sum, mean, max, min with axis specification
- **Shape manipulation**: Reshape, transpose, squeeze, unsqueeze operations

### Neural Network Components
- **Module system**: PyTorch-like module interface for building neural networks
- **Layer implementations**: Linear layers, activation functions (ReLU, Sigmoid, Tanh)
- **Parameter management**: Automatic parameter registration and gradient tracking
- **Loss functions**: Built-in loss functions with automatic gradient computation

### Development Tools
- **Graph visualization**: Export computation graphs for debugging
- **Comprehensive testing**: Gradient checking and numerical verification
- **Type safety**: Written in 100% safe Rust with strong type guarantees

## Quick Start

```rust
use ferrox::prelude::*;

// Create computation graph
let mut graph = AutoFerroxEngine::<f32>::new();
graph.set_training(true);

// Create input tensor
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
let x_node = graph.create_variable(x, true);

// Apply operations
let y_node = graph.apply_operation(
    Box::new(ops::ReLU),
    vec![x_node]
)?;

// Compute gradients
graph.backward(y_node)?;
let gradients = graph.get_gradients(x_node)?;
```

## Architecture

The project is organized into several key modules:

- **Backend**: Device abstraction and tensor storage
- **Graph**: Automatic differentiation engine and computation graph
- **Operations**: All tensor operations with forward and backward implementations
- **Neural Networks**: High-level modules for building neural networks
- **Datasets**: Data loading and preprocessing utilities

## Project Structure

```
src/
├── backend/        # Device management and tensor storage
├── graph/          # Automatic differentiation engine
├── ops/            # Tensor operations (arithmetic, unary, matrix, etc.)
├── nn/             # Neural network modules and layers
├── dataset/        # Data loading utilities
└── lib.rs          # Main library entry point
```

## Requirements

- Rust 1.70+
- Optional: CUDA toolkit for GPU support
- Optional: GraphViz for computation graph visualization

## Building

```bash
# CPU-only build
cargo build --release

# With CUDA support
cargo build --release --features cuda

# With memory optimization
cargo build --release --features jemalloc
```

## Testing

I maintain comprehensive test coverage including gradient checking:

```bash
# Run all tests
cargo test

# Run with CUDA
cargo test --features cuda

# Specific test modules
cargo test graph::
cargo test ops::
```

## Contributing

This is primarily a learning project, but I welcome contributions that help improve the codebase or add educational value. Please ensure all tests pass and include appropriate documentation.


For adding new CUDA kernels or optimizing existing ones, see CUDA_DEVELOPMENT.md for detailed guidelines on kernel development, benchmarking, and validation against PyTorch.

## License

MIT License - see LICENSE file for details.

## Author

**Raul Moldes**
- GitHub: [@RaulMoldes](https://github.com/RaulMoldes)
- Email: raul.moldes.work@gmail.com

## Acknowledgments

- Inspired by the CMU Deep Learning Systems course (dlsyscourse.org)
- PyTorch's autograd system for API design
- The Rust ML community for guidance and inspiration
