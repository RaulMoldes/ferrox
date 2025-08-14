# Ferrox

Ferrox is a high-performance automatic differentiation engine written in Rust, featuring both CPU and GPU acceleration capabilities. Inspired by PyTorch's autograd and the CMU Deep Learning Systems course, it implements reverse-mode automatic differentiation (backpropagation) with dynamic computation graph construction.
# Architecture

## Core Components:

- **Tensor Engine**: Dynamic computation graph with automatic differentiation
- **Backend System**: Pluggable CPU/GPU computation backends
- **Neural Network Library**: High-level APIs for deep learning primitives
- **Optimization Suite**: Advanced optimizers (SGD, Adam)

## Supported Backends:

- **CPU**: ndarray-based computations with BLAS acceleration
- **CUDA**: GPU-accelerated operations with custom CUDA kernels

# Technical details
## Dual Backend

The backend manager (`src/backend/manager.rs`) is the main dispatcher between CPU and GPU backends. It manages data movement between devices and handles load balancing to optimize performance.

## Memory Management
`src/backend/memory.rs`

- **CPU Memory Allocation**: Ferrox uses jemalloc for efficient CPU memory management. To enable jemalloc, users must compile the library with the jemalloc feature enabled:
```sh
cargo build --features jemalloc
```
- **GPU Memory Allocation**: On CUDA devices, Ferrox uses a custom memory allocator optimized for GPU memory reuse. Similar to PyTorch's CUDACachingAllocator, it minimizes costly memory reallocations by caching GPU allocations.

## CUDA
`src/backend/cuda/mod.rs`
Ferrox’s CUDA backend delivers high-performance GPU acceleration by combining Rust with native CUDA code.

### Integration with cudarc (v0.16.5)
The backend uses cudarc, a Rust crate offering safe and ergonomic CUDA runtime bindings, enabling direct GPU memory management, kernel launches, and synchronization while maintaining safety.
Custom Precompiled CUDA Kernels

To maximize performance, Ferrox includes custom CUDA kernels implemented in C++. These kernels are precompiled ahead of time and loaded dynamically at runtime. This hybrid approach combines Rust’s safety and flexibility with the raw speed of optimized CUDA C++ code.
Seamless Tensor Operations

The CUDA backend transparently manages tensor data transfer between CPU and GPU, efficiently executing core tensor operations such as matrix multiplication, element-wise functions, and reductions by fully utilizing GPU parallelism.

## Computational Graph Engine
`src/backend/graph/mod.rs`

At the core of Ferrox lies a powerful and flexible computational graph engine designed for automatic differentiation and efficient execution.
The Ops trait

Ferrox supports an extensible set of operations (ops) by implementing the Ops trait. This design allows users to extend the framework with custom operations tailored to specific needs, fully integrated with the graph engine.
Automatic Reverse-Mode Differentiation

The engine implements reverse-mode automatic differentiation (backpropagation), dynamically constructing the computation graph during the forward pass. This dynamic graph enables efficient gradient computation for complex models, powering gradient-based optimization algorithms such as SGD and Adam.


### Output Caching

To optimize repeated computations, Ferrox caches intermediate outputs of operations within the computational graph. This caching reduces redundant calculations, speeding up both forward and backward passes—particularly beneficial during iterative training loops.

### Neural Networks
`src/nn/`
Ferrox provides a minimalistic but flexible neural network library built atop the computational graph engine. It offers:

- Common layer types: fully connected (linear), activation functions (ReLU, Sigmoid, etc.), dropout

- Easy model composition with a modular, extensible API
- Support for batch operations and GPU acceleration
- Utilities for loss functions and metrics

This enables rapid prototyping and training of deep learning models.

### Optimization

`src/nn/optim/`

Ferrox includes a suite of advanced optimization algorithms commonly used in machine learning:

- **SGD (Stochastic Gradient Descent)** with momentum and weight decay
- **Adam** optimizer with configurable learning rates and betas
- Pluggable optimizer interface for easy extension with custom optimization strategies

These optimizers seamlessly integrate with the computational graph and neural network libraries to enable efficient training workflows.

## System Requirements

### Basic Requirements

- **Rust**: Version 1.75.0 or later
- **Operating System**: Primarily Linux;
- **Build Tools**: cargo and nvcc for compiling CUDA kernels

### GPU Requirements (Optional)

NVIDIA GPU with Compute Capability 8.0 or higher to support the latest cudarc version.

CUDA Toolkit compatible with the targeted GPU architecture

### GPU Acceleration

CUDA acceleration is optional but recommended for training larger models or computationally intensive tasks.

To enable CUDA backend, ensure the CUDA Toolkit is installed and environment variables such as CUDA_HOME are properly configured.

# Examples

Run the provided examples to get started:

## Visualize computation graph and basic engine demo
```sh
cargo run --example graph_visualization
```

# Train a simple Multi-Layer Perceptron regressor

```sh
cargo run --example mlp_regressor
```

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
