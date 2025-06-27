# Ferrox

**Ferrox** is a high-performance automatic differentiation engine written in **Rust**, featuring both CPU and GPU acceleration capabilities. Inspired by [PyTorch's autograd](https://pytorch.org/docs/stable/autograd.html) and the [CMU Deep Learning Systems course](https://dlsyscourse.org/), it implements reverse-mode automatic differentiation (backpropagation) with dynamic computation graph construction.

## Architecture

**Core Components:**
- **Tensor Engine**: Dynamic computation graph with automatic differentiation
- **Backend System**: Pluggable CPU/GPU computation backends
- **Neural Network Library**: High-level APIs for deep learning primitives
- **Optimization Suite**: Advanced optimizers (SGD, Adam, RMSprop, AdaGrad)

**Supported Backends:**
- **CPU**: `ndarray`-based computations with BLAS acceleration
- **CUDA**: GPU-accelerated operations with custom CUDA kernels

## Features

### Core Engine
- Reverse-mode automatic differentiation with dynamic graphs
- Memory-efficient gradient computation and accumulation
- Operator overloading for intuitive mathematical expressions
- Graph visualization with GraphViz integration
- Zero-copy tensor operations where possible

### GPU Acceleration
- Custom CUDA kernels for elementwise operations
- Efficient memory management with cudarc integration
- Asynchronous GPU operations with stream support
- Mixed-precision computation support (f32/f64)

### Neural Networks
- Modular layer system (Linear, Activation functions)
- Parameter management with automatic gradient tracking
- Flexible weight initialization strategies
- Training/evaluation mode switching

### Optimization
- Multiple optimizer implementations (SGD, Adam, RMSprop, AdaGrad)
- Learning rate scheduling support
- Weight decay and momentum
- AMSGrad variant support

## System Requirements

### Basic Requirements
- **Rust**: 1.75.0 or later
- **Operating System**: Linux, macOS, or Windows

### GPU Requirements (Optional)
- **NVIDIA GPU**: Compute Capability 6.0 or higher
- **CUDA Toolkit**: 11.0 or later with cuCtxCreate_v4 support
- **NVIDIA Drivers**: 470.0 or later

## Installation

### Prerequisites

Install Rust and Cargo:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

For graph visualizations, install GraphViz:

**Ubuntu/Debian:**
```bash
sudo apt-get install graphviz
```

**macOS:**
```bash
brew install graphviz
```

**Windows:**
```bash
choco install graphviz
```

Verify installation:
```bash
dot -V
```

### CUDA Setup (Optional)

For GPU acceleration, ensure CUDA is properly installed:
```bash
nvcc --version
nvidia-smi
```

## Building and Running

### CPU-only Build
```bash
cargo build --features cpu
cargo run --features cpu
```

### GPU-enabled Build
```bash
cargo build --features cuda
cargo run --features cuda
```

### Testing
```bash
# Run all tests
cargo test

# Test specific features
cargo test --features cpu
cargo test --features cuda

# GPU-specific integration tests
cargo run --features cuda --example cuda_test
```

## Usage Examples

### Basic Tensor Operations
```rust
use ferrox::tensor::Tensor;
use ferrox::graph::Engine;

let mut engine = Engine::new();

// Create tensors
let a = Tensor::new(ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
let b = Tensor::new(ndarray::arr2(&[[2.0, 0.0], [1.0, 3.0]]).into_dyn());

// Create nodes in computation graph
let a_node = engine.variable(a, true); // requires_grad = true
let b_node = engine.variable(b, true);

// Perform operations
let result = engine.matmul(a_node, b_node)?;

// Compute gradients
engine.backward(result)?;
```

### Neural Network Training
```rust
use ferrox::nn::{Linear, Module};
use ferrox::nn::optim::Adam;

// Create model
let linear = Linear::new(784, 10, true); // input_size, output_size, bias

// Create optimizer
let mut optimizer = Adam::with_defaults(0.001);

// Training loop
for (inputs, targets) in dataloader {
    optimizer.zero_grad(&mut engine);
    
    let outputs = linear.forward(&mut engine, inputs)?;
    let loss = compute_loss(outputs, targets);
    
    engine.backward(loss)?;
    optimizer.step(&mut engine)?;
}
```

### GPU Acceleration
```rust
use ferrox::backend::get_backend;

let backend = get_backend();
if backend.has_cuda() {
    // Tensors automatically use GPU when available
    let gpu_tensor = tensor.to_cuda()?;
    let result = gpu_tensor.add_cuda(&other_gpu_tensor)?;
}
```

## Examples

Run the provided examples:
```bash
# Basic engine demonstration
cargo run --example ferrox_example

# GPU performance testing
cargo run --features cuda --example cuda_test

# Neural network training
cargo run --example neural_network_training
```

## Performance Characteristics

### CPU Backend
- BLAS-optimized linear algebra operations
- Memory-efficient gradient computation
- Single and double precision support

### GPU Backend
- Custom CUDA kernels optimized for ML workloads
- Asynchronous execution with CUDA streams
- Memory coalescing for optimal bandwidth utilization
- Supports batch operations for improved throughput

## Project Status

### Current Implementation Status

**Stable Features:**
- Core automatic differentiation engine
- CPU tensor operations with ndarray backend
- Basic neural network layers (Linear)
- Optimization algorithms (SGD, Adam, RMSprop, AdaGrad)
- Graph visualization system

**GPU Features:**
- CUDA context management and memory allocation
- Basic elementwise operations (add, mul, div, sub)
- Activation functions (ReLU, Sigmoid)
- Reduction operations (sum, mean)

**In Development:**
- Convolution operations (GPU implementation in progress)
- Tensor dimension manipulation (currently CPU-only)
- Advanced layer types (Conv2D, BatchNorm, Dropout)
- Mixed-precision training support

**Planned Features:**
-  Distributed training support
-  Model serialization/deserialization
-  ONNX export capabilities
-  WebGPU backend for web deployment

## Troubleshooting

### Common Issues

**CUDA Compilation Errors:**
- Ensure CUDA toolkit version supports cuCtxCreate_v4
- Verify NVIDIA drivers are up to date
- Check that NVCC is in your PATH

**Memory Issues:**
- Monitor GPU memory usage with `nvidia-smi`
- Reduce batch sizes for large models
- Use gradient checkpointing for memory-intensive training

**Performance Issues:**
- Verify GPU operations are being used (check with `nvidia-smi`)
- Ensure tensors are properly transferred to GPU
- Use appropriate batch sizes for your hardware

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Ensure all tests pass: `cargo test --all-features`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author's Note

**Compatibility Notice:** With the upgrade to cudarc 0.16.5, this project requires CUDA installations that support cuCtxCreate_v4. Systems with older CUDA versions may encounter the following error when launching kernels:

```
DlSym { desc: "/usr/lib64-nvidia/libcuda.so: undefined symbol: cuCtxCreate_v4" }
```

If you encounter this error, please upgrade your CUDA installation to version 11.0 or later, or use the CPU-only build with `--features cpu`.

**Implementation Status:** Convolution operations are currently being implemented and will be available in future releases. Tensor dimension manipulation operations (reshape, transpose, etc.) are currently CPU-only but GPU implementations are planned.