# Ferrox

Ferrox is a high-performance automatic differentiation engine written in Rust, featuring both CPU and GPU acceleration capabilities. Inspired by PyTorch's autograd and the CMU Deep Learning Systems course, it implements reverse-mode automatic differentiation (backpropagation) with dynamic computation graph construction.

## Key Features

**Performance Optimization**

Ferrox incorporates a sophisticated CUDA memory pool designed with intelligent allocation strategies that minimize GPU memory fragmentation. The system employs precompiled GPU kernels to achieve maximum computational throughput, while comprehensive memory usage debugging and profiling tools provide detailed insights into allocation patterns and performance bottlenecks.

**Deep Learning Support**

The framework provides a complete neural network library featuring convolutional layers optimized for modern deep learning workflows. A full automatic differentiation engine with built-in gradient checking ensures numerical accuracy during backpropagation. The architecture supports contemporary neural network designs including convolutional neural networks and multi-layer perceptrons with efficient batching capabilities.

**Dual Backend Architecture**

Ferrox implements a sophisticated dual backend system that seamlessly switches between computational targets. The CPU backend leverages ndarray-based computations enhanced with BLAS acceleration for optimized linear algebra operations. The GPU backend utilizes CUDA acceleration through custom kernel implementations, providing significant performance improvements for large-scale tensor operations.

**Development Tools**

The framework includes comprehensive memory usage statistics that track allocation patterns, peak usage, and efficiency metrics. Built-in debugging utilities enable detailed performance analysis, including memory pool hit rates and computational graph optimization opportunities. The API design follows PyTorch conventions, providing familiar interfaces for developers transitioning from other deep learning frameworks.

## Architecture

### Core Components

- **Tensor Engine**: Dynamic computation graph with automatic differentiation
- **Backend System**: Pluggable CPU/GPU computation backends
- **Neural Network Library**: High-level APIs for deep learning primitives
- **Optimization Suite**: Advanced optimizers (SGD, Adam)

### Recent Improvements

**Enhanced Memory Pool**: The CUDA memory allocation system has been significantly optimized with detailed statistics tracking and advanced debugging capabilities. The new implementation provides comprehensive insights into memory usage patterns, allocation efficiency, and potential optimization opportunities.

**Convolutional Support**: Full 2D convolution implementation with efficient batching support has been integrated into the neural network library. This includes optimized forward and backward passes for convolutional layers, enabling training of modern CNN architectures.

**Performance Monitoring**: Built-in memory usage tracking and cleanup utilities now provide real-time monitoring of computational graph memory consumption. These tools enable developers to identify memory bottlenecks and optimize model performance.

**Kernel Optimization**: Improved CUDA kernel performance specifically targets f64 operations, addressing memory bandwidth limitations and L1 cache efficiency on consumer GPU hardware.

## Technical Details

### Memory Management

**CPU Memory Allocation**: Ferrox utilizes jemalloc for efficient CPU memory management, providing superior allocation performance compared to standard system allocators. This integration reduces memory fragmentation and improves cache locality for CPU-based tensor operations.

```sh
cargo build --features jemalloc
```

**GPU Memory Pool**: The custom CUDA memory allocator implements sophisticated memory management strategies optimized for deep learning workloads, drawing inspiration from PyTorch's CudaCachingAllocator while incorporating several enhancements. Like PyTorch's allocator, the system maintains pools of GPU memory blocks to avoid expensive cudaMalloc/cudaFree operations during training. However, Ferrox's implementation extends this concept with an intelligent bucket-based allocation strategy that categorizes allocations by size ranges, reducing fragmentation more effectively than simple pooling approaches.

The allocator employs configurable LRU eviction policies that automatically manage memory pressure situations, with customizable thresholds per bucket size. Unlike PyTorch's more general-purpose approach, Ferrox's allocator is specifically tuned for the allocation patterns observed in automatic differentiation engines, with optimized bucket sizes that align with typical tensor dimensions in neural network training. Detailed statistics tracking provides comprehensive performance analysis capabilities including hit/miss ratios, memory efficiency metrics, and per-bucket utilization data. The integrated memory leak detection and debugging utilities offer more granular inspection than PyTorch's memory profiler, enabling developers to identify specific allocation patterns that may indicate memory inefficiencies. Additionally, the allocator supports aggressive cleanup modes for memory-constrained environments and provides real-time monitoring of allocation lifecycles.

### CUDA Integration

Ferrox's CUDA backend achieves high-performance GPU acceleration by combining Rust's safety guarantees with optimized low-level GPU programming. The integration with cudarc provides safe CUDA runtime bindings, enabling direct GPU memory management, kernel launches, and synchronization while maintaining memory safety. Custom kernels implemented in optimized C++ deliver maximum computational throughput for critical operations such as matrix multiplication, convolutions, and element-wise operations. The PyTorch-inspired caching allocator minimizes expensive memory allocations by intelligently reusing GPU memory blocks. Seamless tensor operations handle automatic data movement between CPU and GPU contexts, allowing transparent acceleration of computational graphs.

### Computational Graph Engine

At the framework's core lies a sophisticated computational graph engine that enables flexible automatic differentiation and efficient execution. The system supports dynamic graph construction similar to PyTorch, building computation graphs at runtime rather than requiring static graph definition. However, Ferrox's node management system extends beyond PyTorch's approach with a sophisticated caching and cleanup mechanism designed specifically for memory-constrained environments.

The graph engine implements an intelligent node lifecycle management system where each computation node can be configured for different retention policies. Unlike PyTorch's retain_grad mechanism which primarily focuses on gradient retention, Ferrox's system manages both forward pass outputs and gradient information with fine-grained control. Nodes can be marked as persistent (always cached), cleanable (eligible for memory cleanup), or ephemeral (immediately discarded after use). This three-tier system allows developers to optimize memory usage based on specific computational patterns - for instance, keeping activation outputs for layers that will be reused in complex architectures while immediately discarding intermediate results in linear computation chains.

The output caching system provides configurable policies that balance memory usage against recomputation costs. Critical operations like expensive convolutions or matrix multiplications can maintain their outputs in cache for rapid access during gradient computation, while simple element-wise operations may opt for recomputation to conserve memory. The engine tracks node access patterns and automatically suggests optimal caching strategies based on observed usage. Built-in memory debugging capabilities provide detailed statistics about graph memory consumption, including per-node memory footprints, cache hit ratios, and cleanup effectiveness, enabling developers to fine-tune memory usage patterns for their specific workloads.

### Neural Networks

Ferrox provides a comprehensive neural network library built on top of the computational graph engine. The library includes essential layer types such as Linear layers for fully connected networks, Conv2d for convolutional operations, BatchNorm for normalization, Dropout for regularization, and various activation functions. The modular design enables flexible model composition through an extensible API that supports complex network architectures. Efficient batched training support optimizes throughput for large datasets, while integrated loss functions including MSE and CrossEntropy provide complete training workflows. All components are designed to work seamlessly with the automatic differentiation system, ensuring accurate gradient computation throughout the network.

## Examples

### Convolutional Neural Network
```rust
use ferrox::prelude::*;

// Create CNN for image classification
let mut cnn = CNNClassifier::new(
    3,        // RGB input channels
    10,       // number of classes
    (32, 32), // input image size
    Device::Cuda(0)
)?;

// Forward pass with automatic differentiation
let mut graph = AutoFerroxEngine::<f32>::new();
graph.set_training(true);

let logits = cnn.forward(&mut graph, input_batch)?;
let loss = loss_fn.forward(&mut graph, logits, targets)?;

// Backpropagation
graph.backward(loss)?;

// Optimizer step
optimizer.step(&mut graph)?;
```

### Memory Usage Monitoring
```rust
// Get detailed memory statistics
graph.print_stats();

// For CUDA memory pool
if let Some(cuda_pool) = &backend.cuda_pool {
    cuda_pool.print_stats();
}
```

### Basic Autograd
```rust
use ferrox::prelude::*;

// Create computation graph
let mut graph = AutoFerroxEngine::<f32>::new();
graph.set_training(true);

// Create input with gradient tracking
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

## System Requirements

### Basic Requirements
- **Rust**: Version 1.75.0 or later
- **Operating System**: Linux (primary), Windows/macOS (experimental)
- **Build Tools**: cargo and nvcc for CUDA kernels

### GPU Requirements (Optional)
- **NVIDIA GPU**: Compute Capability 8.0 or higher
- **CUDA Toolkit**: Compatible with your GPU architecture
- **Environment**: CUDA_HOME properly configured

## Building

```bash
# CPU-only build
cargo build --release

# Full GPU acceleration
cargo build --release --features cuda

# Memory optimized build
cargo build --release --features "cuda,jemalloc"
```

## Testing

Ferrox maintains comprehensive test coverage including gradient checking:

```bash
# Run all tests
cargo test

# GPU-accelerated tests
cargo test --features cuda

# Specific modules
cargo test graph::
cargo test ops::
cargo test nn::
```

## Performance

### Memory Pool Statistics Example
```
=== CUDA MEMORY POOL STATISTICS ===
Pool Hits: 1547 (89.2%)
Pool Misses: 187 (10.8%)
Total Memory: 245.7 MB
Peak Memory: 267.3 MB
```

### Training Performance
- **CNN Training**: ~3.2x speedup vs CPU on RTX 3050
- **Memory Efficiency**: 89%+ pool hit rate in typical workloads
- **Gradient Computation**: Full autograd with minimal overhead

## Contributing

This project welcomes contributions that improve the codebase or add educational value. Please ensure all tests pass and include appropriate documentation.

For CUDA kernel development, see `CUDA_DEVELOPMENT.md` for detailed guidelines on optimization and validation.

## Acknowledgments

- Inspired by the CMU Deep Learning Systems course (https://dlsyscourse.org/)
- PyTorch's autograd system for API design (https://pytorch.org/)
- The Rust ML community for guidance and inspiration (https://www.arewelearningyet.com/)

## Author

**Raul Moldes**
- GitHub: [@RaulMoldes](https://github.com/RaulMoldes)
- Email: raul.moldes.work@gmail.com

## License

MIT License - see LICENSE file for details.
