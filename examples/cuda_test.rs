// examples/cuda_test.rs
#[cfg(feature = "cuda")]
use ferrox::backend::{Device, cuda};


#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test CUDA device creation
    let cuda_device = cuda(0);
    println!("Created CUDA device: {:?}", cuda_device);
    
    // Test basic tensor operations on CUDA
    let x = cuda_device.zeros::<f32>(&[1000, 1000]);
    let y = cuda_device.ones::<f32>(&[1000, 1000]);
    
    println!("CUDA tensors created successfully");
    println!("X shape: {:?}", x.shape());
    println!("Y shape: {:?}", y.shape());
    
    // Test if device detection works
    println!("Is CUDA device: {}", cuda_device.is_cuda());
    
    Ok(())
}

// If CUDA feature is not enabled, provide a message
#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature is not enabled. Compile with --features cuda to enable.");
}