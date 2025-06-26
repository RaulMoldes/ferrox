#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use crate::backend::CudaBackend;
    use crate::backend::device::cpu;
    #[cfg(feature = "cuda")]
    use crate::backend::manager::get_backend;

    #[test]
    fn test_device_operations() {
        let device = cpu();

        let zeros = device.zeros::<f64>(&[2, 3]);
        assert_eq!(zeros.shape(), &[2, 3]);
        assert!(zeros.iter().all(|&x| x == 0.0));

        let ones = device.ones::<f64>(&[2, 3]);
        assert_eq!(ones.shape(), &[2, 3]);
        assert!(ones.iter().all(|&x| x == 1.0));

        let full = device.full::<f64>(&[2, 2], 5.0);
        assert_eq!(full.shape(), &[2, 2]);
        assert!(full.iter().all(|&x| x == 5.0));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_backend_manager_initialization() {
        println!("=== Backend Manager Initialization Debug ===");

        // Test direct CUDA backend creation (we know this works)
        println!("1. Testing direct CudaBackend::new(0)...");
        match CudaBackend::new(0) {
            Ok(backend) => {
                println!("  ✓ Direct CudaBackend creation SUCCESS");
                println!("    Device ID: {}", backend.id());
                println!("    Name: {}", backend.name());
            }
            Err(e) => {
                println!("  ✗ Direct CudaBackend creation FAILED: {}", e);
                return; // No point continuing if this fails
            }
        }

        // Test backend manager
        println!("2. Testing backend manager...");
        let backend = get_backend();
        println!("  Backend has CUDA: {}", backend.has_cuda());

        if let Some(cuda_backend) = backend.cuda_backend() {
            println!("  ✓ Backend manager has CUDA backend");
            println!("    Device ID: {}", cuda_backend.id());
        } else {
            println!("  ✗ Backend manager does NOT have CUDA backend");
            println!("  This means BackendManager::init() failed to create CudaBackend");
        }
    }
}
