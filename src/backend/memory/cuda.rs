// src/backend/memory/cuda_pool.rs
// CUDA memory pool for efficient CudaSlice<T> allocation and reuse
// Uses CudaContextManager instead of raw CudaContext for consistency

#[cfg(feature = "cuda")]
use super::{MemoryPool, PoolAllocation, PoolBucket, PoolStats};
#[cfg(feature = "cuda")]
use crate::backend::FerroxCudaF;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
pub struct CudaMemoryPool<T: FerroxCudaF> {
    buckets: Vec<PoolBucket<CudaSlice<T>>>,
    allocation_counter: u64,
    stream: Arc<CudaStream>,
    stats: PoolStats,
    active_allocations: HashMap<u64, usize>, // allocation_id -> bucket_index
    max_pool_size_per_bucket: usize,
}

#[cfg(feature = "cuda")]
impl<T: FerroxCudaF> CudaMemoryPool<T> {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        // GPU memory is more expensive so use smaller buckets and be more conservative
        let buckets = vec![
            PoolBucket::new(1, 1024),              // Small GPU tensors: weights, biases
            PoolBucket::new(1025, 262144),         // Medium GPU tensors: layer activations
            PoolBucket::new(262145, 16777216),     // Large GPU tensors: batch processing
            PoolBucket::new(16777217, usize::MAX), // Huge GPU tensors: full model parameters
        ];

        Self {
            buckets,
            allocation_counter: 0,
            stream,
            stats: PoolStats::default(),
            active_allocations: HashMap::new(),
            max_pool_size_per_bucket: 20, // Lower limit for GPU memory conservation
        }
    }

    fn find_bucket(&self, size: usize) -> Option<usize> {
        self.buckets.iter().position(|bucket| bucket.fits(size))
    }

    fn create_new_allocation(&self, size: usize) -> Result<CudaSlice<T>, String> {
        self.stream
            .alloc_zeros(size)
            .map_err(|e| format!("Failed to allocate GPU memory: {}", e))
    }
}

#[cfg(feature = "cuda")]
impl<T: FerroxCudaF> MemoryPool<CudaSlice<T>> for CudaMemoryPool<T> {
    fn allocate(&mut self, size: usize) -> Result<PoolAllocation<CudaSlice<T>>, String> {

         if size == 0 {
            return Err("Cannot allocate zero-sized CUDA memory".to_string());
        }

        self.allocation_counter += 1;
        let allocation_id = self.allocation_counter;

        // Try pool reuse first
        if let Some(bucket_idx) = self.find_bucket(size) {
            self.active_allocations.insert(allocation_id, bucket_idx);

            if let Some(pooled_slice) = self.buckets[bucket_idx].get_allocation() {
                if pooled_slice.len() >= size {
                    self.stats.pool_hits += 1;
                    self.stats.active_allocations += 1;

                    return Ok(PoolAllocation {
                        data: pooled_slice,
                        size,
                        allocation_id,
                    });
                } else {
                    self.buckets[bucket_idx].return_allocation(pooled_slice);
                }
            }
        }

        // Create new allocation using specified stream
        let new_slice = self.create_new_allocation(size)?;

        self.update_stats_for_new_allocation(size);

        Ok(PoolAllocation {
            data: new_slice,
            size,
            allocation_id,
        })
    }

    fn deallocate(&mut self, allocation_id: u64) -> Result<(), String> {
        self.deallocate_and_return(allocation_id, None)
    }

    fn cleanup(&mut self) -> Result<(), String> {
        // Aggressive cleanup for GPU memory conservation
        for bucket in &mut self.buckets {
            let target_size = self.max_pool_size_per_bucket / 3; // Keep only 1/3
            if bucket.allocations.len() > target_size {
                bucket.allocations.truncate(target_size);
            }
        }
        Ok(())
    }

    fn stats(&self) -> PoolStats {
        self.stats.clone()
    }

    fn reset(&mut self) -> Result<(), String> {
        for bucket in &mut self.buckets {
            bucket.allocations.clear();
            bucket.in_use.clear();
        }
        self.active_allocations.clear();
        self.allocation_counter = 0;
        self.stats = PoolStats::default();
        Ok(())
    }
}


#[cfg(feature = "cuda")]
impl<T: FerroxCudaF> CudaMemoryPool<T> {
    // Common allocation logic
    // Try to get slice from pool. If not, get the slice.
    fn allocate_with_stream(
        &mut self,
        size: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<PoolAllocation<CudaSlice<T>>, String> {
        if size == 0 {
            return Err("Cannot allocate zero-sized CUDA memory".to_string());
        }

        self.allocation_counter += 1;
        let allocation_id = self.allocation_counter;

        // Try pool reuse first
        if let Some(bucket_idx) = self.find_bucket(size) {
            self.active_allocations.insert(allocation_id, bucket_idx);

            if let Some(pooled_slice) = self.buckets[bucket_idx].get_allocation() {
                if pooled_slice.len() >= size {
                    self.stats.pool_hits += 1;
                    self.stats.active_allocations += 1;

                    return Ok(PoolAllocation {
                        data: pooled_slice,
                        size,
                        allocation_id,
                    });
                } else {
                    self.buckets[bucket_idx].return_allocation(pooled_slice);
                }
            }
        }

        // Create new allocation using specified stream
        let new_slice = stream
            .alloc_zeros(size)
            .map_err(|e| format!("Failed to allocate GPU memory: {}", e))?;

        self.update_stats_for_new_allocation(size);

        Ok(PoolAllocation {
            data: new_slice,
            size,
            allocation_id,
        })
    }

    // Update statistics when creating new allocation
    fn update_stats_for_new_allocation(&mut self, size: usize) {
        self.stats.pool_misses += 1;
        self.stats.total_allocations += 1;
        self.stats.active_allocations += 1;

        let memory_bytes = size * std::mem::size_of::<T>();
        self.stats.total_memory_bytes += memory_bytes;

        if self.stats.total_memory_bytes > self.stats.peak_memory_bytes {
            self.stats.peak_memory_bytes = self.stats.total_memory_bytes;
        }
    }

    // Async allocation using provided stream
    pub fn allocate_async(
        &mut self,
        size: usize,
        stream: Arc<CudaStream>
    ) -> Result<PoolAllocation<CudaSlice<T>>, String> {
        self.allocate_with_stream(size, &stream)
    }

    // Unified deallocation - handles both deallocate and return_to_pool logic
    pub fn deallocate_and_return(
        &mut self,
        allocation_id: u64,
        slice: Option<CudaSlice<T>>, // None for deallocate-only, Some(slice) for return_to_pool
    ) -> Result<(), String> {
        if let Some(bucket_idx) = self.active_allocations.remove(&allocation_id) {
            self.stats.active_allocations -= 1;

            // If slice provided, try to return it to pool for reuse
            if let Some(cuda_slice) = slice {
                let bucket = &mut self.buckets[bucket_idx];

                // Only keep in pool if under limit - GPU memory conservation
                if bucket.allocations.len() < self.max_pool_size_per_bucket {
                    bucket.return_allocation(cuda_slice);
                }
                // If bucket full, let CudaSlice drop (GPU memory freed automatically)
            }

            Ok(())
        } else {
            Err(format!("Invalid CUDA allocation ID: {}", allocation_id))
        }
    }
}


#[cfg(feature = "cuda")]
impl<T: FerroxCudaF> CudaMemoryPool<T> {
    pub fn return_to_pool(
        &mut self,
        allocation_id: u64,
        slice: CudaSlice<T>,
    ) -> Result<(), String> {
        // Use unified method with slice - deallocates and returns to pool
        self.deallocate_and_return(allocation_id, Some(slice))
    }
}


// Non-CUDA stub implementation
#[cfg(not(feature = "cuda"))]
pub struct CudaMemoryPool<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(not(feature = "cuda"))]
impl<T> CudaMemoryPool<T> {
    pub fn new(_context_manager: ()) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}
