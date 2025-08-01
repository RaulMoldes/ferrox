// src/backend/memory/cpu_pool.rs
// CPU memory pool for efficient Vec<T> allocation and reuse
// Vectors can be easily converted to ArrayD with any compatible shape

use super::{MemoryPool, PoolAllocation, PoolBucket, PoolStats};
use crate::backend::{FerroxCudaF, FerroxF};
use std::collections::HashMap;

pub struct CpuMemoryPool<T: FerroxCudaF> {
    buckets: Vec<PoolBucket<Vec<T>>>,
    allocation_counter: u64,
    stats: PoolStats,
    active_allocations: HashMap<u64, usize>,
    max_pool_size_per_bucket: usize,
}

impl<T: FerroxCudaF> CpuMemoryPool<T> {
    pub fn new() -> Self {
        // Create size buckets for different vector sizes
        let buckets = vec![
            PoolBucket::new(1, 1024),          // Small vectors: scalars, small tensors
            PoolBucket::new(1025, 102400),     // Medium vectors: layer outputs
            PoolBucket::new(102401, 10485760), // Large vectors: batch operations
            PoolBucket::new(10485761, usize::MAX), // Huge vectors: full datasets
        ];

        Self {
            buckets,
            allocation_counter: 0,
            stats: PoolStats::default(),
            active_allocations: HashMap::new(),
            max_pool_size_per_bucket: 50,
        }
    }

    fn find_bucket(&self, size: usize) -> Option<usize> {
        self.buckets.iter().position(|bucket| bucket.fits(size))
    }

    // Create new vector with exact size and zero-initialized values
    fn create_new_allocation(&self, size: usize) -> Vec<T> {
        vec![<T as FerroxF>::zero(); size]
    }

    // Resize existing vector to requested size - reuses capacity when possible
    fn resize_pooled_vector(&self, mut vec: Vec<T>, size: usize) -> Vec<T> {
        vec.clear(); // Clear existing data but keep capacity
        vec.resize(size, <T as FerroxF>::zero()); // Resize to new size with zeros
        vec
    }
}

impl<T: FerroxCudaF> MemoryPool<Vec<T>> for CpuMemoryPool<T> {
    fn allocate(&mut self, size: usize) -> Result<PoolAllocation<Vec<T>>, String> {
        if size == 0 {
            return Err("Cannot allocate zero-sized vector".to_string());
        }

        self.allocation_counter += 1;
        let allocation_id = self.allocation_counter;

        // Find appropriate bucket for this size
        if let Some(bucket_idx) = self.find_bucket(size) {
            self.active_allocations.insert(allocation_id, bucket_idx);

            // Try to reuse existing Vec from pool
            if let Some(pooled_vec) = self.buckets[bucket_idx].get_allocation() {
                // Resize to exact requested size (reuses capacity when possible)
                let resized_vec = self.resize_pooled_vector(pooled_vec, size);

                self.stats.pool_hits += 1;
                self.stats.active_allocations += 1;

                return Ok(PoolAllocation {
                    data: resized_vec,
                    size,
                    allocation_id,
                });
            }
        }

        // No suitable pooled allocation - create new Vec
        let new_vec = self.create_new_allocation(size);

        self.stats.pool_misses += 1;
        self.stats.total_allocations += 1;
        self.stats.active_allocations += 1;

        // Memory accounting
        let memory_bytes = size * std::mem::size_of::<T>();
        self.stats.total_memory_bytes += memory_bytes;

        if self.stats.total_memory_bytes > self.stats.peak_memory_bytes {
            self.stats.peak_memory_bytes = self.stats.total_memory_bytes;
        }

        Ok(PoolAllocation {
            data: new_vec,
            size,
            allocation_id,
        })
    }

    fn deallocate(&mut self, allocation_id: u64) -> Result<(), String> {
        if let Some(_bucket_idx) = self.active_allocations.remove(&allocation_id) {
            self.stats.active_allocations -= 1;
            Ok(())
        } else {
            Err(format!("Invalid CPU allocation ID: {}", allocation_id))
        }
    }

    fn cleanup(&mut self) -> Result<(), String> {
        // Remove excess allocations from each bucket to control memory usage
        for bucket in &mut self.buckets {
            let target_size = self.max_pool_size_per_bucket / 2;
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

impl<T: FerroxCudaF> CpuMemoryPool<T> {
    // Return vector to pool for reuse
    pub fn return_to_pool(&mut self, allocation_id: u64, vec: Vec<T>) -> Result<(), String> {
        if let Some(bucket_idx) = self.active_allocations.get(&allocation_id) {
            let bucket = &mut self.buckets[*bucket_idx];

            // Only keep in pool if we haven't exceeded the bucket limit
            if bucket.allocations.len() < self.max_pool_size_per_bucket {
                bucket.return_allocation(vec);
            }
            // If bucket is full, let vector drop naturally (memory freed)

            Ok(())
        } else {
            Err(format!(
                "Cannot return CPU allocation ID: {}",
                allocation_id
            ))
        }
    }

    // CPU-specific allocation method for convenience
    pub fn allocate_vec(&mut self, size: usize) -> Result<PoolAllocation<Vec<T>>, String> {
        self.allocate(size)
    }
}
