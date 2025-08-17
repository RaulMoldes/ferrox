// src/backend/memory/cuda_pool.rs
// CUDA memory pool for efficient CudaSlice<T> allocation and reuse
// Uses CudaContextManager instead of raw CudaContext for consistency

#[cfg(feature = "cuda")]
use super::{MemoryPool, PoolAllocation, PoolBucket, PoolStats};
#[cfg(feature = "cuda")]
use crate::FerroxCudaN;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use std::time::{SystemTime, UNIX_EPOCH};

/// POOL and Bucket Configs
#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub min_size: usize,
    pub max_size: usize,
    pub max_allocations: usize,
    pub description: String,
    to_be_evicted: usize,
    eviction_threshold: Option<u64>,
}

impl BucketConfig {
    pub fn new(
        min_size: usize,
        max_size: usize,
        max_allocations: usize,
        description: &str,
        to_be_evicted: usize,
        eviction_threshold: Option<u64>,
    ) -> Self {
        Self {
            min_size,
            max_size,
            max_allocations,
            description: description.to_string(),
            to_be_evicted,
            eviction_threshold,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub bucket_configs: Vec<BucketConfig>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        let bucket_configs = vec![
            BucketConfig::new(1, 1024, 500, "Small GPU tensors: weights, biases", 50, None), // Evict 10% of the tensors
            BucketConfig::new(
                1025,
                262144,
                50,
                "Medium GPU tensors: layer activations",
                15,
                None,
            ),
            BucketConfig::new(
                262145,
                16777216,
                10,
                "Large GPU tensors: batch processing",
                5,
                None,
            ),
            BucketConfig::new(
                16777217,
                usize::MAX,
                2,
                "Huge GPU tensors: full model parameters", // These are very big so we evict them all
                2,
                None,
            ),
        ];

        Self { bucket_configs }
    }
}
impl PoolConfig {
    // Create buckets from this configuration
    pub fn create_buckets<T>(&self) -> Vec<PoolBucket<T>> {
        self.bucket_configs
            .iter()
            .map(|config| {
                PoolBucket::new(
                    config.min_size,
                    config.max_size,
                    config.max_allocations,
                    config.to_be_evicted,
                    config.eviction_threshold,
                )
            })
            .collect()
    }

    // Add a new bucket configuration
    pub fn add_bucket(
        &mut self,
        min_size: usize,
        max_size: usize,
        max_allocations: usize,
        description: &str,
        to_be_evicted: usize,
        eviction_threshold: u64,
    ) {
        self.bucket_configs.push(BucketConfig::new(
            min_size,
            max_size,
            max_allocations,
            description,
            to_be_evicted,
            Some(eviction_threshold),
        ));
    }

    // Print configuration summary
    pub fn print_config(&self) {
        println!("Pool Configuration:");
        for (i, config) in self.bucket_configs.iter().enumerate() {
            println!(
                "  Bucket {}: {}-{} elements, max {} allocations - {}",
                i,
                config.min_size,
                if config.max_size == usize::MAX {
                    "∞".to_string()
                } else {
                    config.max_size.to_string()
                },
                config.max_allocations,
                config.description
            );
        }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaMemoryPool<T: FerroxCudaN> {
    buckets: Vec<PoolBucket<CudaSlice<T>>>,
    stream: Arc<CudaStream>,
    stats: PoolStats,
    active_allocations: HashMap<u64, usize>,
    allocation_counter: usize,
}

#[cfg(feature = "cuda")]
impl<T: FerroxCudaN> CudaMemoryPool<T> {
    pub fn new(stream: Arc<CudaStream>, config: Option<PoolConfig>) -> Self {
        let config = config.unwrap_or_default();
        let buckets = config.create_buckets();

        Self {
            buckets,
            stream,
            stats: PoolStats::default(),
            active_allocations: HashMap::new(),
            allocation_counter: 0,
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
impl<T: FerroxCudaN> MemoryPool<CudaSlice<T>> for CudaMemoryPool<T> {
    fn allocate(&mut self, size: usize) -> Result<PoolAllocation<CudaSlice<T>>, String> {
        if size == 0 {
            return Err("Cannot allocate zero-sized CUDA memory".to_string());
        }

        self.allocation_counter += 1;
        let allocation_id = self.allocation_counter as u64;

        // Find appropriate bucket - panic if no bucket can handle this size
        let bucket_idx = self.find_bucket(size).unwrap_or_else(|| {
            panic!(
                "No bucket configured for allocation size {}. Current buckets: {:?}",
                size,
                self.buckets
                    .iter()
                    .map(|b| b.size_range)
                    .collect::<Vec<_>>()
            )
        });

        self.active_allocations.insert(allocation_id, bucket_idx);

        // Try pool reuse first
        if let Some(pooled_slice) = self.buckets[bucket_idx].get_allocation() {
            // If the slice fits, return the allocation
            if pooled_slice.size >= size {
                self.stats.pool_hits += 1;
                self.stats.active_allocations += 1;

                //println!("Pool hit - reusing allocation. Active allocations");
                return Ok(pooled_slice);
            } else {
                self.stats.underflow_misses += 1;
                // Return undersized slice back to pool
                self.buckets[bucket_idx].return_allocation(pooled_slice);
            }
        }

        // No suitable pooled allocation found - create new one
        let new_slice = self.create_new_allocation(size)?;
        self.active_allocations.insert(allocation_id, bucket_idx);
        self.update_stats();

        // println!("Pool miss - created new allocation");
        Ok(PoolAllocation::new(new_slice, size, allocation_id))
    }

    fn deallocate(&mut self, allocation: PoolAllocation<CudaSlice<T>>) -> Result<(), String> {
        let allocation_id = allocation.id();
        if let Some(bucket_idx) = self.active_allocations.remove(&allocation_id) {
            self.stats.active_allocations -= 1;
            let bucket = &mut self.buckets[bucket_idx];
            bucket.return_allocation(allocation);
        }
        Ok(())
    }

    fn cleanup(&mut self) -> Result<(), String> {
        // Aggressive cleanup for GPU memory conservation
        for bucket in &mut self.buckets {
            bucket.clear_all();
        }
        Ok(())
    }

    fn stats(&self) -> &PoolStats {
        &self.stats
    }

    fn reset(&mut self) -> Result<(), String> {
        for bucket in &mut self.buckets {
            bucket.allocations.clear();
        }
        self.active_allocations.clear();
        self.allocation_counter = 0;
        self.stats = PoolStats::default();
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl<T: FerroxCudaN> CudaMemoryPool<T> {
    pub fn print_stats(&self) {
        println!("\n==========================================");

        // Pool-level statistics
        println!(
            "[INFO] Memory allocation counter: {}",
            self.allocation_counter
        );
        println!(
            "[INFO] Active Allocations: {}",
            self.active_allocations.len()
        );
        println!("[INFO] Total Allocations: {}", self.stats.total_allocations);

        // Hit/miss ratios with percentages
        let total_requests =
            self.stats.pool_hits + self.stats.pool_misses + self.stats.underflow_misses;
        let hit_rate = if total_requests > 0 {
            (self.stats.pool_hits as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "[INFO] Pool Hits: {} ({:.1}%)",
            self.stats.pool_hits, hit_rate
        );
        println!(
            "[INFO] Total pool Misses: {} ({:.1}%)",
            self.stats.pool_misses + self.stats.underflow_misses,
            100.0 - hit_rate
        );

        let underflow_rate = if total_requests > 0 {
            (self.stats.underflow_misses as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "[INFO] Underflow misses: {} ({:.1}%)",
            self.stats.underflow_misses, underflow_rate
        );

        for (idx, bucket) in self.buckets.iter().enumerate() {
            bucket.print_stats(idx);
        }
    }

    // Update statistics when creating new allocation
    fn update_stats(&mut self) {
        self.stats.pool_misses += 1;
        self.stats.total_allocations += 1;
        self.stats.active_allocations += 1;
    }
}

impl<T> Drop for CudaMemoryPool<T>
where
    T: FerroxCudaN,
{
    fn drop(&mut self) {
        self.cleanup().unwrap();
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
