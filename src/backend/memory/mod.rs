// src/backend/memory/mod.rs
// Memory pool abstraction that provides efficient allocation/deallocation
// The pool reduces frequent GPU allocations which are expensive operations
#[cfg(feature = "cuda")]
pub mod cuda;

use std::collections::HashMap;

#[cfg(feature = "cuda")]
pub use cuda::CudaMemoryPool;

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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

// Pool allocation result containing both data and metadata for tracking
#[derive(Debug)]
pub struct PoolAllocation<T> {
    pub data: T,
    pub size: usize,
    pub allocation_id: u64,
}

impl<T> PoolAllocation<T> {
    fn new(data: T, size: usize, allocation_id: u64) -> Self {
        Self {
            data,
            size,
            allocation_id,
        }
    }

    pub fn id(&self) -> u64 {
        self.allocation_id
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }

    pub fn into_data(self) -> T {
        self.data
    }
}
// Generic pool trait for different memory backends (CPU/CUDA)
pub trait MemoryPool<T> {
    // Main allocation method - returns pooled memory if available
    fn allocate(&mut self, size: usize) -> Result<PoolAllocation<T>, String>;

    // Return memory to pool for reuse - critical for avoiding memory leaks
    fn deallocate(&mut self, alloc: PoolAllocation<T>) -> Result<(), String>;

    // Pool maintenance - clean up unused allocations periodically
    fn cleanup(&mut self) -> Result<(), String>;

    // Pool statistics for debugging memory usage
    fn stats(&self) -> &PoolStats;

    // Reset pool completely - used for testing or critical cleanup
    fn reset(&mut self) -> Result<(), String>;
}

#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_allocations: usize,
    pub active_allocations: usize,
    pub pool_hits: usize,        // How many times we reused pooled memory
    pub pool_misses: usize,      // How many times we had to allocate new memory
    pub underflow_misses: usize, // When the chunk did not fit in the returned chunk. TODO: reduce this number by implementing block merging.
}

#[derive(Debug, Clone)]
pub struct BucketStats {
    pub total_allocations: usize,
    pub total_evictions: usize,
    pub av_allocations: usize,
    pub max_allocations: usize,
}

impl BucketStats {
    fn new(max_allocations: usize) -> Self {
        println!("Available allocations se inicaliza: {:?}", max_allocations);
        Self {
            total_allocations: 0,
            total_evictions: 0,
            av_allocations: max_allocations,
            max_allocations,
        }
    }
}

#[derive(Debug)]
pub struct PoolBucket<T> {
    pub size_range: (usize, usize),
    pub allocations: Vec<(PoolAllocation<T>, u64)>, // (allocation, last_access_timestamp)
    pub max_allocations: usize, // Maximum number of allocations this bucket can hold
    to_evict: usize, // Number of frames taht will be evicted at the same time when we reach the limit
    eviction_threshold: Option<u64>, // Configurable timestamp: Max time any frame is allowed to be kept in this bucket
    stats: BucketStats,
}

impl<T> PoolBucket<T> {
    pub fn new(
        min_size: usize,
        max_size: usize,
        max_allocations: usize,
        to_evict: usize,
        eviction_threshold: Option<u64>,
    ) -> Self {
        Self {
            size_range: (min_size, max_size),
            allocations: Vec::new(),
            max_allocations,
            to_evict,
            eviction_threshold,
            stats: BucketStats::new(max_allocations),
        }
    }

    // Check if a size fits in this bucket's range
    pub fn fits(&self, size: usize) -> bool {
        size >= self.size_range.0 && size <= self.size_range.1
    }

    // Get available allocation from this bucket
    pub fn get_allocation(&mut self) -> Option<PoolAllocation<T>> {
        self.allocations.pop().map(|(allocation, _)| allocation)
    }

    // Return allocation to this bucket's pool with current timestamp
    pub fn return_allocation(&mut self, allocation: PoolAllocation<T>) {
        // Check if we're at capacity before adding

        let current_allocations = self.allocation_counts();

        if current_allocations >= self.max_allocations {
            // Drop the allocation (let it go out of scope to free memory)
            //   println!("[WARNING] this bucket is almost full. {} allocations will be freed", self.to_evict);
            if let Some(threshold) = self.eviction_threshold {
                self.evict_older_than(threshold); // EVICT ANY ALLOCATIONS OLDER THAN THE SPECIFIED Ts.
            } else {
                self.evict_lru(self.to_evict); // EVICT THE OLDEST ALLOCATIONS
            }
        }

        self.stats.total_evictions += current_allocations - self.allocation_counts();
        self.stats.total_allocations = self.allocation_counts() + 1;

        self.stats.av_allocations = self
            .stats
            .max_allocations
            .saturating_sub(self.allocation_counts() + 1);

        let timestamp = current_timestamp();
        self.allocations.push((allocation, timestamp));
    }

    // Evict oldest allocations from this bucket that are not in active use
    fn evict_lru(&mut self, count_to_evict: usize) -> usize {
        if count_to_evict == 0 || self.allocations.is_empty() {
            return 0;
        }

        let actual_evict = count_to_evict.min(self.allocations.len());

        // Sort by timestamp (oldest first)
        self.allocations.sort_by_key(|(_, timestamp)| *timestamp);

        // Remove the oldest allocations
        self.allocations.drain(0..actual_evict);

        actual_evict
    }

    fn evict_older_than(&mut self, max_age: u64) -> usize {
        if self.allocations.is_empty() {
            return 0;
        }

        let now = current_timestamp();
        let before_len = self.allocations.len();

        // Retener solo las asignaciones recientes
        self.allocations
            .retain(|(_, timestamp)| now.saturating_sub(*timestamp) <= max_age);

        before_len - self.allocations.len()
    }

    // Remove all pooled allocations (not in active use)
    pub fn clear_all(&mut self) -> usize {
        let count = self.allocations.len();
        self.allocations.clear();
        count
    }

    // Get remaining allocation counts
    pub fn allocation_counts(&self) -> usize {
        // Returns (pooled_count, active_count)
        self.allocations.len()
    }

    pub fn stats(&self) -> &BucketStats {
        &self.stats
    }

    pub fn print_stats(&self, id: usize) {
        let stats = self.stats();
        println!("==============================================");
        println!("[INFO] BUCKET {:?}  STATS", id);
        println!(
            "[INFO] Total bucket allocations {}",
            stats.total_allocations
        );
        println!("[INFO] Total bucket evictions {}", stats.total_evictions);
        println!(
            "[INFO] Total available allocations {}",
            stats.av_allocations
        );
        println!(
            "[INFO] Available {} %",
            (stats.av_allocations * 100).div_ceil(stats.max_allocations)
        );
        println!("[INFO] Max allocations {}", stats.max_allocations);
        println!("==============================================");
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
