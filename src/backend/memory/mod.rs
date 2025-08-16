// src/backend/memory/mod.rs
// Memory pool abstraction that provides efficient allocation/deallocation
// The pool reduces frequent GPU allocations which are expensive operations
pub mod cuda;

use std::collections::HashMap;

#[cfg(feature = "cuda")]
pub use cuda::CudaMemoryPool;

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    pub pool_hits: usize,   // How many times we reused pooled memory
    pub pool_misses: usize, // How many times we had to allocate new memory
    pub total_memory_bytes: usize,
    pub peak_memory_bytes: usize,
}

#[derive(Debug)]
pub struct PoolBucket<T> {
    pub size_range: (usize, usize),
    pub allocations: Vec<(PoolAllocation<T>, u64)>, // (allocation, last_access_timestamp)
    pub max_allocations: usize, // Maximum number of allocations this bucket can hold
    max_simultaneous_evictions: usize,
    eviction_threshold: u64,
}

impl<T> PoolBucket<T> {
    pub fn new(
        min_size: usize,
        max_size: usize,
        max_allocations: usize,
        max_simultaneous_evictions: usize,
        eviction_threshold: u64,
    ) -> Self {
        Self {
            size_range: (min_size, max_size),
            allocations: Vec::new(),
            max_allocations,
            max_simultaneous_evictions,
            eviction_threshold,
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
        if self.allocations.len() >= self.max_allocations {
            // Drop the allocation (let it go out of scope to free memory)
            //   println!("[WARNING] this bucket is almost full. {} allocations will be freed", self.max_simultaneous_evictions);
            self.evict_lru(1); // EVICT THE OLDEST ALLOCATION
        }

        let timestamp = current_timestamp();

        self.allocations.push((allocation, timestamp));
    }

    // Evict oldest allocations from this bucket that are not in active use
    pub fn evict_lru(&mut self, count_to_evict: usize) -> usize {
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

    pub fn evict_older_than(&mut self, max_age: u64) -> usize {
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

    // Get count of allocations older than specified age
    pub fn count_free_allocations(&self, max_age_ms: u64) -> usize {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.allocations
            .iter()
            .filter(|(_, last_access)| current_time.saturating_sub(*last_access) > max_age_ms)
            .count()
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

    pub fn register_allocation(&mut self, allocation_id: u64, slice: PoolAllocation<T>) {
        self.allocations
            .insert(allocation_id as usize, (slice, current_timestamp()));
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
