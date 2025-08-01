// src/backend/memory/mod.rs
// Memory pool abstraction that provides efficient allocation/deallocation
// The pool reduces frequent GPU allocations which are expensive operations
#[cfg(feature = "cuda")]
pub mod cuda;

use std::collections::HashMap;

#[cfg(feature = "cuda")]
pub use cuda::CudaMemoryPool;

// Pool allocation result containing both data and metadata for tracking
#[derive(Debug)]
pub struct PoolAllocation<T> {
    pub data: T,
    pub size: usize,
    pub allocation_id: u64, // Used to track allocations for deallocation
}

// Generic pool trait for different memory backends (CPU/CUDA)
pub trait MemoryPool<T> {
    // Main allocation method - returns pooled memory if available
    fn allocate(&mut self, size: usize) -> Result<PoolAllocation<T>, String>;

    // Return memory to pool for reuse - critical for avoiding memory leaks
    fn deallocate(&mut self, allocation_id: u64) -> Result<(), String>;

    // Pool maintenance - clean up unused allocations periodically
    fn cleanup(&mut self) -> Result<(), String>;

    // Pool statistics for debugging memory usage
    fn stats(&self) -> PoolStats;

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


// Pool bucket system - groups allocations by size ranges for efficiency
// This prevents fragmentation by keeping similar-sized allocations together
#[derive(Debug)]
pub struct PoolBucket<T> {
    pub size_range: (usize, usize), // Min and max sizes for this bucket
    pub allocations: Vec<T>,        // Available pooled allocations
    pub in_use: HashMap<u64, T>,    // Currently borrowed allocations
}

impl<T> PoolBucket<T> {
    pub fn new(min_size: usize, max_size: usize) -> Self {
        Self {
            size_range: (min_size, max_size),
            allocations: Vec::new(),
            in_use: HashMap::new(),
        }
    }

    // Check if a size fits in this bucket's range
    pub fn fits(&self, size: usize) -> bool {
        size >= self.size_range.0 && size <= self.size_range.1
    }

    // Get available allocation from this bucket
    pub fn get_allocation(&mut self) -> Option<T> {
        self.allocations.pop()
    }

    // Return allocation to this bucket's pool
    pub fn return_allocation(&mut self, allocation: T) {
        self.allocations.push(allocation);
    }
}
