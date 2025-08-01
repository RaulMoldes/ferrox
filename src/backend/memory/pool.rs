// src/backend/memory/pool.rs
use std::collections::BTreeMap;

/// Memory block representation - stores offset and size within the pool
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub offset: usize,
    pub size: usize,
}

/// Core memory pool trait - defines the interface for both CUDA and CPU pools
pub trait MemoryPool<T> {
    /// Allocate a block of memory from the pool
    fn allocate(&self, size: usize) -> Result<MemoryBlock, String>;

    /// Release a previously allocated block back to the pool for reuse
    fn deallocate(&self, block: MemoryBlock) -> Result<(), String>;

    /// Get total pool size in elements
    fn total_size(&self) -> usize;

    /// Get available (free) space in elements
    fn available_size(&self) -> usize;

    /// Get pool utilization as percentage (0.0 to 1.0)
    fn utilization(&self) -> f32 {
        let total = self.total_size() as f32;
        if total == 0.0 { return 0.0; }
        1.0 - (self.available_size() as f32 / total)
    }
}

/// Free block manager - tracks available memory blocks using size-ordered map
/// Uses BTreeMap for efficient best-fit allocation strategy
#[derive(Debug)]
pub struct FreeBlockManager {
    // Map from size to list of available blocks of that size
    // Using BTreeMap allows efficient range queries for best-fit allocation
    free_blocks: BTreeMap<usize, Vec<MemoryBlock>>,
    total_free: usize,
}

impl FreeBlockManager {
    pub fn new() -> Self {
        Self {
            free_blocks: BTreeMap::new(),
            total_free: 0,
        }
    }

    /// Initialize with a single large block covering the entire pool
    pub fn init_with_full_block(size: usize) -> Self {
        let mut manager = Self::new();
        let block = MemoryBlock { offset: 0, size };
        manager.add_free_block(block);
        manager
    }

    /// Add a block to the free list (used during deallocation)
    /// Merges adjacent blocks to reduce fragmentation
    pub fn add_free_block(&mut self, block: MemoryBlock) {
        // Try to merge with adjacent blocks to reduce fragmentation
        let merged_block = self.try_merge_adjacent(block);

        self.free_blocks
            .entry(merged_block.size)
            .or_insert_with(Vec::new)
            .push(merged_block);

        self.total_free += merged_block.size;
    }

    /// Find and remove a suitable free block (best-fit strategy)
    /// Returns None if no suitable block is available
    pub fn find_free_block(&mut self, required_size: usize) -> Option<MemoryBlock> {
        // Find the smallest block that can fit the required size
        let suitable_size = self.free_blocks
            .range(required_size..)
            .next()
            .map(|(&size, _)| size)?;

        let blocks = self.free_blocks.get_mut(&suitable_size)?;
        let block = blocks.pop()?;

        // Remove empty size entry to keep map clean
        if blocks.is_empty() {
            self.free_blocks.remove(&suitable_size);
        }

        self.total_free -= block.size;

        // If block is larger than needed, split it and add remainder back to free list
        if block.size > required_size {
            let remainder = MemoryBlock {
                offset: block.offset + required_size,
                size: block.size - required_size,
            };
            self.add_free_block(remainder);

            Some(MemoryBlock {
                offset: block.offset,
                size: required_size,
            })
        } else {
            Some(block)
        }
    }

    pub fn total_free_size(&self) -> usize {
        self.total_free
    }

    /// Attempt to merge the given block with adjacent free blocks
    /// This is crucial for preventing memory fragmentation
    fn try_merge_adjacent(&mut self, block: MemoryBlock) -> MemoryBlock {
        let mut merged = block;
        let mut found_merge = true;

        // Keep trying to merge until no more adjacent blocks are found
        while found_merge {
            found_merge = false;

            // Look for blocks that can be merged
            for (&size, blocks) in self.free_blocks.iter_mut() {
                if let Some(pos) = blocks.iter().position(|b| {
                    // Check if blocks are adjacent
                    (b.offset + b.size == merged.offset) || (merged.offset + merged.size == b.offset)
                }) {
                    let adjacent = blocks.remove(pos);
                    self.total_free -= adjacent.size;

                    // Merge the blocks
                    merged = MemoryBlock {
                        offset: merged.offset.min(adjacent.offset),
                        size: merged.size + adjacent.size,
                    };

                    found_merge = true;
                    break;
                }
            }
        }

        // Clean up empty entries
        self.free_blocks.retain(|_, blocks| !blocks.is_empty());
        merged
    }
}
