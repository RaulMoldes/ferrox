// src/backend/cuda/stream_manager.rs
// Stream management helper - does NOT own the CUDA context
use crate::FerroxCudaF;
use crate::backend::manager::alloc_cpu_vec;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Helper for managing named CUDA streams - used by CudaContextManager
pub struct StreamManager {
    default_stream: Arc<CudaStream>,
    streams: Arc<Mutex<HashMap<String, Arc<CudaStream>>>>,
    stream_states: Arc<Mutex<HashMap<String, bool>>>, // Track if stream is ready
}

impl StreamManager {
    /// Create new empty stream manager
    pub fn new(ctx: &Arc<CudaContext>) -> Self {
        let default_stream = ctx.default_stream();
        Self {
            default_stream,
            streams: Arc::new(Mutex::new(HashMap::new())),
            stream_states: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.default_stream.clone()
    }

    /// Create or get a named stream using provided context
    pub fn create_stream(&self, ctx: &Arc<CudaContext>, name: &str) -> Result<(), String> {
        let mut streams = self.streams.lock().unwrap();
        let mut states = self.stream_states.lock().unwrap();

        if !streams.contains_key(name) {
            let stream = ctx
                .new_stream()
                .map_err(|e| format!("Failed to create stream '{}': {}", name, e))?;
            streams.insert(name.to_string(), stream);
            states.insert(name.to_string(), true); // New streams start as ready
        }
        Ok(())
    }

    /// Get stream reference for kernel launches
    pub fn get_stream(&self, stream_name: &str) -> Option<Arc<CudaStream>> {
        self.streams.lock().unwrap().get(stream_name).cloned()
    }

    /// Check if stream exists and is ready (completed all operations)
    pub fn is_stream_ready(&self, stream_name: &str) -> Result<bool, String> {
        let states = self.stream_states.lock().unwrap();
        states
            .get(stream_name)
            .copied()
            .ok_or_else(|| format!("Stream '{}' not found", stream_name))
    }

    /// Synchronize a specific stream (blocking)
    pub fn sync_stream(&self, stream_name: &str) -> Result<(), String> {
        let streams = self.streams.lock().unwrap();
        let stream = streams
            .get(stream_name)
            .ok_or_else(|| format!("Stream '{}' not found", stream_name))?;

        stream
            .synchronize()
            .map_err(|e| format!("Failed to sync stream '{}': {}", stream_name, e))?;

        // Mark stream as ready after synchronization
        let mut states = self.stream_states.lock().unwrap();
        states.insert(stream_name.to_string(), true);

        Ok(())
    }

    /// Synchronize all managed streams
    pub fn sync_all_streams(&self) -> Result<(), String> {
        let streams = self.streams.lock().unwrap();
        for (name, stream) in streams.iter() {
            stream
                .synchronize()
                .map_err(|e| format!("Failed to sync stream '{}': {}", name, e))?;
        }
        Ok(())
    }

    /// Get names of all managed streams
    pub fn stream_names(&self) -> Vec<String> {
        let streams = self.streams.lock().unwrap();
        streams.keys().cloned().collect()
    }

    /// Setup parallel streams commonly used in deep learning
    pub fn setup_parallel_streams(&self, ctx: &Arc<CudaContext>) -> Result<(), String> {
        self.create_stream(ctx, "copy_h2d")?; // Host to device transfers
        self.create_stream(ctx, "copy_d2h")?; // Device to host transfers
        self.create_stream(ctx, "compute")?; // Kernel execution
        self.create_stream(ctx, "memset")?; // Memory operations
        Ok(())
    }

    /// Async host to device transfer using named stream with provided context
    pub fn host_to_device_async<T>(
        &self,
        ctx: &Arc<CudaContext>,
        data: &[T], // Changed from Vec<T> to &[T] for efficiency
        stream_name: Option<&str>,
    ) -> Result<CudaSlice<T>, String>
    where
        T: FerroxCudaF,
    {
        let stream = match stream_name {
            Some(name) => {
                let streams = self.streams.lock().unwrap();
                streams.get(name).cloned().ok_or_else(|| {
                    format!(
                        "Stream '{}' not found. Create it first with create_stream()",
                        name
                    )
                })?
            }
            None => ctx.default_stream(),
        };

        // Allocate GPU memory first
        let mut device_buffer = stream
            .alloc_zeros(data.len())
            .map_err(|e| format!("Failed to allocate GPU memory: {}", e))?;

        // Copy data from host to device using the correct cudarc API

        stream
            .memcpy_htod(data, &mut device_buffer) // data is now &[T]
            .map_err(|e| format!("Async host to device transfer failed: {}", e))?;

        Ok(device_buffer)
    }

    /// Async device to host transfer using named stream
    pub fn device_to_host_async<T>(
        &self,
        ctx: &Arc<CudaContext>,
        data: &CudaSlice<T>,
        stream_name: Option<&str>,
    ) -> Result<Vec<T>, String>
    where
        T: FerroxCudaF,
    {
        let stream = match stream_name {
            Some(name) => {
                let streams = self.streams.lock().unwrap();
                streams.get(name).cloned().ok_or_else(|| {
                    format!(
                        "Stream '{}' not found. Create it first with create_stream()",
                        name
                    )
                })?
            }
            None => ctx.default_stream(),
        };

        // Allocate host buffer
        let mut alloc_result = alloc_cpu_vec::<T>(data.len())?;
        let mut host_buffer = alloc_result.data;

        // Copy data from device to host using the correct cudarc API
        stream
            .memcpy_dtoh(data, &mut host_buffer)
            .map_err(|e| format!("Async device to host transfer failed: {}", e))?;

        Ok(host_buffer)
    }

    /// Get number of managed streams
    pub fn stream_count(&self) -> usize {
        self.streams.lock().unwrap().len()
    }

    /// Remove a stream by name
    pub fn remove_stream(&self, name: &str) -> Result<(), String> {
        let mut streams = self.streams.lock().unwrap();
        let mut states = self.stream_states.lock().unwrap();

        if streams.remove(name).is_some() {
            states.remove(name);
            Ok(())
        } else {
            Err(format!("Stream '{}' not found", name))
        }
    }

    /// Clear all streams
    pub fn clear_streams(&self) {
        self.streams.lock().unwrap().clear();
        self.stream_states.lock().unwrap().clear();
    }
}
