// build.rs
// Automated CUDA kernel compilation script for Ferrox
// This build script automatically compiles CUDA kernels to PTX when building with CUDA feature

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Only compile CUDA kernels when the cuda feature is enabled
    if cfg!(feature = "cuda") {
        println!("cargo:rerun-if-changed=kernels/");
        compile_cuda_kernels();
    }
}

fn compile_cuda_kernels() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let kernels_dir = Path::new(&manifest_dir).join("kernels");

    // Check if nvcc is available
    if !check_nvcc_available() {
        println!("cargo:warning=nvcc not found. CUDA kernels will not be compiled.");
        println!("cargo:warning=Make sure CUDA toolkit is installed and nvcc is in PATH.");
        return;
    }

    // List of kernels to compile
    let kernels = [
        "elementwise",
        "reduction",
        "matmul",
        "activations",
        "comparison", // New kernel
        "convolutions",
        "fill",
    ];

    println!("Compiling CUDA kernels...");

    for kernel in &kernels {
        let cu_file = kernels_dir.join(format!("{kernel}.cu"));
        let ptx_file = kernels_dir.join(format!("{kernel}.ptx"));

        // Skip if .cu file doesn't exist
        if !cu_file.exists() {
            println!(
                "cargo:warning=Kernel source {} not found, skipping",
                cu_file.display()
            );
            continue;
        }

        // Check if we need to recompile (source newer than PTX)
        if ptx_file.exists() {
            let cu_modified = std::fs::metadata(&cu_file)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            let ptx_modified = std::fs::metadata(&ptx_file)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);

            if cu_modified <= ptx_modified {
                println!("✓ {kernel} is up to date");
                continue;
            }
        }

        println!("Compiling {kernel}...");

        // Compile kernel to PTX
        let output = Command::new("nvcc")
            .arg("-ptx")
            .arg("-O3")
            .arg("-ccbin")
            .arg("gcc-11")
            .arg("--std=c++11")
            .arg("-arch=sm_86") // You can make this configurable
            .arg("-I") // Add include directory flag
            .arg(&kernels_dir) // Point to kernels directory for headers
            .arg(&cu_file)
            .arg("-o")
            .arg(&ptx_file)
            .output();

        match output {
            Ok(result) => {
                if !result.status.success() {
                    println!(
                        "cargo:warning=Failed to compile {}: {}",
                        kernel,
                        String::from_utf8_lossy(&result.stderr)
                    );
                    continue;
                }

                // Fix PTX version for compatibility
                fix_ptx_version(&ptx_file);
                println!("✓ Compiled {kernel}");
            }
            Err(e) => {
                println!("cargo:warning=Error compiling {kernel}: {e}");
            }
        }
    }

    println!("CUDA kernel compilation complete!");
}

fn check_nvcc_available() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn fix_ptx_version(ptx_file: &Path) {
    // Read the PTX file and fix the version from 8.5 to 7.5 for compatibility
    if let Ok(content) = std::fs::read_to_string(ptx_file) {
        let fixed_content = content.replace("version 8.5", "version 7.5");
        if let Err(e) = std::fs::write(ptx_file, fixed_content) {
            println!(
                "cargo:warning=Failed to fix PTX version for {}: {}",
                ptx_file.display(),
                e
            );
        }
    }
}
