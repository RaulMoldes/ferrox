#!/bin/bash

set -e

VERSION=${1:-"0.0.2"}
DATE=$(date +%Y%m%d)
RELEASE_NAME="ferrox-v${VERSION}-${DATE}"

echo "Building Ferrox release ${RELEASE_NAME}..."

# Clean and build
cargo clean
cargo build --release --features "cuda,jemalloc"
cargo build --release --examples --features "cuda,jemalloc"

# Prepare release directory
rm -rf ferrox-release
mkdir -p ferrox-release/{lib,examples,kernels,src,docs}
mkdir -p ferrox-release/examples/source

# Copy library and examples
cp target/release/libferrox.rlib ferrox-release/lib/
cp target/release/examples/* ferrox-release/examples/ 2>/dev/null || true

# Copy kernels and source
cp kernels/*.ptx ferrox-release/kernels/
cp -r src/* ferrox-release/src/
cp -r examples/* ferrox-release/examples/source/

ls
# Copy documentation
cp Readme.md LICENSE Cargo.toml ferrox-release/
cp CUDA_DEVELOPMENT.md ferrox-release/docs/ 2>/dev/null || true

# Create archive
zip -r "${RELEASE_NAME}.zip" ferrox-release/

echo "Release created: ${RELEASE_NAME}.zip"
echo "Size: $(du -h ${RELEASE_NAME}.zip | cut -f1)"
echo "Contents:"
echo "- Library: $(ls -la ferrox-release/lib/)"
echo "- Examples: $(ls ferrox-release/examples/ | grep -v source | wc -l) executables"
echo "- Kernels: $(ls ferrox-release/kernels/*.ptx | wc -l) PTX files"

rm -rf ferrox-release
