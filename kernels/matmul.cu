// matmul.cu
#define TILE_SIZE 16

extern "C" __global__ void matmul(
    const float* A,
    const float* B,
    float* C,
    int M,  // rows of A and C
    int N,  // cols of B and C
    int K   // cols of A and rows of B
) {
    // shared memory for tiles
    // TILE_SIZE is the size of the tile, which should be a multiple of the warp size (32)
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile of A into shared memory. Will be reused across threads in the block
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            // Pad with zeros if out of bounds
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int b_row = tile * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            // Pad with zeros if out of bounds
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
