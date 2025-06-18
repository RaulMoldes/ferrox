// kernels/transpose.cu - 2D Matrix transpose kernel
extern "C" __global__ void transpose_2d(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int input_idx = row * cols + col;      // input[row][col]
        int output_idx = col * rows + row;     // output[col][row]
        output[output_idx] = input[input_idx];
    }
}