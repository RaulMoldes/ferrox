// transpose.cu 
#define CUDART_INF_F __int_as_float(0x7f800000)
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)

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

extern "C" __global__ void transpose_2d_f64(
    const double* input,
    double* output,
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