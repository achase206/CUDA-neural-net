from pycuda.compiler import SourceModule
import pycuda.driver as drv


def get_add_bias_kernel():
    cuda_module = SourceModule(
        """
    __global__ void add_bias_and_activations(
        float *preactivations,
        float *biases,
        float *activations,
        int M,
        int N,
        int relu) // 1 for true, 0 for false
        {

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N){
            // flatten 2D idx and apply bias
            int idx = row * N + col; 
            float val = preactivations[idx] + biases[row];

            // apply ReLU if hidden layer
            if (relu == 1){
                if (val > 0.0f){
                    activations[idx] = val; 
                }
                else {
                    activations[idx] = 0.0f; 
                }
            }
            // on output layer, just keep vals as they are
            else {
                activations[idx] = val; 
            }
        }
    }
    """
    )

    return cuda_module.get_function("add_bias_and_activations")


def get_sum_delta_rows_kernel():
    cuda_module = SourceModule(
        """
        __global__ void sum_delta_rows(
            float *delta,
            float *biases_grad,
            int M,
            int N
        ){

            int row = blockIdx.x * blockDim.x + threadIdx.x; 

            if (row < M){
                float sum = 0.0f;
                for (int col = 0; col < N; ++col){
                    sum += delta[row * N + col]; 
                } 
                biases_grad[row] = sum;
            }
        }
        """
    )

    return cuda_module.get_function("sum_delta_rows")


def get_compute_delta_output_kernel():
    cuda_module = SourceModule(
        """
        __global__ void compute_delta_output(
            float *activations,
            float *references,
            float *delta,
            int M,
            int N){

            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < M && col < N){
                int idx = row * N + col;
                delta[idx] = activations[idx] - references[idx]; 
            }
        }
        """
    )

    return cuda_module.get_function("compute_delta_output")


def get_compute_ReLU_deriv_kernel():
    cuda_module = SourceModule(
        """
        __global__ void compute_ReLU_deriv(
            float *temp_delta,
            float *preactivations,
            float *biases,
            float *delta,
            int M,
            int N){

            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < M && col < N){
                int idx = row * N + col; 
                float z = preactivations[idx] + biases[row];
                if (z > 0.0f){
                    delta[idx] = temp_delta[idx];
                } else {
                    delta[idx] = 0.0f; 
                }
            } 
        }
    """
    )

    return cuda_module.get_function("compute_ReLU_deriv")


def get_transpose_matrix_kernel():
    cuda_module = SourceModule(
        """
    #define TILE_SIZE 32
    __global__ void transpose_matrix(float *C_in, float *C_out, int M, int N){
        __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

        int x = blockIdx.x * TILE_SIZE + threadIdx.x;
        int y = blockIdx.y * TILE_SIZE + threadIdx.y;
        
        // Load tile into shared memory from C_in to be transposed
        if (x < N && y < M){
            tile[threadIdx.y][threadIdx.x] = C_in[y * N + x];
        }

        __syncthreads();

        int trans_x = blockIdx.y * TILE_SIZE + threadIdx.x;
        int trans_y = blockIdx.x * TILE_SIZE + threadIdx.y;
        
        // Write transposed tile to C_out
        if (trans_x < M && trans_y < N){
            C_out[trans_y * M + trans_x] = tile[threadIdx.x][threadIdx.y];
        }
    }
    """
    )

    return cuda_module.get_function("transpose_matrix")


def get_apply_weights_biases_kernel():
    cuda_module = SourceModule(
        """
        __global__ void apply_weights_and_biases(
            float *weights,
            float *weights_grad,
            float *biases,
            float *biases_grad,
            int batch_size,
            int M, // Current layer size
            int K, // Previous layer size
            float training_rate){

            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Apply weights and biases with training rate
            if (row < M && col < K){
                int idx = row * K + col;
                float weight_diff = (training_rate / batch_size) * weights_grad[idx];
                weights[idx] = weights[idx] - weight_diff;
                weights_grad[idx] = 0.0f;

                // Biases is 1D vec, only update on first column thread
                if (col == 0){
                    float bias_diff = (training_rate / batch_size) * biases_grad[row];
                    biases[row] = biases[row] - bias_diff;
                    biases_grad[row] = 0.0f;
                }
            }
        }

        """
    )

    return cuda_module.get_function("apply_weights_and_biases")


def get_matrix_multiply_kernel():
    cuda_module = SourceModule(
        """
    // Additional kernels to implement
    // Add bias column vec to preactivation matrix and apply ReLU
    // Compute derivative of the ReLU for the hidden layer delta calc
    // Apply weight/bias gradients using the training rate

    #define TILE_SIZE 32
    // M = rows of A and C
    // N = cols of B and C
    // K = cols of A and rows of B, shared dim
    __global__ void matrix_multiply(float *A, float *B, float *C, int M, int N, int K){
        int itile = blockIdx.y;
        int jtile = blockIdx.x;
        int i = threadIdx.y;
        int j = threadIdx.x;

        // Get row and column of the C element to work on
        int row = itile * TILE_SIZE + i;
        int col = jtile * TILE_SIZE + j;
        
        // Need to fix tile size approach at some point
        __shared__ float A_share[TILE_SIZE][TILE_SIZE];
        __shared__ float B_share[TILE_SIZE][TILE_SIZE];

        // j * jstride + i * istride
        float sum = 0.0;

        // Need remainder guard here for non evenly divisible!!!!
        for (int ktile = 0; ktile < (K + TILE_SIZE - 1) / TILE_SIZE; ++ktile){

            // Remainder guard for A
            if (row < M && (ktile * TILE_SIZE + j) < K){
                A_share[i][j] = A[row * K + (ktile * TILE_SIZE + j)];
            }
            else {
                A_share[i][j] = 0.0; // pad the oob remainder w/ zeros
            }

            // Remainder guard for B
            if (col < N && (ktile * TILE_SIZE + i) < K){
                B_share[i][j] = B[(ktile * TILE_SIZE + i) * N + col];
            }
            else {
                B_share[i][j] = 0.0; // pad the oob remainder w/ zeros
            }
            __syncthreads();
            
            for (int k = 0; k < TILE_SIZE; ++k){
                sum += A_share[i][k] * B_share[k][j];
            } 
            __syncthreads();
        }

        // Guard write to C from oob values
        if (row < M && col < N){
            C[row * N + col] = sum;
        }
    }
    """
    )

    return cuda_module.get_function("matrix_multiply")


if __name__ == "__main__":

    import math
    import time
    import numpy as np

    from pycuda import gpuarray

    # auto init is creating a cuda context automatically
    drv.init()
    device = drv.Device(0)
    context = device.make_context()

    # mat_size = 8192
    M = 128
    K = 64
    N = 200
    tile_size = 32
    blocksize = tile_size

    # grid dimensions
    grid_x = math.ceil(N / tile_size)
    grid_y = math.ceil(M / tile_size)

    nmats = 8

    A_cpu = [np.random.rand(M, K).astype(np.float32) for imat in range(nmats)]
    B_cpu = [np.random.rand(K, N).astype(np.float32) for imat in range(nmats)]
    C_cpu = [np.zeros((M, N), dtype=np.float32) for imat in range(nmats)]

    ref = [A_cpu[imat] @ B_cpu[imat] for imat in range(nmats)]

    # Matrix Multiplication
    mm = get_matrix_multiply_kernel()

    streams = [drv.Stream() for imat in range(nmats)]
    kernel_start = [drv.Event() for imat in range(nmats)]
    kernel_end = [drv.Event() for imat in range(nmats)]

    total_time = 0.0
    start_time = time.time()
    A_gpu = [
        gpuarray.to_gpu_async(A_cpu[imat], stream=streams[imat])
        for imat in range(nmats)
    ]
    B_gpu = [
        gpuarray.to_gpu_async(B_cpu[imat], stream=streams[imat])
        for imat in range(nmats)
    ]
    C_gpu = [
        gpuarray.to_gpu_async(C_cpu[imat], stream=streams[imat])
        for imat in range(nmats)
    ]
    end_time = time.time()
    total_time += end_time - start_time

    for imat in range(nmats):
        start_time = time.time()
        kernel_start[imat].record(streams[imat])
        mm(
            A_gpu[imat],
            B_gpu[imat],
            C_gpu[imat],
            np.int32(M),
            np.int32(N),
            np.int32(K),
            block=(blocksize, blocksize, 1),
            grid=(grid_x, grid_y, 1),
            stream=streams[imat],
        )
        kernel_end[imat].record(streams[imat])
        end_time = time.time()
        total_time += end_time - start_time

    for imat in range(nmats):
        start_time = time.time()
        C_cpu[imat] = C_gpu[imat].get_async(stream=streams[imat])
        end_time = time.time()
        total_time += end_time - start_time
        print(f"error: {float(np.max(np.abs(ref[imat] - C_cpu[imat])))}")
        print(
            f"C[{imat}][0][0]: {C_gpu[imat][0][0]}    Ref[{imat}][0][0]: {ref[imat][0][0]}"
        )
        print(f"mat {imat + 1} time: {end_time-start_time}")

    total_kernel_time = 0.0
    for imat in range(nmats):
        kernel_end[imat].synchronize()
        print(f"Kernel started: {kernel_start[imat].query()}")
        print(f"Kernel ended:   {kernel_end[imat].query()}")
        kernel_time = kernel_start[imat].time_till(kernel_end[imat]) / 1000.0
        print(f"Kernel time:    {kernel_time}")
        total_kernel_time += kernel_time

    print(f"avg kernel time: {total_kernel_time/nmats}")
    print(f"total time: {total_time}")

    context.pop()
