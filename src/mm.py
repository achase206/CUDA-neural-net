import math
import time
import numpy as np

# import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as drv

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

cuda_module = SourceModule(
    """
#define TILE_SIZE 32
// M = rows of A and C
// N = cols of B and C
// K = cols of A and rows of B, shared dim
__global__ void mm(float *A, float *B, float *C, int M, int N, int K){
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

nmats = 8

A_cpu = [np.random.rand(M, K).astype(np.float32) for imat in range(nmats)]
B_cpu = [np.random.rand(K, N).astype(np.float32) for imat in range(nmats)]
C_cpu = [np.zeros((M, N), dtype=np.float32) for imat in range(nmats)]

ref = [A_cpu[imat] @ B_cpu[imat] for imat in range(nmats)]

# Matrix Multiplication
mm = cuda_module.get_function("mm")

streams = [drv.Stream() for imat in range(nmats)]
kernel_start = [drv.Event() for imat in range(nmats)]
kernel_end = [drv.Event() for imat in range(nmats)]

total_time = 0.0
start_time = time.time()
A_gpu = [
    gpuarray.to_gpu_async(A_cpu[imat], stream=streams[imat]) for imat in range(nmats)
]
B_gpu = [
    gpuarray.to_gpu_async(B_cpu[imat], stream=streams[imat]) for imat in range(nmats)
]
C_gpu = [
    gpuarray.to_gpu_async(C_cpu[imat], stream=streams[imat]) for imat in range(nmats)
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
