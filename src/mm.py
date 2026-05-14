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
mat_size = 128
tile_size = 32
ntiles = math.ceil(mat_size / tile_size)
blocksize = tile_size

cuda_module = SourceModule(
    """
#define TILE_SIZE 32
__global__ void mm(float *A, float *B, float *C, int tile_size, int mat_size){
    int itile = blockIdx.y;
    int jtile = blockIdx.x;
    int i = threadIdx.y;
    int j = threadIdx.x;
    // Need to fix tile size approach at some point
    __shared__ float A_share[TILE_SIZE][TILE_SIZE];
    __shared__ float B_share[TILE_SIZE][TILE_SIZE];

    // j * jstride + i * istride
    float sum = 0.0;

    // Need remainder guard here for non evenly divisible!!!!
    for (int ktile = 0; ktile < (mat_size + tile_size - 1) / tile_size; ++ktile){
        A_share[i][j] = A[j + i * mat_size + ktile * tile_size + itile * mat_size * tile_size];
        B_share[i][j] = B[j + i * mat_size + jtile * tile_size + ktile * mat_size * tile_size];
        __syncthreads();
        
        for (int k = 0; k < tile_size; ++k){
            sum += A_share[i][k] * B_share[k][j];
        } 
        __syncthreads();
    }
    C[j + i * mat_size + jtile * tile_size + itile * mat_size * tile_size] = sum;
}
"""
)

nmats = 8

A_cpu = [np.random.rand(mat_size, mat_size).astype(np.float32) for imat in range(nmats)]
B_cpu = [np.random.rand(mat_size, mat_size).astype(np.float32) for imat in range(nmats)]
C_cpu = [np.zeros((mat_size, mat_size), dtype=np.float32) for imat in range(nmats)]

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
        np.int32(tile_size),
        np.int32(mat_size),
        block=(blocksize, blocksize, 1),
        grid=(ntiles, ntiles, 1),
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
    # print(
    #     f"C[{imat}][0][0]: {C_gpu[imat][0][0]}    Ref[{imat}][0][0]: {ref[imat][0][0]}"
    # )
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
