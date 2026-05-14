import torch
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import numpy as np

# context is shared between pytorch and cuda
drv.init()
device = drv.Device(torch.cuda.current_device())
context = device.retain_primary_context()
context.push()

cuda_module = SourceModule(
    """
__global__ void test_kernel(float *x, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n){
        x[index] = 1.3 * index;
    }
}
"""
)

test_kernel = cuda_module.get_function("test_kernel")

# some tensor of size n
n = 1000000
x = torch.randn(n, device="cuda", dtype=torch.float32)

block_size = 1024
test_kernel(
    x,
    np.int32(n),
    block=(block_size, 1, 1),
    grid=((n + block_size - 1) // block_size, 1, 1),
)

print(f"x[0]:    {x[0]}")
print(f"x[100]:  {x[100]}")

context.pop()
