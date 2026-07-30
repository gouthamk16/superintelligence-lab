#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    // checks — same shape, CUDA, float32, contiguous
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    a = a.contiguous();
    b = b.contiguous();

    // allocate c = empty_like(a)
    auto c = torch::empty_like(a);

    // launch kernel
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
    
    return c;
}