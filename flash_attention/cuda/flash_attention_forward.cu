// Flash Attention forward (single head, float32).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace {

constexpr int kBr = 16;
constexpr int kBc = 16;

__global__ void flash_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,
    int T,
    int d,
    float scale,
    bool causal
) {
    // This block owns query rows [i, i + Br)
    const int i = blockIdx.x * kBr;
    if (i >= T) return;

    const int Br_actual = min(kBr, T - i);
    const int tid = threadIdx.x;

    // Dynamic shared memory layout (all float):
    //   Q_tile[Br * d] | K_tile[Bc * d] | V_tile[Bc * d] | S[Br * Bc]
    //   | m[Br] | l[Br] | o[Br * d]
    extern __shared__ float smem[];
    float* Q_tile = smem;
    float* K_tile = Q_tile + kBr * d;
    float* V_tile = K_tile + kBc * d;
    float* S = V_tile + kBc * d;
    float* m = S + kBr * kBc;
    float* l = m + kBr;
    float* o = l + kBr;

    // Load Q tile from HBM to SRAM
    for (int t = tid; t < Br_actual * d; t += blockDim.x) {
        int row = t / d;
        int col = t % d;
        Q_tile[t] = Q[(i + row) * d + col];
    }
    // Initialize online-softmax state for this query tile
    for (int r = tid; r < Br_actual; r += blockDim.x) {
        m[r] = -INFINITY;
        l[r] = 0.f;
    }
    for (int t = tid; t < Br_actual * d; t += blockDim.x) {
        o[t] = 0.f;
    }
    __syncthreads();

    // Sweep key/value tiles
    for (int j = 0; j < T; j += kBc) {
        const int Bc_actual = min(kBc, T - j);

        // Load K and V tiles
        for (int t = tid; t < Bc_actual * d; t += blockDim.x) {
            int row = t / d;
            int col = t % d;
            K_tile[t] = K[(j + row) * d + col];
            V_tile[t] = V[(j + row) * d + col];
        }
        __syncthreads();

        // Each thread handles one query row: S_row = Q_row @ K_tile^T * scale, where scale is 1/sqrt(d)
        if (tid < Br_actual) {
            const int r = tid;
            const int global_row = i + r;

            for (int c = 0; c < Bc_actual; c++) {
                float dot = 0.f;
                for (int col = 0; col < d; col++) {
                    dot += Q_tile[r * d + col] * K_tile[c * d + col];
                }
                float s = dot * scale;
                if (causal && (j + c) > global_row) {
                    s = -INFINITY;
                }
                S[r * kBc + c] = s;
            }

            // Online softmax update for this row
            float row_max = -INFINITY;
            for (int c = 0; c < Bc_actual; c++) {
                row_max = fmaxf(row_max, S[r * kBc + c]);
            }
            // Fully masked tile for this row, no update
            if (row_max != -INFINITY) {
                float m_old = m[r];
                float m_new = fmaxf(m_old, row_max);
                float l_old = l[r];
                float l_new = expf(m_old - m_new) * l_old;
                for (int c = 0; c < Bc_actual; c++) {
                    l_new += expf(S[r * kBc + c] - m_new);
                }

                for (int col = 0; col < d; col++) {
                    float o_val = o[r * d + col] * (l_old * expf(m_old - m_new) / l_new);
                    for (int c = 0; c < Bc_actual; c++) {
                        float p = expf(S[r * kBc + c] - m_new) / l_new;
                        o_val += p * V_tile[c * d + col];
                    }
                    o[r * d + col] = o_val;
                }
                m[r] = m_new;
                l[r] = l_new;
            }
        }
        __syncthreads();
    }

    // Write O and L back to HBM
    for (int t = tid; t < Br_actual * d; t += blockDim.x) {
        int row = t / d;
        int col = t % d;
        O[(i + row) * d + col] = o[t];
    }
    if (L != nullptr) {
        for (int r = tid; r < Br_actual; r += blockDim.x) {
            L[i + r] = m[r] + logf(l[r]);
        }
    }
}

} 

// Host function: check inputs, allocate output tensors, launch kernel
std::vector<torch::Tensor> flash_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "expected (T, d)");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q, k, v shape mismatch");

    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();

    const int T = q.size(0);
    const int d = q.size(1);

    auto O = torch::empty_like(q);
    auto L = torch::empty({T}, q.options());

    const int threads = 128;
    const int blocks = (T + kBr - 1) / kBr;

    // Shared memory: Q + K + V + S + m + l + o
    const size_t smem_bytes =
        (kBr * d + kBc * d + kBc * d + kBr * kBc + kBr + kBr + kBr * d) * sizeof(float);

    flash_fwd_kernel<<<blocks, threads, smem_bytes>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        O.data_ptr<float>(),
        L.data_ptr<float>(),
        T,
        d,
        1.f / sqrtf(static_cast<float>(d)),
        causal
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "flash_fwd_kernel launch failed: ", cudaGetErrorString(err));
    return {O, L};
}
