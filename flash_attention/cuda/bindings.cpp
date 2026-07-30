#include <torch/extension.h>
#include <vector>

torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b);
std::vector<torch::Tensor> flash_attention_forward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "Vector add (CUDA)");
    m.def(
        "flash_attention_forward",
        &flash_attention_forward_cuda,
        "Flash attention forward (CUDA) — returns (O, L)"
    );
}
