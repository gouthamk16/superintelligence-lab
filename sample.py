"""Smoke-test CUDA kernels against PyTorch reference."""
import torch
import flash_attn_cuda
from flash_attention.flash_attention import flash_attention_forward, standard_attention

torch.manual_seed(0)
T, d = 64, 32
q = torch.randn(T, d, device="cuda")
k = torch.randn(T, d, device="cuda")
v = torch.randn(T, d, device="cuda")

for causal in [True, False]:
    out_ref, L_ref = flash_attention_forward(
        q.cpu(), k.cpu(), v.cpu(), Br=16, Bc=16, causal=causal
    )
    out_cuda, L_cuda = flash_attn_cuda.flash_attention_forward(q, k, v, causal)

    ok_o = torch.allclose(out_ref.cuda(), out_cuda, atol=1e-3, rtol=1e-3)
    ok_l = torch.allclose(L_ref.cuda(), L_cuda, atol=1e-3, rtol=1e-3)
    ok_std = torch.allclose(
        standard_attention(q.cpu(), k.cpu(), v.cpu(), causal=causal).cuda(),
        out_cuda,
        atol=1e-3,
        rtol=1e-3,
    )
    print(f"causal={causal}  O vs pytorch flash: {ok_o}  L: {ok_l}  O vs standard: {ok_std}")
