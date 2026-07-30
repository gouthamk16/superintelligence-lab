import math
import torch

def standard_attention(q, k, v, causal=True):
    """
    q, k, v: (T, d)
    Returns: (T, d)
    """
    T, d = q.shape
    scale = 1.0 / math.sqrt(d)
    s = (q @ k.T) * scale

    if causal:
        mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool)) # Mask out the upper triangle of the scores
        s = s.masked_fill(~mask, float('-inf'))

    p = torch.softmax(s , dim=-1)
    return p @ v

def flash_attention_forward(q, k, v, Br=32, Bc=32,causal=True):
    """
    q, k, v: (T, d)
    Br, Bc: block size
    Returns: (T, d)
    """
    T, d = q.shape
    scale = 1.0 / math.sqrt(d)
    o = torch.zeros(T, d, device=q.device, dtype=q.dtype)
    L = torch.zeros(T, device=q.device, dtype=q.dtype)

    for i in range(0, T, Br):
        q_block = q[i:i+Br] # (Br, d)
        o_block = torch.zeros(q_block.shape[0], d, device=q.device, dtype=q.dtype) # (Br, d)
        m_block = torch.full((q_block.shape[0],), float('-inf'), device=q.device, dtype=q.dtype) # (Br,)
        l_block = torch.zeros(q_block.shape[0], device=q.device, dtype=q.dtype) # (Br,)

        for j in range(0, T, Bc):
            k_block = k[j : j + Bc]
            v_block = v[j : j + Bc]
            s_block = (q_block @ k_block.T) * scale

            if causal:
                row_idx = torch.arange(i, i + q_block.shape[0], device=q.device)[:, None]
                col_idx = torch.arange(j, j + k_block.shape[0], device=q.device)[None, :]
                s_block = s_block.masked_fill(col_idx > row_idx, float('-inf'))

            m_new = torch.maximum(m_block, s_block.max(dim=-1).values)
            p = torch.exp(s_block - m_new.unsqueeze(-1))
            l_new = torch.exp(m_block - m_new) * l_block + p.sum(dim=-1)

            o_block = (
                (l_block / l_new * torch.exp(m_block - m_new)).unsqueeze(-1) * o_block
                + (p / l_new.unsqueeze(-1)) @ v_block
            )
            m_block, l_block = m_new, l_new

        L[i:i+Br] = m_block + torch.log(l_block)
        o[i:i+Br] = o_block
    return o, L

def grad_check():
    T, d = 32, 16
    q = torch.randn(T, d, requires_grad=True)
    k = torch.randn(T, d, requires_grad=True)
    v = torch.randn(T, d, requires_grad=True)

  # standard attention gradients
    q1, k1, v1 = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)
    out_std = standard_attention(q1, k1, v1, causal=True)
    out_std.sum().backward()

  # flash attention gradients (via autograd through YOUR forward)
    q2, k2, v2 = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)
    out_flash = flash_attention_forward(q2, k2, v2, Br=8, Bc=8, causal=True)
    out_flash.sum().backward()

    print("dQ match:", torch.allclose(q1.grad, q2.grad, atol=1e-3))
    print("dK match:", torch.allclose(k1.grad, k2.grad, atol=1e-3))
    print("dV match:", torch.allclose(v1.grad, v2.grad, atol=1e-3))

def flash_attention_backward(q, k, v, O, L, dO, Br=32, Bc=32, causal=True):
    T, d = q.shape
    scale = 1.0 / math.sqrt(d)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # Softmax correction D_i = dO_i . O_i (one scalar per query)
    D = (dO * O).sum(dim=-1) # (T, )

    for i in range(0, T, Br):
        q_block = q[i:i+Br]
        dO_block = dO[i:i+Br]
        L_block = L[i:i+Br]
        D_block = D[i:i+Br]
        dQ_block = torch.zeros_like(q_block)

        for j in range(0, T, Bc):
            k_block = k[j:j+Bc]
            v_block = v[j:j+Bc]
            # scores for this tile (same as forward)
            s_block = (q_block @ k_block.T) * scale
            # causal mask (same as forward)
            if causal:
                row_idx = torch.arange(i, i + q_block.shape[0], device=q.device)[:, None]
                col_idx = torch.arange(j, j + k_block.shape[0], device=q.device)[None, :]
                s_block = s_block.masked_fill(col_idx > row_idx, float('-inf'))

            # Rebuild P using saved L (not torch.softmax on the tile)
            # p_ij = exp(S_ij - L_i)
            p_block = torch.exp(s_block - L_block.unsqueeze(-1))
            # After causal amsk, -inf scores -> exp(-inf - L) = 0, which is correct

            # Gradients for this tile
            dv[j : j + v_block.shape[0]] += p_block.T @ dO_block
            dP_block = dO_block @ v_block.T
            dS_block = p_block * (dP_block - D_block.unsqueeze(-1))
            dQ_block += (dS_block @ k_block) * scale
            dk[j : j + k_block.shape[0]] += (dS_block.T @ q_block) * scale
        
        dq[i : i + q_block.shape[0]] = dQ_block
    
    return dq, dk, dv

if __name__ == "__main__":
    torch.manual_seed(42)
    for T, d in [(8, 4), (37, 16), (128, 64)]:
        q = torch.randn(T, d)
        k = torch.randn(T, d)
        v = torch.randn(T, d)
        ref = standard_attention(q, k, v, causal=True)
        out, L = flash_attention_forward(q, k, v, Br=4, Bc=4, causal=True)
        ok = torch.allclose(ref, out, atol=1e-4)
        print(f"T={T:3d} d={d:2d}  causal=True   match={ok}")
        ref2 = standard_attention(q, k, v, causal=False)
        out2, L2 = flash_attention_forward(q, k, v, Br=7, Bc=7, causal=False)
        ok2 = torch.allclose(ref2, out2, atol=1e-4)
        print(f"T={T:3d} d={d:2d}  causal=False  match={ok2}")
    
    T, d = 32, 16
    q = torch.randn(T, d, requires_grad=True)
    k = torch.randn(T, d, requires_grad=True)
    v = torch.randn(T, d, requires_grad=True)

    # Reference grads from standard attention
    out_std = standard_attention(q, k, v, causal=True)
    out_std.backward(torch.ones_like(out_std))
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # Your flash forward + manual backward
    q2, k2, v2 = q.detach(), k.detach(), v.detach()
    O, L = flash_attention_forward(q2, k2, v2, Br=8, Bc=8, causal=True)
    dO = torch.ones_like(O)
    dq, dk, dv = flash_attention_backward(q2, k2, v2, O, L, dO, Br=8, Bc=8, causal=True)

    print("dQ", torch.allclose(dq_ref, dq, atol=1e-3))
    print("dK", torch.allclose(dk_ref, dk, atol=1e-3))
    print("dV", torch.allclose(dv_ref, dv, atol=1e-3))