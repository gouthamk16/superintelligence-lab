import torch

def softmax_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Standard softmax operation for a 1D tensor.
    """
    m = x.max()
    l = x - m
    return torch.exp(l) / torch.exp(l).sum()

def online_softmax_1d(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Same result, but processes x in chunks of chunk_size.
    Algorithm:
        1. For a 1d vector x, split it into chunks of size chunk_size.
        2. Track running max of all the scores so far, running sum of exp(score-m) over all scores seen
        3. When a new chunk is processed, update the running max and sum.
        4. After processing all chunks, divide the running sum by the running max to get the softmax -> exp(x-m)/sum(exp(x-m)).
    """
    m = float('-inf')
    l = 0.0
    # Loop over chunks, maintain m, l (no overflow needed for pure softmax)
    for i in range(0, x.numel(), chunk_size):
        chunk = x[i : i + chunk_size]
        m_new = max(m, chunk.max())
        l_new = torch.exp(m - m_new) * l + torch.exp(chunk - m_new).sum() # The first part is to rescale the old contributions 
        # to the new max, else old values become negligible compared to the new max.
        m, l = m_new, l_new
    
    return torch.exp(x - m) / l

if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(1000)

    ref = softmax_1d(x)
    for chunk_size in [1, 7, 32, 500]:
        out = online_softmax_1d(x, chunk_size)
        ok = torch.allclose(ref, out, atol=1e-6)
        print(f"Chunk size: {chunk_size}, OK: {ok}")
    