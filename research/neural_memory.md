# Thinking Beyond Transformers: Memory-Centric Neural Architectures.

## Research goal

Come up with a neural architecture with a fundamentally better long-term memory mechanism than existing recurrent or attention-based models, enabling more efficient reasoning, longer context retention and improved scalability towards general intelligence.

## Core Research Questions

- Why do current architectures forget?
- What should constitute a "memory"?
- When should a memory be formed?
- How should memories be organized?
- How should memories be retrieved?
- Can memory formation itself be learned?
- Can memory scale independently of computation?

## Why move away from Transformers?

It's not that Transformers are inefficient, we are trying to figure out if there are any fundamental bottlenecks that prevent transformers from scaling efficiently toward AGI/Superintelligence.

Some basic inefficiencies of Transformers:
1. O(n^2) complexity in attention.
2. The concept of KV cache - grows linearly with the number of tokens in the input.
3. No persistent memory bw sessions/documents.
4. Every token reprocesses history instead of maintaining some kind of state.
5. Huge memory bandwidth requirements for large models.
6. Context window is not equivalent to memory. (once tokens are gone, they're gone)
7. Poor continual/lifelong learning.
8. Compute scales with seq len.

Paves the way for the research question:  
> "Can we design an architecture whose computation depends on the amount of useful information rather than the number of past tokens?" 

But before we try to answer this question, we need to build a strong case against Transformers. 

### KV Cache Explosion

KV Cache was born out of an inherent limitation of transformers - the need to remember every token representation in the context window. The transformer is without any compression state and to calculate which previous tokens are relevant to the current token, it needs to either recompute K and V for every token or store all previous K, Vs for that token in memory i.e., Every generated token must keep its Keys and Values alive for future attention.

Hence memory grows as:

- O(seq len)
- O(num layers)
- O(KV Heads)
- O(batch size)



