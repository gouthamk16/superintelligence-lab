# Thinking Beyond Transformers: Memory-Centric Neural Architectures.

## Research goal

Come up with a neural architecture with a fundamentally better long-term memory mechanism than existing recurrent or attention-based models, enabling more efficient reasoning, longer context retention and improved scalability towards superintelligence.

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

Simply cannot store every token forever - poor design. Existing solutions (e.g., periodic KV rewriting, learned compression, bottlenecked Transformers) are too dependent on KV Cache to such an extent that the field is largely optimizing KV management rather than solving the root problem. This will lead to a point where the model will not be able to scale any further.

Is this an actual problem - yes, does it affect current reasoning performances to a huge extent - no, will this problem be a bottleneck for scaling to ASI - yes.

### Quadratic Attention Complexity

Comparing every token with each other (kind of like a full connected graph bw all tokens) is a O(n^2) operation - expensive. As context grows:

- Compute explodes.
- Latency increases.
- Energy consumption increases (not that i care, but a case to be made).
- Cost per inference increases.

Hence ultra-long context reasoning (ASI should not be restricted by context window limits, unlimited context?) becomes economically impractical. Long context windows demands immense GPU memory bandwidth, hence creating scaling walls for high-res inputs or ultra-long docs. Intelligence should not require comparing every memory with every other memory.

Now, what about solutions like flash attention? Here's why (read it from the perspective of a person who is trying hard to prove that Transformers are not the best architecture for ASI):

**FlashAttention**
- Doesn't reduce O(n^2).
- Only makes it GPU efficient.

**Sparse attention**
- Assumes most interactions aren't needed.
- Can miss important long-range dependencies.

**Linear attention**
- Changes the attention function.
- Often loses expressiveness or requires architectural tradeoffs.

**SSMs**
- Compress history into a fixed state.
- Can struggle with recalling arbitrary past information.

The solution is not to compute pairwise interactions faster, but it might be to not compute pairwise interactions at all.

### Memory Bandwidth Bottleneck

SK Hynix and Samsung will be pissed if I come up with a solution for this :). Modern GPUs are often limited not by arithmetic throughput, but by how fast they can move data between high-bandwidth memory (HBM) and on-chip SRAM, leading to model systems spending an enormous amount of time moving data, not computing. Why:

- attention required reading all previous K/V vectors (flash attention solves this to a great extent).
- each layer has billions of params.
- activations don't fit entirely in SRAM (flash attention saves the day again).

Future models will have larger contexts, more parameters, more simultaneous users etc, and each one of these is only going to increase memory traffic. Eventually the bottleneck becomes bytes/second (i think we might have already hit this point).

Even if future hardware becomes 10× faster computationally, intelligence won't scale proportionally if the architecture still spends much of its time waiting on memory transfers. This is a classic example of Amdahl's Law: speeding up computation helps less and less if data movement dominates execution time.

### Poor to No Continual Learning

Once trained, the model is essentially frozen. It cannot naturally learn from new experiences. Inference does not update the model. In an ideal scenario, ASI should:

- improve daily.
- adapt continuously.
- accumulate knowledge from new experiences.
- update beliefs.
- never stop learning.

But transformers are not designed for this. They are designed for one-time training and inference. Finetuning is not a solution, can't scale to millions of updates/day. Solutions like replay buffers and adapter based learning might be of use to us for this use case.

### Common link - memory

The first five bottlenecks all revolve around memory:

- KV cache - working memory.
- Attention - accessing memory.
- Context window - temporary memory.
- IO - moving memory.
- Continual learning - updating memory.

We're converging on the first major research question to work on:
> "What is the correct computational model of memory for an intelligent system?"

## The theory of memory: What should constitute a "memory"?

