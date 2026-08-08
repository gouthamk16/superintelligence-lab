# Summary: Memory Caching — RNNs with Growing Memory

**Paper:** [arXiv:2602.24281](https://arxiv.org/abs/2602.24281) (Behrouz et al., Google Research / Cornell / USC, Feb 2026)  
**Tag:** `memory_caching`  
**Local source:** `~/.cache/research_papers/knowledge/2602.24281/` (`main.tex`)

---

## One-line takeaway

Cache **checkpoints of an RNN’s memory state** after each segment, then let each new token retrieve from the **online memory + past cached memories**. Effective capacity grows with sequence length at cost $\mathcal{O}(NL)$ — a dial between fixed-state RNNs ($\mathcal{O}(L)$) and Transformers ($\mathcal{O}(L^2)$).

---

## Problem they target

- Transformers: growing associative memory → strong recall, quadratic cost + KV cache.
- Modern RNNs / linear attention / Titans: compress history into a **fixed-size** state → efficient, but forget under long / recall-heavy workloads.
- MC’s claim: the bottleneck is not recurrence itself, but **not being able to revisit compressed past states**.

This sits squarely on the research questions in `research/neural_memory.md`: *what is memory, how is it organized, how is it retrieved, can capacity grow without full pairwise attention?*

---

## Core idea

1. Split sequence into segments $S^{(1)}, \ldots, S^{(N)}$.
2. Run the usual recurrent update inside each segment:

$$
\mathcal{M}_{t}^{(s)} = f\left(\mathcal{M}_{t-1}^{(s)};\, \boldsymbol{k}_{t},\, \boldsymbol{v}_{t}\right)
\quad\text{where}\quad
1 \leq t \leq L^{(s)}
$$

3. **Cache** the end-of-segment state $\mathcal{M}_{L^{(s)}}^{(s)}$.
4. For query $\boldsymbol{q}_{t}$ in segment $s$, retrieve via aggregation over online + cached memories:

$$
\mathbf{y}_{t}
=
\operatorname{Agg}\!\left(
\{\mathcal{M}_{L^{(1)}}^{(1)}(\cdot),\ldots,\mathcal{M}_{L^{(s-1)}}^{(s-1)}(\cdot)\};\;
\mathcal{M}_{t}^{(s)}(\cdot);\;
\boldsymbol{q}_{t}
\right)
$$

**Complexity.** Update is still $\mathcal{O}(L)$; retrieval is $\mathcal{O}(N)$ per token → total $\mathcal{O}(NL)$.

- $N = 1$: plain RNN.
- $N = L$ (segment size 1): recovers (gated) attention-like behavior in the limit.
- Constant segment size $C = L/N$: roughly $\mathcal{O}(L^{2}/C)$.
- Log segmentation: $\mathcal{O}(L\log L)$, cheaper but worse long-past resolution.

---

## Four aggregation variants

### Residual Memory

Simplest aggregator — residual sum over online + cached memories:

$$
\mathbf{y}_{t}
=
\underbrace{\mathcal{M}_{t}^{(s)}(\boldsymbol{q}_{t})}_{\text{online memory}}
+
\underbrace{\sum_{i=1}^{s-1}\mathcal{M}_{L^{(i)}}^{(i)}(\boldsymbol{q}_{t})}_{\text{cached memories}}
$$

For *linear* $\mathcal{M}$ this collapses to one summed matrix:

$$
\mathbf{y}_{t}
=
\left(
\mathcal{M}_{t}^{(s)}
+
\sum_{i=1}^{s-1}\mathcal{M}_{L^{(i)}}^{(i)}
\right)\boldsymbol{q}_{t}
$$

Still helps empirically as a retention operator.

### Gated Residual Memory (GRM)

Same residual form with input/context-dependent gates $0 \leq \gamma_{t}^{(i)} \leq 1$:

$$
\mathbf{y}_{t}
=
\gamma_{t}^{(s)}\,\mathcal{M}_{t}^{(s)}(\boldsymbol{q}_{t})
+
\sum_{i=1}^{s-1}
\gamma_{t}^{(i)}\,\mathcal{M}_{L^{(i)}}^{(i)}(\boldsymbol{q}_{t})
$$

Gates use similarity between a projected query and each segment’s pooled context (then softmax-normalized):

$$
\gamma_{t}^{(i)}
=
\left\langle
\boldsymbol{u}_{t},\;
\operatorname{MeanPooling}(S^{(i)})
\right\rangle
\qquad\text{where}\qquad
\boldsymbol{u}_{t} = x_{t}W_{\boldsymbol{u}}
$$

Prevents collapse even for linear memory. Best overall in most tables.

### Memory Soup

Interpolate **parameters** of cached memories into one data-dependent module $\mathcal{M}_{t}^{*}$, then retrieve:

$$
\mathbf{y}_{t} = \mathcal{M}_{t}^{*}(\boldsymbol{q}_{t})
$$

with

$$
\boldsymbol{\theta}_{\mathcal{M}_{t}^{*}}
:=
\left\{
\sum_{i=1}^{s}\gamma_{t}^{(i)}W_{1}^{(i)},\;
\ldots,\;
\sum_{i=1}^{s}\gamma_{t}^{(i)}W_{c}^{(i)}
\right\}
$$

Equivalent to GRM when $\mathcal{M}$ is linear; distinct for deep/nonlinear memory (DLA, Titans).

### Sparse Selective Caching (SSC)

MoE-style Top-$k$ router over segment relevance scores:

$$
\mathbf{r}_{t}^{(i)}
=
\left\langle
\boldsymbol{u}_{t},\;
\operatorname{MeanPooling}(S^{(i)})
\right\rangle,
\qquad
\mathcal{R}_{t}
=
\arg\operatorname{Top\text{-}k}\!\left(\{\mathbf{r}_{t}^{(i)}\}_{i=1}^{s-1}\right)
$$

Retrieve only from selected caches (+ online memory):

$$
\mathbf{y}_{t}
=
\gamma_{t}^{(s)}\,\mathcal{M}_{t}^{(s)}(\boldsymbol{q}_{t})
+
\sum_{i \in \mathcal{R}_{t}}
\gamma_{t}^{(i)}\,\mathcal{M}_{L^{(i)}}^{(i)}(\boldsymbol{q}_{t})
$$

Best efficiency / quality tradeoff at long context.

---

### Two write modes (Section 3.4)

**1. Checkpoints of one optimizer** — continuous test-time optimization; each segment continues from the previous end state:

$$
\mathcal{M}_{0}^{(s)}(\cdot) = \mathcal{M}_{L^{(s-1)}}^{(s-1)}(\cdot)
$$

**2. Independent compressors** — fresh init per segment so each cache is a clean summary of that block (less interference):

$$
\mathcal{M}_{0}^{(s)}(\cdot)
\quad\text{independent of}\quad
\mathcal{M}_{L^{(s-1)}}^{(s-1)}(\cdot)
$$

Both have tradeoffs; paper leaves this as an empirical design choice.

---

## Nested-learning framing (why this is interesting for us)

They cast sequence models as **test-time associative memory** optimizing an attentional bias (Miras / Nested Learning line of work):

$$
\mathcal{M}_{t+1}
=
\arg\min_{\mathcal{M}}\;
\mathcal{L}\!\left(\mathcal{M}(\boldsymbol{k}_{t});\, \boldsymbol{v}_{t}\right)
+
\operatorname{Ret}\!\left(\mathcal{M};\, \mathcal{M}_{t}\right)
$$

Cached states = **checkpoints of that inner optimization**. Retrieval is not “attend to raw tokens” but “query past optimized memory snapshots.” That is a cleaner answer to “what constitutes a memory” than KV lists: memory is a *learned map*, and the archive is *compressed maps*, not token embeddings.

Extreme case (segment size 1, value-less vector memory + gating) **recovers gated global attention**. Hybrid “RNN then attention” layers are reinterpreted as MC with checkpointing and $\boldsymbol{q}_{t} = \mathbf{1}$. Deep memory (MLP Titans/DLA) does *not* collapse to hybrid attention — each token can be represented by a small network whose readout depends on the query.

---

## What they applied it to

Proof-of-concept on:

- Linear Attention / SWLA ($c = 2$)
- Deep Linear Attention (DLA)
- Titans (LMM) — deep memory + momentum-style optimizer on $\|\mathcal{M}(\boldsymbol{k}) - \boldsymbol{v}\|_{2}^{2}$
- Log-Linear++ (their improved log-segmentation baseline for fairness)

Also note: **MC as post-training / inference trick** — cache memory every training-length chunk, decode with moving average of past caches (no learned gates). Helps length extrapolation even without retraining.

---

## Empirical headlines (academic scale)

Setup: 760M / 30B tokens and 1.3B / 100B on FineWeb (+ long-data mix); default context 4K, segment 256 for LM; 16K context for NIAH / retrieval / LongBench.

- **LM + commonsenses:** MC consistently lifts SWLA / DLA / Titans. Titans + GRM is strongest (e.g. 1.3B avg ~58.3 vs Titans 56.8 vs Transformer++ 53.2).
- **NIAH:** big gains over base RNNs; closes a lot of the gap to Transformers at 4–16K, especially Titans + GRM.
- **In-context retrieval (SWDE, SQuAD, FDA, …):** Transformers still win on average; MC closes the gap and beats strong recurrent baselines. Recall is still the hard case.
- **LongBench:** MC helps across single/multi-doc QA, few-shot, code.
- **MQAR:** MC-enhanced models competitive / best-per-dimension vs Atlas-class recurrents.
- **Efficiency:** throughput sits between RNNs and Transformers; **SSC** adds little overhead vs base RNN at long $L$.
- **Ablations:** context-dependent $\gamma$, gating, and deep memory all help; constant-size segments beat log segmentation for recall (as expected from resolution).

Honest reading: MC is a strong **band-aid for fixed-state recurrence**, not a proof that Transformers are obsolete for associative recall. Transformers remain best on pure retrieval; MC is how you buy most of that without full KV.

---

## Relevance to this repo (`nano-gpt` / research notes)

### Connection to `research/neural_memory.md`

This paper is already cited as ref [6] under “Cached Memory RNNs.” It directly addresses several of your core questions:

| Your question | MC’s answer |
|---|---|
| Why do architectures forget? | Fixed compressor overwrites; no revisitable past states. |
| What is a memory? | Parameters of an associative map $\mathcal{M}$, not a token list. |
| When is memory formed? | Continuously via recurrent update; **snapshots** at segment boundaries. |
| How organized? | Segmented archive of compressed states (optionally Top-$k$ sparse). |
| How retrieved? | Query each cached $\mathcal{M}$ (gated / soup / sparse). |
| Can capacity scale ≠ compute? | Partially: capacity ~ $N$ caches, compute $\mathcal{O}(NL)$. Still grows with length, but *controllably*. |

What MC does **not** solve from your ASI checklist:

- No persistent memory across sessions/documents.
- No continual / lifelong learning outside the forward pass.
- Archive still grows with sequence (unless you aggressively sparsify / evict).
- Mean-pool routing is a weak content address; Hopfield-style attractors / better routers are open.

So treat MC as: **best current recipe for “growing memory without full attention” inside a single context**, and a stepping stone toward something with independent memory scaling.

### Connection to `v1.0.py` / `bigram.py`

Current codebase is a small **Transformer** (causal multi-head attention, `block_size=256`, Shakespeare char LM). Natural experiments if we want to *feel* this paper:

1. **Replace attention with linear attention / fixed-state RNN**, measure recall collapse on longer `block_size`.
2. **Add MC (start with Residual or GRM)** on that recurrent baseline: segment size e.g. 32–64, cache end-of-segment matrices, sum/gate query readouts.
3. **Ablate $N$** (1 vs $L/C$ vs log) on a tiny needle / associative-recall toy — matches your MQAR / NIAH interest without 1B-scale training.
4. **Post-training MC**: train plain linear RNN at short context, at decode cache every `block_size` and average — cheapest length-extrapolation experiment.
5. Longer term: **deep memory (Titans-style)** + MC, then ask whether independent compressors vs continuous checkpoints better match your “memory consolidation” intuition.

Minimal first prototype for nano-gpt scale (linear residual MC):

$$
\mathcal{M}_{t}^{(s)}
=
\mathcal{M}_{t-1}^{(s)}
+
\boldsymbol{v}_{t}\boldsymbol{k}_{t}^{\top},
\qquad
\mathbf{y}_{t}
=
\gamma_{t}^{(s)}\,\mathcal{M}_{t}^{(s)}\boldsymbol{q}_{t}
+
\sum_{i=1}^{s-1}
\gamma_{t}^{(i)}\,\mathcal{M}_{L^{(i)}}^{(i)}\boldsymbol{q}_{t}
$$

That is enough to study the $\mathcal{O}(NL)$ dial on Shakespeare / synthetic recall before committing to Titans-scale machinery.

---

## Ideas worth stealing for the research agenda

1. **Memory = optimized map, archive = checkpoints** — better ontology than “KV cache with compression.”
2. **Segment size as the compression–compute knob** — explicit answer to “compute should depend on useful information,” though still proxying “useful” by fixed chunks.
3. **Sparse selective caching** — closest thing in the paper to content-addressable / MoE memory; replace MeanPool with something Hopfield-like.
4. **Independent compressors** — closer to episodic memory slots that don’t overwrite each other; checkpoints closer to continual optimization with snapshots.
5. **Don’t stop at hybrids** — paper argues deep-memory MC is *not* just “RNN + attention”; that’s a path past the “modern Hopfield = attention” dead end in your notes.

---

## Limitations / open threads

- Still grows with $N$; not unbounded persistent memory.
- Routing via mean-pool is crude.
- Strongest gains when base model already has decent deep memory (Titans).
- Transformers win pure in-context recall.
- Checkpoint vs independent compressor underexplored in the main tables.
- Academic scale only ($\leq 1.3$B); no proof at frontier.

---

## Bottom line for us

Memory Caching is the cleanest recent formalization of: *keep recurrence for writing, grow a sparse archive of compressed states for reading.* It is immediately actionable in nano-gpt as a recurrent + cache experiment, and conceptually it advances your memory research from “KV is the wrong object” to “cache optimized associative maps, retrieve by query, control growth via segmentation/sparsity.” Next research step is not “make MC bigger,” but **make the archive persistent, content-addressable, and writable outside a single context window** — using MC’s segment caches as the short-term / working layer.
