# Additional inference papers — 23 October 2026

Research decisions for dense Bonsai 2 27B PTQ1 Lean on the intended 16 GB Intel
CPU/shared-RAM iGPU laptop. These are recommendations, not new runtime features
or measured Intel speed gains. No model was loaded and no inference was run.

## What is worth taking forward

| Priority | Addition | Existing overlap | Decision |
|---|---|---|---|
| 1 | Account for the entire speculative cycle, not acceptance alone | BanditSpec selector and persistent profiles already exist | Extend measurement/validation of the existing selector; do not add a second controller |
| 2 | Recurrent-state oracle and accepted-prefix publication tests | Existing rollback, checkpoint/replay and hidden-row fixes | Strengthen backend acceptance tests before changing state algorithms |
| 3 | Shape-aware attention work partitioning | Flash-Decoding review and upstream split attention | Conditional long-context experiment after profiling; not another default attention path |
| Research | Reconstruct only accepted GDN state | ReplaySSM already reviewed | Compare competing state strategies, not stacked optimizations; defer mathematical reformulation under strict parity |

## New papers and structural findings

### SmartSpec — complete-cycle cost modeling

[Optimizing Speculative Decoding for Serving Large Language Models Using Goodput](https://arxiv.org/html/2406.14066v2), §§4.1–4.4.
The useful structure is expected committed tokens divided by draft plus target
execution time, with forward cost conditioned on context and batch shape. Hardware
coefficients are profiled, not universal. Serving-oriented batching is less useful
for our single slot.

Adaptation: audit that our measured arm cost covers draft lookup/MTP, verification,
state save/restore/replay, synchronization and selection overhead without double
counting. Report accepted drafts and bonus tokens separately. Keep the current
MTP3/MTP4/suffix4 restricted experiment unchanged; this paper is not authorization
to restore an off arm or enlarge the window. Recalibrate on Intel, because the
existing GPU profile cannot predict shared-LPDDR contention. Prefer bounded
measurements to fitting an unnecessary new online model. No selector change made.

### Decoding Speculative Decoding — separate drafter quality from runtime value

[Decoding Speculative Decoding](https://arxiv.org/html/2402.01528v2), §4 and tables 1–2.
The paper separates drafter latency from token acceptance and shows that a higher
acceptance rate need not give higher throughput. Its trained shallow/wide drafters
are outside our unchanged-head scope.

Adaptation: evaluate fixed MTP3, fixed MTP4 and frozen selection with matched
workflows, report all cycle components and whole-task wall time. Our last restricted
selector run already illustrates the issue: +4.63% aggregate decode tokens/s but
+1.83% HTTP wall time, with different generated lengths. That is not an established
task-latency win. No new model or head training is recommended here.

### GDN Tree-Scan — useful correctness contract before a tree implementation

[GDN Tree-Scan](https://arxiv.org/html/2609.23900v1), §§IV–V and evaluation scope.
Its structural contribution is consistent ancestry across candidate metadata,
attention visibility, recurrent computation and accepted-state publication.
Every node starts from its parent's state; only the accepted chain becomes durable.
The implementation uses vLLM/FA2/Triton and its empirical probability-rescore
evidence is not a full distribution-distance proof.

Adaptation: use the contract to design native-order recurrent-oracle tests for
our existing chain: every rejection position, zero/full acceptance, next decode,
convolution state, target KV, MTP hidden row and cancellation. Check logits and
state as well as tokens. This complements existing model-free rollback fixtures;
it does not certify that the actual backend passes. Defer tree packing and extra
candidate branches until chain correctness and cost are measured on Intel.

### TreeWY — relevant GDN memory reduction, incompatible with an automatic parity claim

[TreeWY](https://arxiv.org/html/2608.20961v1), §§3–5, equations 2–3 and table 1.
A triangular solve obtains corrected values for candidate outputs, then rebuilds
only the accepted state. This avoids full per-candidate state materialization.
It addresses GDN's rank-one correction, unlike scalar-decay-only tree methods.
However, §4 explicitly says token streams are not bit-identical. Its batch-one
table reports throughput ratios below one for the evaluated GPU cases; memory
savings do not automatically accelerate a single user.

Decision: research only. It overlaps ReplaySSM, already deferred here. Compare
snapshot/replay, deferred materialization and reconstruction as alternatives;
do not stack their state ownership strategies. First measure our actual bounded
rollback buffers rather than assume vLLM's store-all allocation. A reformulated
floating-point recurrence cannot be enabled under the present exactness requirement
without a separate correctness decision and backend validation. No Intel gain claimed.

### LeanAttention — balanced work rather than a fixed KV split

[LeanAttention](https://arxiv.org/html/2405.10480v1), §§III–IV.
It distributes attention tiles using a Stream-K-style work partition and combines
partial online-softmax results, addressing poorly balanced fixed splits.
CUDA occupancy and launch behavior do not transfer directly to Intel CPU/Vulkan.

Adaptation: only if softmax-attention layers become a significant long-context
bottleneck, compare existing split scheduling with bounded shape-aware tiles.
Do not parallelize GDN state as if it were KV attention. The existing CPU split
path's F16/F32 eligibility does not establish Q4 KV support. Keep iGPU prefill
unchanged. Mathematical exact attention can still reorder floating reductions,
so logits/argmax/state parity and deterministic merging are gates. This extends
the previous attention-review item, not a second independent implementation.

### Block Verification — no extra benefit at our temperature zero

[Block Verification Accelerates Speculative Decoding](https://arxiv.org/html/2403.10444v3),
§6.2 explicitly states that at temperature zero it degenerates to token verification.
Its joint acceptance coupling is useful for stochastic sampling; it cannot rescue
a wrong greedy token while preserving the target's deterministic continuation.
Defer for a separately requested nonzero-temperature mode. This is a different
concept from our already batched target forward pass.

### NoMAD-Attention — CPU-friendly but approximate

[NoMAD-Attention](https://arxiv.org/html/2403.01273v1), §§3.1–3.5.
Product-quantized keys and dynamically quantized dot-product tables enable fast
SIMD register lookup. Its blocked transposed layout is interesting, but the
attention-score approximation and learned codebooks conflict with unchanged
numerical execution. Reject the algorithm for this scope. Existing exact PTQ1 LUT
work already covers the generic register-lookup lesson; no approximate KV format
or second key cache is selected.

## Concrete checklist additions and acceptance gates

1. Audit the existing selector's timing boundaries and add only missing component
   accounting. Keep profile identity tied to backend/build/hardware/settings;
   freeze learning for held-out comparisons. Overhead must be measured too.
2. Complete real-backend native-order state-oracle tests for chain windows 3/4
   before any recurrent reformulation. Preserve existing fallback paths.
3. Measure actual recurrent checkpoint bytes/copies and attention share at short
   and long context. Allocation telemetry alone is not DRAM-controller traffic.
4. If attention dominates, prototype one reversible scheduling variant using
   existing dispatch; reject if reduction differences or extra scratch defeat it.
5. Measure cold/warm whole-task latency, generated lengths, valid tool actions,
   tokens/s, memory traffic where counters permit, RAM/swap and sustained thermals.
   Keep CPU decode and Intel iGPU prefill results separate.

No published speed multiplier is a prediction for this laptop. The highest-priority
work remains validating the already prepared PTQ1 and verification candidates on
the target before accumulating more unmeasured kernels.

## Local comparison evidence

Reviewed existing `ADDITIONAL_INFERENCE_REVIEW.md`, `ADDITIONAL_ENGINE_REVIEW.md`,
`BONSAI_BACKEND_STATUS.md`, README and function checklist. Rechecked composed Prism
`common/speculative.cpp` for owned suffix workspace, source policy and accepted-row
handling. Native Xenolith remains a gated Gemma graph; Bonsai execution belongs to
the external Prism backend. This review changes documentation only.
