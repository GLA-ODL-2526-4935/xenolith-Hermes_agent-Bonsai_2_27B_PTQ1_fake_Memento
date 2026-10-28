# Additional inference review — 22 October 2026

Current decision policy: conditional combination plan (local development notes; excluded from source distribution),
25 October 2026. It supersedes blanket either/or selections below; retain this
review as historical mechanisms/source-status evidence. Existing target-testing
deferrals and correctness exclusions remain in force. No new runtime work.

Target: Bonsai2 27B PTQ1 Lean, Intel i9-13900H-class CPU and shared-RAM Intel iGPU,
16GB Linux laptop. Source preparation only. No model, native engine or GPU run.
Read alongside `INTEL_INFERENCE_PAPER_REVIEW.md`,
`ENGINE_DECODE_STRUCTURE_ROADMAP.md` and the engine README. The new optional
`prism-intel-decode-increment` is a delta on exact v3; existing candidates are
not counted again as additional optimizations.

## New paper investigations and decisions

### FLUTE — packed layout aligned with the execution tile

[Fast Matrix Multiplications for Lookup Table-Quantized LLMs](https://arxiv.org/html/2407.10960v1),
§3: restructures quantized matrices offline, vectorizes LUT access and partitions
work with Stream-K on GPUs. The transferable idea is to place packed elements in
the order the microkernel consumes them. GPU shared-memory duplication and Stream-K
are not direct stock-AVX2 recipes. Current row-tiled lookup already amortizes tables;
do not implement that twice. Its floating partial reductions also need different
parity reasoning from our unchanged FP32 accumulation order.

Decision: investigate a lossless block-major/row-tile permutation as a later
single-representation backend layout. Sixteen original PTQ1 blocks still occupy
448 payload/scale bytes; alignment, descriptors and tails add cost. Byte-per-pair
predecoded indices would need 64+2 bytes per128 weights versus current28, so they
are unsuitable as an unconditional resident cache. Repacking every generated token
adds stores and does not remove source reads. No automatic converter or extra
model copy is implemented. A CPU-only layout must also serve or reversibly feed
the existing Vulkan prefill without duplicating resident weights.

### LUT-GEMM — fuse lookup with accumulation

[LUT-GEMM](https://arxiv.org/html/2206.09557v4), §3: uses binary-coded weights and
activation lookup to avoid a separate dequantized matrix in GPU inference.
Our packed AVX2 and exact pair-LUT paths already consume PTQ1 without materializing
a full dequantized matrix. Its BCQ conversion, group choices and accuracy tradeoffs
are not permission to requantize Bonsai. No CUDA port or duplicate dequantization
fusion is selected. Continue bounded local LUT/dot work and measure lookup setup
plus accumulation rather than kernel arithmetic alone.

### Ghidorah — profile heterogeneous work and contention

[Ghidorah](https://arxiv.org/html/2505.23219v2), §III: profiles speculative workload
and partitions linear/attention work across heterogeneous units on unified memory.
Shared address space does not remove memory bandwidth contention or synchronization.
Its ARM sparse kernels and Jetson GPU partitioning are not Intel implementations.
Our Sandwich phase plans already cover CPU phase-specific topology. A new concurrent
CPU/iGPU tensor partitioner would need backend scheduling, one coherent resident
layout and target contention measurements. Defer it; preserve the present separate
Intel Vulkan route. Do not run a second independent online tuner alongside MTP.

### ReplaySSM — input replay instead of eager state writes

[ReplaySSM](https://arxiv.org/html/2610.13778v1), §§4–5 and AppendixB: caches recurrent
inputs, computes outputs without materializing every state, and commits buffered
prefixes. This is relevant to Bonsai's hybrid recurrent state, rather than only its
attention KV. The paper explicitly allows floating-point differences from unrolled
evaluation; Gated DeltaNet also needs correction/triangular dependencies. Its CUDA
vLLM results do not establish Intel CPU/iGPU benefit. Do not replace checkpoint
rollback with a pointer move in the existing graph. Defer this graph/backend rewrite
under the present numerical-correctness requirement; low-precision checkpoint
variants and approximate state skipping are excluded.

## Implemented additional source candidates

- **E1, exact LUT local work:** optional int16 group accumulation with proven
  [-4096,4096] bound, widening once per group, SIMD algebraic table construction,
  bounded vector tail decode and no unnecessary full-tile scratch clearing.
  No floating arithmetic is reordered, and all four Q8 scales remain separate.
- **E2, MTP owned hidden-row capture:** keep earlier rows in reusable owned storage,
  final row directly in pending carryover, removing one intermediate copy. Existing
  acceptance indices, prefill catch-up, hidden-state ownership and KV/recurrent
  rollback remain unchanged. This is additional to v3's full-acceptance copy elision.

Each has an independent compile flag and an unchanged default path. Only three
source paths differ from v3. Existing BITCOS codec/cost gates and optional LUT/batch
dispatch remain unchanged. Full runtime/numerical/performance acceptance is open.

## Other requested areas: audit disposition

| Area | Decision and evidence needed |
| --- | --- |
| Weight transfer | Existing unpack reuse for2–5 columns is retained. Global alternative packing awaits real density and memory-controller measurements; file size or source load count is not actual DRAM traffic. |
| Activation traffic | Existing LUT row tiles reuse tables. E1 removes scalar table staging and repeated widening; compiler spills/cache behavior still need measurement. Cross-QKV/gate/up activation fusion requires exact graph/transform identity and an allocation trace first. |
| KV traffic and rollback | Existing pinned server already distinguishes bounded recurrent rollback from checkpoint/replay and clones the sampler only when replay can be needed. Preserve those decisions and checkpoint contents. No new checkpoint elision is claimed. |
| Attention fusion | Previously reviewed upstream tiled/online attention and eligible split-KV remain. Current Q4_0 KV does not select that reviewed F16/F32 CPU split-KV path. No new kernel or silently dequantized full KV cache is added. |
| Buffers | E2 preserves vector capacity and owns rows past decode. No borrowed output pointer across graphs, forced shrink, heap full-weight expansion or second resident model. |
| Intel iGPU | Vulkan code, selection, graph layout and live profiles are untouched. Unified RAM is not a reason to introduce CPU/GPU transfers or assume zero-copy backend ownership. |

## Acceptance still required

Run native integer/bitwise tests and strict-FP full-runtime checks on the target;
then logits/tokens, per-prefix MTP rejection/replay, cancellation and reset. Test
each flag separately and combined against the same v3 build. Record actual DRAM
bytes/token, whole-workflow latency, peak RSS/shared allocations, swap and sustained
thermals, plus iGPU prefill independently. A source-level reduction or mathematical
oracle is not an executed SIMD test, target benchmark or correctness certificate.
Retain the fallback for failures, memory overflow or gains indistinguishable from
noise. No target speedup percentage is predicted from any paper's published number.
