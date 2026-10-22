# Intel inference optimization review — 22 October 2026

Prepared source only for `xenolith+Hermes_agent+Bonsai_2_27B_PTQ1_fake_Memento`.
The Bonsai engine route is the external Prism process, not native Gemma Xenolith.
No model/engine/new native test executable was run. No measured speed or bandwidth
improvement is claimed. The Intel Linux target remains unvalidated.

## Decisions from the five sources

| Source | What applies here | Decision |
|---|---|---|
| [T-MAC](https://arxiv.org/html/2407.00088v2), §§3–5 | LUT locality, tiling, reused unpack and AVX2 PSHUFB; lookup setup/cache costs matter. Table quantization and fast aggregation introduce approximation. | Implement optional AVX2 unpack sharing across 2–5 verification columns. Supply an exact-int16 LUT comparison prototype. Do not adopt approximate table arithmetic or assume a speed win. |
| [FlashAttention](https://arxiv.org/html/2205.14135v2), §3 | Tiled Q/K/V processing and online normalization avoid large score intermediates. | Already present in pinned CPU and Vulkan attention. Preserve them; no duplicate attention implementation or CUDA dependency. |
| [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | Split the KV sequence, compute partial attention, then rescale/reduce. More parallelism trades against reduction/scheduling traffic. | CPU source already splits eligible single-query FP16/FP32 KV at length ≥512. Vulkan also has split-K machinery. Quantized CPU KV and few-query MTP extensions need separate implementation/parity work. |
| [FlashInfer](https://arxiv.org/html/2501.01005v1), §3 and Appendix D | Shared-prefix reuse, variable tiles and deterministic scheduling can reduce cache/workspace traffic. | Preserve current prefix/MTP state safeguards. A new paged KV layout/scheduler is deferred: single slot, hybrid recurrent state and CPU/iGPU ownership require evidence first. |
| [LLM in a Flash](https://arxiv.org/abs/2312.11514) | Reusing resident work and contiguous access are useful principles. Its predictive sparse-neuron loading targets flash-resident weights. | No predictor, sparse skipping, weight offload or extra expanded model. Dense resident LPDDR5 execution should prioritize packed streaming and reuse. |

The source-derived decisions above are engineering judgments for this project,
not paper-reported Intel speedups. Real-arithmetic exact attention can still change
floating reduction order; mathematical equivalence alone cannot certify identical
greedy tokens or actions.

## What the pinned runtime actually does

Audited [sudoingX ff414120](https://github.com/sudoingX/llama.cpp/tree/ff414120c343e6e6cb868013c99f1dde52b27e70),
the existing MTP patch base, rather than substituting a newer Prism source pin.

- `ggml-common.h`: PTQ1 stores 128 trits in 24 five-trit bytes, two four-trit bytes
  and one FP16 scale: 28 bytes, 1.75 bits/weight. It is not a binary sign bitmap.
  Four Q8 blocks contribute separate scales. Do not collapse these into one scale.
- `ggml-cpu/ggml-cpu.c`, `ggml_compute_forward_mul_mat_one_chunk`: 16×16 tiling
  calls a full dot separately for each column. Cached weight reads can already be
  reused, but base-3 unpacking repeats. The new eligible path unpacks once per
  block and applies all columns while preserving source accumulation order.
- `ggml-cpu/ops.cpp`: `ggml_compute_forward_flash_attn_ext_tiled` stages query/KV
  tiles and tracks running maxima/sums; `ggml_compute_forward_flash_attn_ext_f16`
  selects the split-KV path only for one query, one batch, F32 queries, matching
  F16/F32 K/V, length ≥512 and non-reference execution. Each worker computes a KV
  range followed by a barrier and deterministic partial reduction. AVX512-specific
  specialization is not an assumed feature of this AVX2 Intel target.
- The profile's Q4_0 KV does **not** satisfy that split-KV guard. Its quantized
  vector attention still uses online normalization, but not this split/tiled path.
  Switching to F16 KV is not a free optimization: extra RAM and traffic can outweigh
  parallelism. No cache-type change was made.
- `ggml-vulkan/ggml-vulkan.cpp` has PTQ1 matmul pipelines and flash-attention/split-K
  selection. `flash_attn.comp` stages KV in shared memory and rescales softmax
  accumulators. PTQ1 lacks the coopmat2 route noted in that source; generic pipelines
  remain. Driver/device shape support still determines actual dispatch. No shader,
  GPU placement, Intel prefill path or driver was changed.
- `src/models/qwen35.cpp` contains both recurrent gated-delta and full-attention
  layers; appended MTP layers are non-recurrent. Attention optimizations cannot be
  extrapolated to every layer. `src/llama-graph.cpp` already memoizes shared rotated
  activations; creating another persistent rotated weight model would waste RAM.

CPU decoding and Intel iGPU prefill are distinct measurement cases. The present
launcher selects a CPU or Vulkan route; it does not implement automatic GPU-prefill
then CPU-decode migration. On an iGPU, shared physical RAM does not eliminate
allocation, staging, cache coherence or synchronization costs. No unsupported
phase-switching behavior is claimed or introduced.

## Source delivered

`engine/patches/prism-ptq1-reuse/` contains the cumulative v2 patch, strict source
installer, hashes, single-AVX2 and batch headers, exact LUT prototype, future parity
test and microbenchmark, compile-check fixtures, verification record and license.
It replaces the initial AVX2 patch; CPU-MTP patch files are disjoint. Nothing was
applied to an installed runtime or enabled in a launch profile.

Batch reuse is opt-in with `GGML_PTQ1_ENABLE_BATCH`. It covers ordinary 2-D PTQ1
matmul with 2–5 token columns, including a possible four-draft-plus-one target
verification batch. Reference mode, broadcasts/higher dimensions, one-column chunks,
larger prefill batches and unsupported CPU features retain existing dispatch.
It benefits eligible batched trunk operations without changing the head or its
acceptance algorithm, sampler, KV commit/rollback, or recurrent checkpoints.

The LUT prototype uses positive and negative four-trit masks with exact int16
subset sums. It is a scalar comparison, not an optimized T-MAC port. Tables expand
activation working storage substantially; benchmark setup time and cache misses
must count. It stays outside production dispatch until there is evidence to justify
a tiled SIMD implementation. No packed weight file or quantization is changed.

## Verification and next gates

Passed: patch application, all resulting source hashes, reverse round-trip,
installer syntax, actual patched x86 dispatch and entire matmul chunk compilation
using a small type scaffold in AVX2 batch-on/off, forced baseline and SSE2 variants;
parity-test and benchmark C++ object compilation. Not linked into a full runtime.

Pending: executing those tests, guard-page/sanitizer and upstream backend tests,
full Linux build, GGML dispatch/graph coverage, model logits/greedy tokens/actions,
MTP all-accept/reject/replay cases, and target measurements. Preserve existing
fallbacks until those pass. Compilation cannot establish numerical correctness.

Use the package README's benchmark procedure. Measure prompt processing separately
from decode, MTP off/on and columns 2–5, several context lengths, actual matrix shapes,
hot-cache versus larger-than-LLC weights, thread counts and sustained temperature.
Report end-to-end accepted tokens/second, task time, validation parity, peak RAM,
swap/page faults, and actual memory-controller bytes per generated token when
counter access exists. Neither repeated loads nor fewer unpack instructions prove
lower DRAM traffic. Revert any regressing candidate; do not choose a winner by theory.

The user has restricted the 5 GiB free-RAM reserve to the preparation laptop. The
existing launch profiles/validator still contain it; choosing an appropriate reserve
for the 16 GB target remains a separate configuration gap. This source increment
does not silently select a replacement reserve or remove memory safeguards.
