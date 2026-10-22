# PTQ1 unpack reuse v2 — source candidate, inactive

Prepared 22 October 2026 for the Linux Intel CPU Prism process backend of the
combined Xenolith/Hermes project. No model, new kernel/test executable, engine,
GPU or WSL was run. No speedup, numerical parity, or reduced DRAM traffic has
been measured. This is not a native Xenolith Bonsai graph implementation.

## Scope and composition

Exact base: [sudoingX/llama.cpp ff414120c343e6e6cb868013c99f1dde52b27e70](https://github.com/sudoingX/llama.cpp/tree/ff414120c343e6e6cb868013c99f1dde52b27e70).
`manifest.json` records normalized UTF-8 source hashes and patch hash.
`apply.py SOURCE` checks the commit, every input, and patch applicability without
changing SOURCE. `--apply` explicitly applies source only. It does not build or run.

This is a **cumulative replacement** for `prism-ptq1-avx2` v1, including its
single-column AVX2 candidate. Do not stack v1 and v2. If v1 is already present,
use a separate clean pinned checkout, or explicitly reverse the exact v1 patch
after checking its hashes and local edits. Do not reset a working tree.
`prism-cpu-mtp` v1 touches disjoint files; composition is possible, but a combined
full-runtime build/test remains required. Do not also stack its older greedy patch.

## Implemented source changes

- Optional `ggml_vec_dot_ptq1_0_q8_0_batch`: 2–5 Q8 activation columns share one
  bounded PTQ1 payload load/unpack per 128-weight block. Five columns accommodate
  up to four draft tokens plus a verification token where that graph batches them.
  This operates on existing batched matmuls; it does not change MTP scheduling.
- One weight row is traversed across its blocks, applying each decoded block to
  all columns. Only five AVX2 code vectors plus eight final trits are retained
  locally, with five FP32 totals. No full tensor/row expansion or second weight copy.
- Four original Q8 scales and the PTQ1 FP16 scale remain distinct. Integer sums
  cannot saturate at the supported range, including signed activation `-128`.
  Per-column floating scale/reduction order is preserved in source. Compiler flags
  and end-to-end numerical behavior still need target verification.
- Matmul dispatch is limited to PTQ1/Q8, AVX2, ordinary 2-D tensors, 2–5 total
  columns and 2–5 columns in the current chunk, one-row dot traits, and FP32 output.
  Supports quantized activation column padding and output strides. Reference
  requests, broadcasts, other dimensions/types, larger prefill batches, one-token
  chunks and unsupported CPUs retain the original dispatcher. Activation conversion,
  barriers, ownership, Hadamard transforms, recurrent checkpoints and rollback remain.
- `GGML_PTQ1_ENABLE_BATCH` is **absent/off by default**. It must be defined consistently
  for the CPU C translation units to test batch reuse. `GGML_PTQ1_DISABLE_AVX2`
  restores the pinned baseline dispatch even if batch is requested. Otherwise the
  cumulative patch includes the previous single-column AVX2 path.
- Vulkan/Intel iGPU sources, placement and prefill routes are untouched. CPU reuse
  cannot accelerate a matmul dispatched to Vulkan. No CUDA dependency is introduced.

## Exact LUT investigation

`ptq1-exact-lut.h` is a **comparison prototype only**, not runtime-dispatched.
It builds exact signed-int16 subset tables for groups of four Q8 activations and
subtracts negative-trit subsets from positive-trit subsets. All packed bytes,
including noncanonical encodings, use the pinned base-3 interpretation. Original
scales are applied in the same order. There is no table requantization, approximate
averaging, weight rewrite, persistent weight repack or offloading.

The table occupies 1024 bytes plus 16 bytes of scales per 128 activations, versus
136 bytes of Q8 input. Construction is amortized across weight rows in the supplied
benchmark, but lookup/index construction and cache pressure can outweigh saved
arithmetic. It is scalar and not a tuned T-MAC implementation. Its result cannot
establish a general T-MAC speed limit. A future PSHUFB/repacked-row implementation
would need its own exactness, extra-RAM and cache-footprint evidence before adoption.

## Checks performed, and target gates

`verification.json`: patch apply/output hashes/reverse round-trip; installer syntax;
actual patched x86 dispatch and complete matmul chunk function compiled with a
minimal GGML type scaffold in four flag variants; C++ parity-test and benchmark
objects compiled. This is not a full GGML/runtime build or a linked integration test.

The following commands are **future target instructions**, not executed here.
Use AVX2 hardware after explicit permission for target tests. Do not use fast-math,
disable assertions, or assume a compile pass certifies numerical parity.

```sh
c++ -std=c++17 -O2 -mavx2 -ffp-contract=off -fno-fast-math test-reuse.cpp -o test-reuse
./test-reuse
c++ -std=c++17 -O2 -mavx2 -ffp-contract=off -fno-fast-math benchmark.cpp -o benchmark
./benchmark 32768 8192 5 3
```

`test-reuse` checks baseline/single-AVX2/batch/LUT bitwise outputs for 0/1/3/16
blocks, 2–5 columns, padded/unpadded Q8 and padded output, finite positive/negative
half scales, encoding bytes and activation extremes, random inputs and sentinels.
It does not replace guard-page/ASan/UBSan or upstream backend tests. It does not
exercise GGML's full scheduler, F32-to-Q8 conversion or model logits.

`benchmark` checks whole-result parity before timing, rotates kernel order and
includes LUT construction time in every LUT pass. It is single-threaded. Default
weights occupy 58,720,256 bytes; also sweep smaller cache-resident and actual model
shapes, columns 2/3/4/5, six/eight/etc. threads through the full runtime, and sustained
thermals. Report medians/variation; this is not model tokens/second. At K=8192 the
LUT allocation is 66,560 bytes/column, 332,800 bytes for five columns.

For the full runtime, compare separate clean builds: pinned SSE/SSSE3; v2 single
AVX2 with batch off; v2 AVX2 with batch on; then each with the existing CPU-MTP patch
if selected. Example C definition: `-DGGML_PTQ1_ENABLE_BATCH` in C compiler flags.
Do not force AVX2 on other machines; preserve runtime CPU feature dispatch.
Use upstream backend matmul tests including odd output rows, column strides,
broadcast/fallback, empty chunks and reference requests; instrument batch call
counts to confirm graph eligibility. Check logits and greedy tokens at matching
positions, all-accepted/rejected draft cases, recurrent replay, and tool/action
outputs. Keep the same existing head and model bytes/hash.

Measure CPU decode and Vulkan prefill separately at fixed thread settings, context,
cache type, prompt and output budget. Warm residency, compare MTP off/on, and record
accepted/generated/drafted tokens, verification time, TTFT, task latency and peak
RAM/swap. MTP is useful only when whole-cycle accepted-token throughput improves.

Actual DRAM bytes need available platform memory-controller read/write counters
(Intel PCM or supported Linux perf uncore events), an idle baseline and generated
token counts. LLC misses or `weights_size * forward_calls` are not DRAM-byte
measurements. The pinned dispatcher already tiles 16 weight rows and 16 columns;
its repeated payload accesses can hit cache. Do not claim a fivefold bandwidth gain
from five-column reuse. Reject regressions or token differences; keep fallback.


## Research attribution

[T-MAC](https://arxiv.org/abs/2407.00088) informed the locality/reuse and exact-LUT
comparison work. This is an independent source adaptation, not its full kernel,
approximate table arithmetic or a replicated speed result. The scalar exact LUT
is a prototype; the production batch candidate reuses unpacking across columns.
The [engine references](../../README.md#decode-research-roadmap-and-paper-references)
also cite FlashAttention, Flash-Decoding, FlashInfer and LLM in a Flash, with their
precise existing-upstream, future-work or rejected-path status. No new attention
kernel, predictive loader, CUDA dependency or dense offloading was added here.
The [new structure roadmap](../../docs/ENGINE_DECODE_STRUCTURE_ROADMAP.md) plans
seven further paper adaptations; it does not change this patch or activate it.
