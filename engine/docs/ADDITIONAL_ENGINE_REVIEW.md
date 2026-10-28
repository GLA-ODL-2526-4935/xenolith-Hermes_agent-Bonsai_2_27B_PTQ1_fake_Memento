# Additional open-source engine structures

Reviewed 23 October 2026. Six repositories beyond the earlier Arcint,
llama.cpp-ng, ik_llamacpp, SGLang, LLM.xpu and MNN review. This is a targeted
source review, not a complete audit or a ranking by published speed.

## Result

Implement **request-owned, bounded suffix scratch reuse and a synchronous borrowed
history view**, behind `LLAMA_SPEC_SUFFIX_WORKSPACE=1`. These remove work visible
in our actual code rather than adding another PTQ1 kernel, cache or thread pool.
Leave the other structures as specific acceptance/profiling items where existing
implementations overlap or backend correctness is not yet established.

| Engine | Structure inspected | Comparison with our implementation | Decision |
|---|---|---|---|
| MLC-LLM | Bounded speculative workspace, explicit slot allocation/release and reference ownership | Existing MTP buffers/rollback already have owners; suffix lookup still allocated four vectors per call | Adapt ownership/lifetime locally to suffix scratch; no TVM, probability tensors or speculative trees |
| OpenVINO | Reusable memory blocks, explicit release and lifetime-aware memory control | GGML already has graph allocation reuse and the new reservation cap; suffix host scratch is outside that graph allocator | Implement local reuse/release; do not add a second general arena or switch backend |
| ONNX Runtime | Execution-frame memory patterns, allocation reuse and fallback; bounded arena extension/shrinkage | Full graph-pattern allocator would duplicate GGML; request-scoped scratch has a fixed small maximum | Borrow the planning/ownership principle; no global allocator dependency |
| XNNPACK | First/last-use lifetime tracking, allocation offsets and shared workspace ownership | GGML already schedules tensor lifetimes; our CPU tile work already has bounds and fallbacks | Use for alias/lifetime audit; no duplicate workspace planner |
| KTransformers | AVX2 native-lane activation ordering, shape-sensitive staging, bounded small-batch scratch | PTQ1 already converts activations and reuses weights for 2–5 columns, with LUT/QIGen alternatives | Profile shuffle/conversion cost before another kernel; MXFP4/MoE layout cannot be copied into PTQ1 |
| mistral.rs | Prefix owners with recurrent snapshots, independent capacity/byte accounting and eviction | Existing recurrent rollback and prefix-cache gates overlap, but native Xenolith features are not automatically Prism features | Track combined KV/recurrent owner accounting as a future backend task; no paged-cache migration now |

The decisions above are engineering judgments from the inspected code and our
current architecture. They are not claims that any engine's measured speed
transfers to this Intel laptop or this dense PTQ1 model.

## What was implemented

The legacy `common_suffix_chain` copies the last 8,191 prompt tokens plus the last
token into a new history vector. It also creates match, vote and output vectors
on each call. The new path keeps the same longest-match and conditional-majority
algorithm, but borrows committed prompt data during a synchronous call, reuses
per-request match/vote storage and writes into the caller's result vector.

No prompt pointer survives the call. Nothing is reused as token history across
requests. Scratch is allocated only on a useful suffix lookup, released by the
existing request-end hook, defensively released on begin and owned by the
speculative object's destructor. A save-profile error occurs after release and
does not keep scratch alive. This is a single-owner path, not a shared global pool.
The end hook is called from the existing server slot-release route.

Requested scratch is 96 KiB on the intended 64-bit/32-bit-token configuration,
plus at most four caller-owned output tokens and allocator overhead. This is less
than the old worst-case history+match+vote payload, though short requests now
reserve the bounded scratch maximum on first use. Freeing it does not promise an
immediate RSS drop or secure erasure of allocator pages. This is not a RAM quota.

All candidates still pass through the existing ordinary target verifier. The
algorithm preserves the minimum match length 2, maximum 16, history boundary,
width/order traversal, majority condition, exhausted-occurrence denominator and
token tie-break. The ordinary legacy helper remains the off-switch fallback.

The patch is additive after runtime-lifecycle in the exact composed Prism source.
Reverse this increment before reversing runtime-lifecycle or older packages.
Manifest guards cover HEAD, changed files and dependencies. No model, runtime
binary, profile default, PTQ1 representation, scale, Hadamard operation, recurrent
state precision, cache commit/rollback or Intel iGPU placement was changed.
The three-arm selector speed-test build remains separate.

## Verification

All five packaged compile/execution checks pass against the applied source:

- 836,772 exact comparisons with the actual legacy suffix helper, including
  exhaustive small histories, random histories through 17K tokens, retained-window
  edges, ambiguous continuations, ties, short/invalid caps and alias fallbacks.
- Independent workspace ownership exercised on four threads.
- After warm-up, zero temporary allocations across 1,000 repeated lookups per
  fixture, versus 4,000 allocations in each corresponding legacy fixture.
- Actual request begin/end functions executed with minimal host mocks, including
  sequence isolation, reallocation after release and profile-save exception cleanup.
- Full modified speculative.cpp translation unit compiles. This is not a linked
  runtime or a model/driver test.

The source package additionally passes real git apply/hash/reverse checks in a
fixture, wrong-HEAD/drift refusal, and an actual-source reverse applicability check.
The existing Xenolith performance-environment digest includes the new switch;
standalone benchmark wrappers must bind it into their settings identity too.

Illustrative warm-lookup timings from the packaged fixture on the Windows
preparation host, microseconds per call (1,000 calls; sequential single pass):

| History tokens | Pattern | Legacy | Reuse/view |
|---:|---|---:|---:|
| 128 | repeated | 0.71 | 0.30 |
| 128 | random | 1.85 | 0.50 |
| 2,048 | repeated | 8.80 | 6.50 |
| 2,048 | random | 24.09 | 6.30 |
| 8,192 | repeated | 42.21 | 29.13 |
| 8,192 | random | 94.79 | 24.79 |

These numbers measure just the suffix helper. They exclude model execution,
first allocation, selector transitions, verification, prefill and HTTP time. No
end-to-end tokens/s, actual DRAM-controller traffic, sustained Intel speedup or
quality improvement is established. No model/GPU workload was started for this
increment. The observed elimination is a history-vector copy, not a measurement
that all corresponding bytes would otherwise reach DRAM on every call.

## Remaining structures worth testing

1. **KTransformers-inspired native-lane staging:** first measure PTQ1 shuffle and
   activation-conversion cost separately at 1, 2–5 and prefill-sized columns. Reuse
   existing QIGen dispatch rather than add a competing tuner. Any lane mapping must
   retain all four original Q8 scale groups, Hadamard semantics and FP32 reduction
   order. Do not assume a threshold from a different format/model applies here.
2. **XNNPACK/ONNX/OpenVINO lifetime audit:** use our allocation/copy telemetry to
   find a concrete alias-safe opportunity beyond already implemented GGML planning.
   Do not alias output/activation/checkpoint storage across overlapping async work.
3. **mistral.rs-style joint owner accounting:** measure live KV plus recurrent
   checkpoint storage and protected owners in the actual Prism backend. Check
   correction/prefix invalidation, exact state identity and eviction rollback before
   proposing tighter cache limits. Do not present the graph-compute cap as total RAM.
4. **Target acceptance:** full linked Linux CPU/Vulkan build, real-model greedy
   output/logit parity, request cancellation/cleanup and cold/warm whole-task timing.
   Recalibrate selector costs on the target hardware/build/settings.

## Pinned primary sources and licenses

All implementations were inspected at pinned commits; no engine code was copied
into the new helper. Full fetched snapshots, tree indexes and metadata are saved
under `work/engine-patterns-2`; JSON catalogs record missing initial fetches and
corrected extra-source paths. Upstream license files were read.

- **MLC-LLM**, Apache-2.0, `8978ea9aec626d64096aaa92fbc89bda2e3ea802`:
  [draft workspace manager](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/cpp/serve/draft_token_workspace_manager.cc),
  [manager interface](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/cpp/serve/draft_token_workspace_manager.h).
- **OpenVINO**, Apache-2.0, `695d69efe543b9e6c75361bc5d3df9c866d9c6ed`:
  [CPU memory control](https://github.com/openvinotoolkit/openvino/blob/695d69efe543b9e6c75361bc5d3df9c866d9c6ed/src/plugins/intel_cpu/src/memory_control.cpp).
- **ONNX Runtime**, MIT, `84829e76d4fbb4a7258202f777fa4ff0876a3564`:
  [execution-frame memory patterns](https://github.com/microsoft/onnxruntime/blob/84829e76d4fbb4a7258202f777fa4ff0876a3564/onnxruntime/core/framework/execution_frame.cc),
  [bounded arena](https://github.com/microsoft/onnxruntime/blob/84829e76d4fbb4a7258202f777fa4ff0876a3564/onnxruntime/core/framework/bfc_arena.cc).
- **XNNPACK**, BSD-style, `26186f85939753b9a509734b193aad2bcb17518f`:
  [lifetime planner](https://github.com/google/XNNPACK/blob/26186f85939753b9a509734b193aad2bcb17518f/src/memory-planner.c),
  [runtime workspace](https://github.com/google/XNNPACK/blob/26186f85939753b9a509734b193aad2bcb17518f/src/runtime.c).
- **KTransformers**, Apache-2.0, `a694e718e8b3ec2e221277d674e369f3be38d2a4`:
  [AVX2 MXFP4 kernel](https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/kt-kernel/operators/avx2/mxfp4-moe.hpp).
- **mistral.rs**, MIT, `012b9bec941e62eecf6acc0f5a8a3e696f7bf906`:
  [prefix/recurrent cache](https://github.com/EricLBuehler/mistral.rs/blob/012b9bec941e62eecf6acc0f5a8a3e696f7bf906/mistralrs-core/src/prefix_cacher.rs).
