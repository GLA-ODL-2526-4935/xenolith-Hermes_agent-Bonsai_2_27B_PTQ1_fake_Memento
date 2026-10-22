# Greedy MTP candidate patch

22 October 2026. Source-only candidate; **not compiled, installed into a runtime, or benchmarked**. No model was loaded. This is a patch for the external Prism-derived runtime, not native Xenolith C/Level Zero support.

Exact base: [sudoingX/llama.cpp ff414120c343e6e6cb868013c99f1dde52b27e70](https://github.com/sudoingX/llama.cpp/tree/ff414120c343e6e6cb868013c99f1dde52b27e70). It does not apply automatically to Prism release 73bfe6c or an installed executable. The published optimized Windows binary reports a different build identifier; source inspection does not certify that binary. Retain upstream licensing when building/distributing a runtime.

## Changes

- `common/speculative.cpp`: when draft `p_min == 0`, replace top-10/distribution sampling with the existing greedy sampler, use its selected token directly, and skip confidence probabilities, candidate sorting and draft RNG. Preserve CPU model suppression/logit-bias samplers. Backend sampling uses its existing argmax implementation when supported; normal CPU fallback remains. All nonzero confidence thresholds retain the original path. Draft selection changes tied-logit ordering potentially; target verification still decides acceptance.
- `tools/server/server-context.cpp`: clone the target sampler only when checkpoint/replay could occur. The original rule for cache restore, accepted-prefix state, recurrent rollback and draft synchronization is unchanged. Since `accepted.size() >= 1`, the actual rollback is at most `n_draft`; therefore a bounded recurrent window covering the entire draft cannot need replay. Full-checkpoint contexts still retain the clone.

The launcher sets temperature0, top-k1, top-p1, min-p0, repeat penalty1, seed3407, sampler chain `top_k;temperature`, in both server defaults and owning-client defaults. The target's existing single-candidate distribution path already skips softmax (it still advances the RNG). There is no new global replacement of grammar, model suppression, repetition, reasoning-budget or log-probability semantics. Other clients can override sampling defaults; this is an opt-in controller configuration, not an endpoint enforcement proxy.

## Existing work retained

Target verification already decodes a batch of candidate positions; this allows the matrix kernels to reuse weights across positions. It still computes each position's hidden state and vocabulary scores. Argmax removes normalization/sampling work, not those matrix products. No duplicate full model or new weight copy is introduced by this patch. Actual shared allocations/RSS still require measurement.

The Qwen-derived hybrid model has recurrent state as well as attention KV. The runtime already has bounded per-token recurrent snapshots and falls back to checkpoint/replay outside the window. Removing that state or treating rollback as a KV pointer decrement can corrupt subsequent answers. This patch does not change rollback depth, state allocation, cache precision, or memory ownership. It removes only an unnecessary *sampler* copy, not a full KV/state copy.

## Check and apply, source only

Use a separate checkout of the exact base commit. From this directory:

```sh
python3 apply.py /path/to/pinned/llama.cpp
python3 apply.py /path/to/pinned/llama.cpp --apply
```

The first command is check-only. The second requires the exact commit and both source-file hashes, then uses `git apply`; it does not reset changes, fetch, configure, build, launch, or read model files. Changes to other source files are not certified by this check. Record the full resulting source diff and build dependencies. Build only the CPU/Vulkan target with matching libraries on the intended Linux toolchain; do not replace a working runtime until acceptance passes.

## Required target acceptance

1. Compile the patched source, run upstream sampling tests and verify CPU and available Vulkan sampler backends. Check normal and suppressed/tied logits, EOS, grammar/JSON constraints, reasoning-budget transitions, and logging/log-probability requests. Verify supported backend sampling falls back cleanly when unavailable; CPU and backend suppression behavior need parity testing.
2. Compare no-MTP, original-MTP and patched-MTP **token IDs** on identical prompts/settings/model/cache types. Begin `n_max=1`, then2/3/4, p_min0; regression-test nonzero0.3/0.5/0.7 separately. The earlier Windows n_max3 result is a candidate, not the Intel default. Profile remains one-token Vulkan, CPU MTP off until measured.
3. Force reject-first/middle/last, all-accepted, EOS, cancellation, short/long context and repeated cycles. Exercise recurrent window equal to/above/below draft length, plus full-checkpoint replay. Compare continuation tokens and recurrent/cache state against ordinary decoding; confirm replay always has a sampler clone.
4. Measure sampler CPU time, draft/verify/rollback time, accepted/drafted, tokens per verification cycle, prefill/decode and whole-task latency, peak shared RAM, swap and steady-state thermals. Keep >=5GiB available RAM. No speedup is claimed from static checks; fall back to the pinned original runtime if any correctness or lifecycle check fails.

Optimization decisions for kernel tiling/weight reuse and recurrent snapshot copying require these profiles on Intel. No CUDA-specific optimization is presented as an Intel improvement.
