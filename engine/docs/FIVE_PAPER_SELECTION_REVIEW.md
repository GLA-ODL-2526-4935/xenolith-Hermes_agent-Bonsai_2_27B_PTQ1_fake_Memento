# Draft selection and CPU kernel review — 22 October 2026

Original research/design review; updated source status below. During the review,
no inference code, live profile, model/head, runtime binary,
GPU or WSL changed or run. Applies to the external Bonsai Prism backend and its
inactive source candidates, not the historical native Xenolith Gemma loader.

## Decisions

| Priority | Paper | Decision for this engine |
| --- | --- | --- |
| 1 | BanditSpec | Add a single cost-aware selector with ordinary decoding as an explicit arm; replace, rather than nest, the existing MTP-window tuner. |
| 1 | SuffixDecoding | Add a bounded suffix-derived single-chain draft source complementing the existing community MTP head; compare against existing n-gram lookup first. |
| 2 | QIGen | Transfer cache/register cost modeling to bounded PTQ1 kernel tuning; retain current weights, scales and floating reduction order. |
| 3 | FlashNorm | Audit adjacent fusion only. Existing norm/scale fusion overlaps part of the goal; reject automatic folding, norm cancellation or deferred-quantization changes. |
| Exploratory | STree | Defer tree execution until a Gated DeltaNet-compatible recurrence, exact accepted-path restoration and Intel kernels are established. |

These are architecture-based priorities, not measured winners. First establish the
current candidates' native numerical correctness and full-runtime acceptance.

## Paper mechanisms and compatibility

### SuffixDecoding

[Paper](https://arxiv.org/html/2411.04975), §3, AppendixB.1: suffix trees retrieve
continuations from current prompts/outputs and historical outputs. Occurrence
counts score candidate trees; match length controls speculation budget, and a
score threshold permits a model-based fallback. This is well matched to repeated
structured agent output, but not a guarantee that our particular output repeats.

Select a **single-chain adaptation**, initially at most4 draft tokens, not a full
reproduction of its tree method. This fits the existing1–5-column verification
candidate including the target token. Compare its incremental hit/acceptance
benefit with current n-gram implementations; do not build another index merely
to rename existing lookup. A bounded session cache is required on16GB shared RAM;
the paper's server memory assumptions are unsuitable here. Start request-local;
later reuse only already-retained visible outputs under the controller's existing
retention rules. Do not silently preserve discarded reasoning in a second cache.

### BanditSpec

[Paper](https://arxiv.org/html/2505.15141), §§3–6, AppendixB: training-free UCBSpec
and EXP3Spec choose speculative configurations online. Its formal stopping-time
objective counts decoding rounds and their token progress; do not transfer its
guarantees directly to unequal CPU wall-clock costs. Our selected adaptation uses
measured committed tokens per complete elapsed cycle and an explicit off arm.

This addresses a concrete missing feature: current window policy chooses only1..4.
Use one selector for source and length, with bounded exploration and ordinary
decoding as fallback. UCB-style confidence is a practical starting candidate;
discounted observations, variable costs and change detection are engineering
extensions, not a claimed implementation of the paper's theorem. EXP3 is an
alternative for measured nonstationarity, not a second concurrent tuner. Keep
the existing MTP head; no new draft model, teacher or training is required.

### FlashNorm

[Paper](https://arxiv.org/html/2407.09577), §§3,6–7: folds normalization weights
into projections and/or defers scalar normalization. Deferral can be used without
weight folding. Norm cancellation has an explicit epsilon caveat. Algebraic
equivalence does not establish identical rounded execution through quantization.

Do not fold per-channel factors into fixed packed PTQ1 weights: arbitrary input
factors generally cannot use the existing per-block scale unchanged. Deferring
normalization across Q8 activation quantization or Hadamard transforms needs
separate rounding/graph proof. The pinned CPU already has eligible F32 RMS_NORM+MUL
fusion, so implementing that again adds nothing. Investigate only genuinely
adjacent, unfused norm/activation preparation, preserving epsilon, sum order,
intermediate F32 rounding and quantizer decisions. Potential benefit is narrower
than a whole FlashNorm port; Intel iGPU prefill needs independent evidence.

### QIGen

[Paper](https://arxiv.org/html/2307.03738), §3: generates CPU quantized GEMV kernels
using cache-sized blocks, register-resident output accumulators and empirical
microtile selection. Its GPTQ bit-packed integers/zero points are not PTQ1 base-3
blocks with four Q8 activation scales. Transfer the tuning method, not its format.

Select bounded comparisons among existing dot, batch-reuse and exact-LUT paths,
with per-shape row/column tiles and real P/E-core cache/register information.
Include scratch, lookup tables, packed blocks and activation bytes in cache cost;
inspect spills and false sharing. Existing row tiling/reuse overlaps the concept.
No global Morton/Z-order repack, expanded full-weight cache, FP reassociation or
automatic ISA upgrade is selected. A future lossless layout must retain one
resident representation usable by the existing Intel Vulkan route.

### STree

[Paper](https://arxiv.org/html/2505.14969), §3: uses tree masks, accumulated SSM
transitions, cached intermediate activations and accepted-path replay. Its
derivation uses diagonal transitions and its implementation exploits GPU shared
memory. This is relevant research, not a drop-in tree mask for our hybrid model.

Bonsai's Qwen3.5-derived Gated DeltaNet transition is not the paper's simple
diagonal recurrence. Establish a compatible derivation, convolution history,
attention positions and recurrent accepted-path restore first. Tree evaluation
may change floating reduction order and exceed the current skinny-kernel window.
Do not expand the community head into a tree or import CUDA kernels on that basis.
Defer behind the existing Sequoia/ReplaySSM research gates; keep exact chain
checkpoint/rollback and current Intel architecture.

## Source audit: what already exists

Source pin: sudoingX/llama.cpp `ff414120c343e6e6cb868013c99f1dde52b27e70`
plus the exact `intel-decode-v3` source package; optional local-work increment
does not change the selection conclusions. Research evidence hashes are saved
with this review's workspace outputs. No installed binary behavior is inferred.

- `common/speculative.cpp`, `common_speculative_init` and
  `common_speculative_draft`: multiple n-gram sources precede MTP in a fixed
  preference order. First nonempty drafts win; this is not wall-time arbitration
  or SuffixDecoding's scored suffix-tree algorithm.
- `common/mtp-cpu-policy.h`: four initial observations per window1..min(4,cap),
  EMA microseconds/(accepted drafts+1), periodic probe every32 valid cycles. No
  zero/off arm, source choice or UCB/EXP3 exists. Shortened cycles are discarded.
- `common_speculative_process`: all enabled implementations process target batches.
  Keeping MTP available may therefore retain its catch-up/head cost on lookup/off
  cycles. An off choice inside `draft()` alone is not a true MTP-off baseline.
- `common_speculative_accept`: the chosen implementation gets its acceptance hook;
  others get `is_other=true`. State serialization has an explicit TODO for multiple
  stateful implementations. Preserve the ownership/lifecycle contract when switching.
- `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_cpu_try_fuse_ops`: eligible F32 RMS_NORM+MUL
  fusion already exists; reference/disabled-fusion paths bypass it.
- `tools/server/server-context.cpp`: bounded recurrent removal and checkpoint/replay
  differ. Replay may return before the final acceptance hook. An elapsed sample
  closed at the first verification pass would undercount restoration/replay work.

The older saved Qwen35 graph at Prism73bfe6c shows gated recurrent layers and
normalization/transform paths, but is not proof of the current pin's entire graph.
Any implementation must re-audit the exact active model/backend and descriptor
paths before adding graph fusion or a new recurrence.

## Unified selection design — original proposal; source status below

1. **Arms:** ordinary decoding; MTP windows1..4; available suffix-chain windows1..4.
   Clip all to current output/context/microbatch and recurrent rollback limits.
   Initially validate greedy temperature0 only. Keep all drafted tokens subject to
   ordinary target verification; historical matches never bypass the verifier.
   Stochastic support needs its own acceptance/distribution/RNG tests.
2. **Eligibility and lookup cost:** bounded exact token IDs, no matches across
   unrelated record boundaries, explicit cache byte/token limits and eviction.
   Score a chain for eligibility, not correctness. If the selector skips retrieval,
   perform no lookup. Charge probes that find no match, index maintenance and
   scoring to measured overhead; do not pretend a zero-token draft costs nothing.
3. **Reward:** aggregate `sum(committed_tokens)/sum(elapsed_us)` per arm within a
   controlled context bucket; also retain cycle variance and latency. Time from
   before lookup/source work through successful verification, reject restore/replay,
   state catch-up and committed boundary. Count unique progress once, never replayed
   tokens twice. Compare ordinary decoding using the same timing boundary. Zero
   progress/errors are not successful reward samples; cancellations are censored.
4. **Safe switching:** change sources only at a committed boundary; clear stale
   implementation/choice records and reconcile target/draft hidden, KV, recurrent
   and convolution state, sampler/RNG and positions. Correct `is_other` hooks remain
   necessary. Choose explicitly between warm off (MTP maintained, charged) and cold
   off (MTP suspended, later catch-up charged). Do not skip lifecycle hooks blindly.
   Initial implementation may use warm off, but must label it and compare against
   a separate true MTP-disabled run to expose its residual overhead.
5. **One controller:** replace the existing window tuner when enabling arbitration.
   Keep Sandwich phase plans fixed during a selection experiment. Bound exploration
   on short completions, cool down losing arms and periodically recheck them. On
   sparse/noisy evidence prefer off or the established baseline; no guarantee of
   zero slowdown during exploration. Never change sampling to improve acceptance.
6. **Attribution/reset:** key statistics by model/head/tokenizer hashes, build/flags,
   CPU/iGPU/thread phase plan, context/KV bucket, sampler and cache policy. Reset on
   incompatible changes; start request-local. Any cross-request tuning/index reuse
   is explicit and bounded, not silent persistence of discarded controller context.

## Implementation and acceptance order

**F1:** source-level telemetry/ordinary-decoding arm first; distinguish warm/cold
off, close replay timing correctly and verify request/sequence reset. Compare
fixed MTP, present tuner and true MTP-disabled baseline before adding retrieval.

**F2:** benchmark existing n-gram first, then a bounded scored suffix-chain source.
Measure lookup misses/cost, cache growth and accepted tokens on realistic Hermes
JSON/tools/memory outputs plus nonrepetitive controls. Increase chain length above4
only after native verification/recurrent/shape tests support it; no tree required.

**F3:** integrate cost-aware joint arbitration after both sources are independently
correct; test every source transition, no-match,0..4 accepted prefix, EOS, output
caps, sequence reuse, cancellation, prompt corrections and full checkpoint replay.
Compare logits/greedy tokens/actions and whole-task latency with repeated runs,
memory/shared-RAM/swap and sustained thermals; include lookup and transition costs.

**F4:** bounded QIGen-style kernel schedule search after native parity of existing
PTQ1 candidates. Tune decode/short verification separately from iGPU prefill.

**F5:** audit genuinely unfused adjacent normalization preparation; preserve exact
quantized behavior. **F6:** STree remains research-only pending a compatible proof
and Intel implementation. No paper speedup is a prediction for this laptop.


## Source status update — 23 October 2026

[Source package](../patches/prism-source-selection/README.md), checklist4D, implements
inactive F1/F2/F3 adaptations: warm-off full-cycle accounting, bounded scored suffix
chains and joint deterministic cost-aware selection. Initial scope is one server
slot, temperature0, unchanged verifier/state hooks, request/context-bucket reset.
An explicit forced-source control supports later fixed-arm comparison. Timing spans
restoration/replay and synchronized warm head catch-up; cold off remains deferred.
The suffix source is a bounded scan, not a suffix-tree index. Existing n-gram
performance comparison is pending, so no superiority is asserted.

F4 adds a separately switched bounded PTQ1 schedule and offline recommendation
tool using existing exact kernels/row tiles. It does not implement GPTQ codegen,
hardware-discovered cache/register modeling or a new weight representation. Native
spills/cache/DRAM/topology evaluation and actual speed selection are pending.
Seven offline tests and source/compiler checks passed; native parity tests were
compiled only. No engine/model/binary/profile activation occurred. F5/F6, multi-slot
and stochastic verification remain open; no full-paper or numerical/speed claim.
