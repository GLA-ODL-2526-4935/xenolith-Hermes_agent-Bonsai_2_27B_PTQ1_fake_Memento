# Intel draft-source and PTQ1 schedule candidate

23 October 2026. **Inactive source package.** No engine/model/native test was run,
and no inference speedup or model parity is established. This is a reversible
delta on **intel-decode-v3, then intel-decode-increment-v1**, both at
sudoingX/llama.cpp `ff414120c343e6e6cb868013c99f1dde52b27e70`.
It is not another cumulative replacement. Do not stack the older replacement
PTQ1/greedy-MTP packages with v3. Reverse this delta before its prerequisites.

## What is implemented

- One request-local source/window policy in the server: warm off, MTP1..4 and
  optionally scored suffix-chain1..4. It replaces the existing adaptive-window
  policy when enabled. Single slot, only the MTP implementation, draft maximum
  1..4 and minimum <=1 are required; incompatible startup settings throw.
- Aggregate committed tokens / aggregate elapsed microseconds, two off baseline
  samples, two samples before promoting a candidate, >=5% measured advantage,
  a bounded optimistic bonus, and one deterministic exploration opportunity per
  16 completed cycles. Off is periodically rechecked too. No extra sampler RNG.
  This is a practical cost-aware adaptation, not BanditSpec's formal algorithm or
  a promise of zero exploration slowdown. Statistics reset on request begin and
  4096-token context buckets; no cross-request tuning is retained.
- Timer begins before checkpoint/source work and stays open through reject
  restoration and replay. Unique delivered tokens are credited once, including
  EOS/output-stop truncation. Draft catch-up is synchronized before the timer
  closes. Both ordinary and speculative paths close after output processing.
  Per-cycle debug telemetry contains arm, tokens, elapsed time and context bucket;
  aggregate statistics cover the current bucket. Failed/cancelled unfinished
  cycles are censored when the next request begins. No new rollback algorithm.
- A suffix source scans the current committed token stream, bounded to 8191 prompt
  tokens plus the last token. It selects the longest repeated suffix of length
  2..16 and majority-scored conditional continuations, truncating below 50% support
  and at four drafts. Every draft passes the unchanged target verifier. A miss
  performs ordinary decoding and its full lookup cost remains charged to the
  suffix arm. Scratch is bounded to about 128KiB plus vector objects/output on
  a 64-bit target; no index, persistent history, second model or hidden reasoning
  store is retained. No cross-request or fabricated cross-record matches.
- Warm MTP processing and existing checkpoint/accept/is_other hooks remain active
  when suffix wins. Ownership is cleared before a new source decision. Source
  switching occurs only at committed boundaries; replay retains its original arm.
  **Warm off still pays head maintenance.** Cold off/catch-up suspension is deferred;
  compare with a separate process configured without MTP to reveal that overhead.
- QIGen-inspired bounded schedule dispatch selects existing exact packed PTQ1
  dot, activation-column reuse, LUT8-row or LUT16-row schedules. Weight format,
  four activation scales and per-output FP reduction order stay unchanged.
  An offline recommendation tool accepts measured correctness and repeated timing
  records; it does not run benchmarks or activate anything. This is not QIGen's
  GPTQ code generator or a new generated SIMD kernel. LUT8 still uses the existing
  16-lane kernel and can lose to LUT16; no speed benefit is assumed.

## Independent controls

All controls are off/unset by default. Keep environment/build/thread/phase settings
fixed for each run; restart and reset statistics after changing them.

| Control | Effect |
| --- | --- |
| `LLAMA_SPEC_SOURCE_POLICY=1` | Enable server source arbitration for exactly temperature 0 requests. Other temperatures use legacy MTP; they are not validated selector modes. |
| `LLAMA_SPEC_SUFFIX=1` | Make suffix arms available to the controller; no lookup when an off/MTP arm is chosen. Requires source policy for any effect. |
| `LLAMA_SPEC_SOURCE_FORCE=0..8` | Optional A/B override: 0 warm off, 1..4 fixed MTP window, 5..8 fixed suffix window 1..4. Suffix arms require the suffix switch. An arm exceeding the current context/output cap becomes off. Unset for adaptive selection. |
| compile `GGML_PTQ1_ENABLE_QIGEN` | Include optional schedule dispatch and batch/LUT entry points. Add the define consistently to CPU C/C++ compile flags, not as an invented CMake option. AVX2 remains required for these kernels. |
| `GGML_PTQ1_QIGEN=1` | Enable schedule reading in a build containing that define. `0`/unset restores previous compile-time dispatch. |
| `GGML_PTQ1_QIGEN_SCHEDULE=12344` | Example only: exactly five digits, indexed by actual chunk column count 1..5. 0 previous dispatch; 1 dot; 2 batch reuse; 3 LUT8; 4 LUT16. Batch at width1 falls back to dot. Invalid/missing schedule restores prior dispatch. This example is not a measured recommendation. |

`GGML_PTQ1_PAIR_LUT_COMPACT` and `LLAMA_MTP_COMPACT_VERIFY` remain separate earlier
switches. QIGen tuning works with either compact-LUT setting; compare separately.
Schedules are parsed once per CPU matmul chunk, without mutable static caches,
per-token calibration or background tuning. Account for this overhead in runtime
timing. Reference requests, non-AVX2, disabled AVX2, wide/higher-dimensional and
ineligible strides keep their fallbacks. Intel Vulkan prefill routing is untouched.

## Source installation and rollback, later

Use a separate exact pinned source checkout and the prerequisite package helpers.
This package's helper verifies the pin and all prerequisite source hashes plus
the header before changing anything:

```text
python apply.py /path/to/exact/prerequisite/source
python apply.py /path/to/exact/prerequisite/source --apply
python apply.py /path/to/exact/prerequisite/source --reverse
```

The first command checks only. A full build, native tests and model acceptance
are required before any live profile selects these paths. Keep a known-good
binary/profile for rollback. No startup script/profile is modified by this package.

## Validation and later measurements

Here: seven Python offline tool/contract tests pass. Patch application, hashes,
reversal, drift and duplicate-application checks pass in disposable fixtures
(only their nonexistent pinned HEAD query is mocked). The actual generic source
hooks and complete CPU matmul chunk compile, including seven flag/ISA variants.
Kernel objects, policy/lifecycle tests, QIGen environment/tile parity tests and
benchmark objects compile with strict FP options. Native artifacts are **not
executed**. API scaffolds do not certify the complete runtime or backend state.

Target gates:

1. Build the full runtime, inspect linkage/dispatch and run upstream tests plus
   `test-source-policy.cpp`, `test-qigen.cpp` and prior kernel tests. Compile
   strict FP (`-ffp-contract=off -fno-fast-math`) and test compact flags separately.
   Use sanitizers/guard pages for memory edges. Test both 8/16 row tiles, strides,
   all finite supported scales and output tails; compiled tests are a starting set.
2. Verify all source transitions, acceptance prefixes0..4, miss, EOS, cancellation,
   output/context caps, sequence reuse/corrections and full recurrent checkpoint
   replay. Compare logits/greedy tokens/actions with fixed MTP and true MTP-off;
   matching greedy output is not established merely by target verification.
3. Run existing n-gram retrieval independently against suffix; measure incremental
   benefit rather than assuming this scored scan wins. Compare forced0, forced MTP,
   forced suffix, adaptive MTP-only, adaptive with suffix and true head-disabled.
   Hold Sandwich pools/affinity/backend sampling/target questions fixed.
4. For QIGen, `benchmark-qigen.cpp` adds kernel5=LUT8 to the prior benchmark:
   kernel1=AVX2 dot (mode1), kernel4=column reuse (mode2), kernel5=LUT8 (mode3),
   kernel2=LUT16 (mode4). Include LUT setup and representative shapes/cache states.
   It verifies finite synthetic output against baseline when executed, and emits
   `qigen_case=` JSON records with all raw sample times for the offline tool. Single
   threaded microbenchmarks cannot establish the best full-runtime thread schedule.
5. Feed `qigen-select.py` a JSON file with a nonempty `target_key` recording actual
   CPU/build/flags/threads/shape/cache/thermal conditions and `cases` entries:
   `{ "columns": 2, "mode": 4, "correct": true, "times_us": [70,71,70,72,70] }`.
   Supply baseline mode1 for every column1..5; at least five positive finite
   measurements each. Only measured-correct candidates with >=5% median gain and
   no p90 regression can win. It prints a recommendation with `activated:false`.
   Gather separate profiles for materially different row/K sizes and topology;
   the current runtime map is column-only, not automatic shape/topology detection.
6. Measure repeated whole-task latency, actual DRAM traffic, lookup cost, spills,
   cache misses, RSS/shared RAM/swap and sustained thermals on the real Intel P/E
   laptop. Preserve iGPU prefill independently. No speedup is claimed here.

## Paper attribution and deferred work

- [SuffixDecoding](https://arxiv.org/abs/2411.04975): occurrence-scored retrieval
  informs our bounded single-chain scan. Full suffix tree/index/tree verifier not ported.
- [BanditSpec](https://arxiv.org/abs/2505.15141): online speculative-configuration
  selection informs our cost-aware controller; no round-based regret claim.
- [QIGen](https://arxiv.org/abs/2307.03738): bounded empirical microtile/kernel
  selection informs the optional schedule path; no GPTQ conversion/codegen.
- [FlashNorm](https://arxiv.org/abs/2407.09577): existing norm/scale fusion audited;
  no new epsilon/quantization-sensitive folding or normalization movement.
- [STree](https://arxiv.org/abs/2505.14969): exploratory only; no tree recurrence
  or reassociated recurrent state evaluation added.

Earlier kernel and scheduler papers remain cited in the engine README and previous
packages. This delta reuses those implementations rather than counting them again.
Cold off, multi-slot arbitration, stochastic selection, a suffix index, hardware-
discovered cache/register code generation and full-runtime acceptance remain open.
