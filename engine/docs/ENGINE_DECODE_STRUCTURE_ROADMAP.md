# Engine decode structures: selection and compatibility

22 October 2026. Documentation/design only. Read with engine README references,
checklist section 4A, and `INTEL_INFERENCE_PAPER_REVIEW.md`. IDs below identify
work items. Intel decode v3 now supplies inactive source candidates for the three
priority papers; numerical and target-performance acceptance remains open.

## Target and baseline

The recorded target is an ASUS UX3402VA-KN521W, i9-13900H, Iris Xe-class iGPU,
16 GB shared RAM, Linux/Fedora KDE. Actual CPUID, core/cache topology, driver and
memory bandwidth must be obtained on that target, not this Windows preparation PC.
Design for stock AVX2; feature-dispatch BMI2/POPCNT or any optional dot instruction.
Do not assume AVX-512, AVX-VNNI-INT8, Xe2 instructions or Xe2 matrix hardware.

Bonsai uses the external Prism process. Native Xenolith's Gemma loader is not a
Bonsai backend. Preserve exact model/head values, GGUF hash, Hadamard/sign transforms,
four Q8 scales per 128 PTQ1 weights, FP32 scale/reduction order and fallback paths.
No retraining, new model, arbitrary sparse skipping or numerical approximation.
Lossless alternative packing is a derived representation, not permission to alter
weights/scales or overwrite the original GGUF. Account for its entire resident cost.

The pinned source is sudoingX `ff414120c343e6e6cb868013c99f1dde52b27e70`.
Existing source candidates: AVX2 unpack/dot and opt-in 2–5-column verification reuse
in `prism-ptq1-reuse` v2; cumulative CPU-MTP policy in `prism-cpu-mtp` v1.
Neither full runtime nor the new kernel has target numerical/performance acceptance.
The initial PTQ1 v1 and cumulative v2 are alternatives, not stackable patches.

## Why decoding benefits: our cost model

Single-token dense decode streams large weights for one short activation vector.
Prefill amortizes weights across many input tokens; MTP verification is an intermediate
2–5-column case. Lower bit-width helps only if unpack, quantization, cache misses,
synchronization and state management do not become the exposed bottleneck.

Use an approximate roofline, **our engineering model**, not a speed prediction:

`T_linear >= max(bytes_actually_transferred / sustained_bandwidth, exposed_compute_time)`

`T_MTP_per_committed_token = (draft_time + verification_time + replay/rollback_time + scheduling_time) / committed_tokens`

Measure bytes with memory-controller counters where available. Weight-file size,
load instruction counts and LLC misses alone are not measured DRAM traffic. End-to-end
task time includes prefill, reasoning, tool waits and short-answer warmup; kernel
speedup is neither whole-model speedup nor multiplicative with another paper's gain.

## Seven papers: mechanisms and selections

### D-B: BITCOS — first feasibility gate, conditional layout candidate

[Breaking the 1.58-bit Barrier for Ternary LLMs](https://arxiv.org/html/2609.16338v1),
§§II-A–C, III-C and Table I. A presence bitmap plus compact nonzero signs reduces
bytes when zeros are common; POPCNT/PDEP reconstructs signs. The paper's AVX2
contraction uses AVX-VNNI-INT8 and 128-element activation groups, not our four Q8 scales.

Our calculation: current PTQ1 costs 28 bytes/128 weights = 1.75 bpw. Holding the
16-bit scale fixed, ideal BITCOS costs `2.125-z` bpw; it beats PTQ1 only for `z>0.375`
before alignment, indices and padding. Byte-aligned per-block signs need the exact
calculation `16 + ceil(nonzeros/8) + 2 + index_overhead` bytes. Packing across blocks
reduces rounding but adds indexing/dependency costs.

The paper's Bonsai 27B row has 29.66% zeros: ideal 1.8284 versus 1.75 bpw, about
4.48% more bytes. This is a prior, **not a measurement of our exact Bonsai 2 Lean**.
Select five-trit PTQ1 as default. First measure per-tensor density and exact bytes;
test BITCOS only where packing savings or cheaper unpack could offset all costs.
No AVX-VNNI-INT8 requirement: use a stock-AVX2 exact integer fallback if prototyped.

### D-N: bitnet.cpp — selected exact software-kernel experiment

[Bitnet.cpp](https://arxiv.org/html/2502.11880v1), §§3.1–3.2. Element-wise ternary
LUTs exploit a smaller symbol alphabet; mirror consolidation and separated sign/index
streams fit byte-shuffle tables. Lossless TL1_1/TL2_1 split int16 tables into byte
planes, perform two lookups, then reconstruct int16, avoiding table requantization.

Select a TL1_1-style pair-table experiment first: register-local tables reused over
output-row tiles, with our PTQ1 input and Q8 boundaries retained. Compare against
the existing AVX2 integer dot, including table creation and code-to-index conversion.
TL2 triples require tails at each 32-element Q8 scale boundary; never combine scales.
Do not import I2_S's per-tensor activation quantization or TL*_0 approximation.

The paper's 5-bit/3-trit TL2 representation alone uses more symbol bits than our
26-byte/128 payload; it is not a free compression upgrade. Choose existing five-trit
storage over a whole-model TL2/I2_S repack initially. Promote a LUT kernel only if
its reduced arithmetic outweighs extra shuffles, indices and register spills.

### D-S: Sandwich — selected primary system structure

[Sandwich](https://arxiv.org/html/2507.18454v2), §§3–6. Distinct phase execution
plans, topology-aware partial-core allocation, and shape-specific microkernel
schedules reduce resource contention and poor skinny-matrix parallelization.

Adapt to one process and one resident model: three explicit phases—prefill,
one-token decode, and short MTP verification. Discover P/E classes, SMT siblings
and shared caches; do not copy symmetric/power-of-two server topology assumptions.
Search physical P cores, P+E groups and SMT choices with real sustained timings.
Existing separate `-t`/`-tb`/draft threads and affinity are foundations, not full
hot-switching. Switch validated pools/plans only at safe graph boundaries.

Select this ahead of new global weight layouts: it works with PTQ1, LUT or BITCOS
without altering numerical kernels or allocating another model. Preserve Vulkan
prefill as a separate candidate. CPU pool switching does not create automatic
CPU/iGPU phase migration. Full compiler/vLLM replacement and multi-process all-reduce
are not selected for this single-user laptop.

### D-R: TriTera/TriRun — retain division-free packing; GPU transfer conditional

[Scaling Laws and Efficient Inference for Ternary Language Models](https://aclanthology.org/2025.acl-long.1294.pdf),
§§3–4. TriTera is the model suite. CPU TQ1 uses multiply/shift trit extraction;
TQ2 trades more weight bytes for simpler decode. TriRun fuses packed INT2 decode
with register/shared-memory computation and pipelined loads on NVIDIA GPUs.

Our PTQ1 decoder already uses division-free extraction; credit this structural
overlap, do not add a duplicate decoder or assume TQ1 and PTQ1 byte layouts match.
Retain PTQ1 over global 2-bit expansion absent evidence. Study bounded tile staging
for Intel Vulkan prefill, but CUDA cp.async/tensor-core code is not portable to
Iris Xe. No TriTera checkpoint, training method or quantization scheme is substituted.

### D-F: FairyFuse — select sharing principle, reject full model-specific fusion

[FairyFuse](https://arxiv.org/html/2604.20913v1), §§3.1–3.2. Its AVX-512 masked
add/sub path fuses eight real sub-GEMVs of a complex widely-linear layer, sharing
activation/mask loads and retaining intermediate sums locally.

Bonsai is not that complex-valued graph; do not create eight artificial projections
or switch to FP32 masked sums. Audit actual same-input projections for already-fused
QKV/gate/up operations and rotation reuse before adding anything. Transfer only
compatible activation/code reuse and bounded local accumulators, extending current
MTP reuse where exact scale/reduction boundaries survive. Prefer the compact PTQ1
integer path over an AVX-512-only, 2-bit/FP32 replacement on this target.

### D-Q: Sequoia — select cost-aware budgeting, defer tree verification

Corrected source: [Sequoia](https://arxiv.org/html/2402.12374v3), §3, §4.1 and
Appendix F. The supplied `2402.06663` is an unrelated wireless-security paper.
Token trees amortize target passes over alternative drafts; dynamic programming
allocates a node budget and a hardware-aware optimizer includes verification cost.

Select its cost-aware objective for the existing community MTP head: choose by
committed tokens per entire cycle, including MTP-off, rather than acceptance alone.
Prefer the current short chain at temperature 0. Trees add siblings, masks and
branch-specific hybrid recurrent state; they also exceed current 2–5-column
specialization. Defer full trees until branching/replay correctness, memory and
benefit are demonstrated. Sampling-without-replacement requires exact distribution
correction outside greedy mode; it is not a drop-in argmax shortcut.

### D-T: T-SAR — select software dataflow experiment, reject unavailable ISA

[T-SAR](https://arxiv.org/html/2511.13676v1), §§III–IV. Dynamic register LUTs and
activation-/output-persistent dataflows reduce table traffic and writeback. Its
TLUT/TGEMV operations require microarchitecture/ISA changes evaluated with gem5.

Do not emit those opcodes or infer support from AVX2. Transfer only stock-instruction
AP-versus-OP tile experiments to D-N/Sandwich tuning. Keep accumulators within the
available register budget and preserve scale/reduction order. Select ordinary AVX2
software over the unavailable ISA path; simulated gains are not laptop forecasts.

## Compatibility and choices

| Competing/composable structures | Selected choice and structural reason |
|---|---|
| BITCOS vs fixed five-trit PTQ1 | PTQ1 default: low-zero prior, smaller payload, fixed offsets and existing CPU/Vulkan support. BITCOS is conditional on density, exact offsets/padding, measured unpack cycles and DRAM traffic; compression alone cannot decide. |
| TL2/I2_S full repack vs current PTQ1 | PTQ1 storage plus an exact lookup candidate. TL2/2-bit expansion raises bytes; quantization changes would violate correctness. Do not stack two formats on the same tensor. |
| Exact LUT vs AVX2 multiply/add | AVX2 remains fallback/default. Experiment with byte-plane exact LUTs; choose per measured shape/core/phase cost, never on "multiplication-free" wording. Neither is proven faster here. |
| Sandwich + PTQ1/LUT/BITCOS | Composable scheduling above a format-specific kernel; retune after any kernel change. One resident representation per chosen route. No two full model servers. |
| Sandwich + MTP adaptive controller | One joint policy, separate verification phase, include off, hysteresis and frozen evaluation. Independent competing controllers would confound timings and oscillate. |
| T-SAR ISA vs bitnet-style stock AVX2 | Choose stock AVX2. In-register AP/OP ideas can be composed with exact LUTs; hardware extensions cannot be installed in software. |
| FairyFuse full graph vs Bonsai integer dots | Keep Bonsai graph and integer dots. Share only genuinely common inputs/rotations/codes, not imaginary widely-linear layers. |
| Sequoia trees vs short greedy MTP chain | Choose chain initially: bounded verification and state footprint. Reconsider a small tree only after measured whole-cycle benefit and branch-state correctness; no assumed tree speedup. |
| TriRun/Xe2 kernels vs Iris Xe Vulkan | Keep existing Vulkan path. Port only hardware-supported tile/dataflow ideas; no CUDA, XeTLA/Xe2 or tensor-core dependency. |
| FlashAttention/Flash-Decoding/FlashInfer | Exact tiling, partial reduction and layouts can compose but duplicate existing work unless dispatch gaps are proven. Q4 KV currently misses CPU F16/F32 split-KV eligibility. |
| LLM-in-a-Flash prediction/offload vs resident dense execution | Choose packed resident streaming/reuse. Dense weights are not skipped; no unnecessary flash offloading or speculative sparse predictor. |

These are our architecture choices under current constraints, not measured winners.
For physically compatible compute alternatives with unknown costs, preserve the
current path and specify an experiment rather than fabricate a speed ranking.
Mutually exclusive layouts can be separate candidates but must not cause concurrent
multi-GB copies. A CPU-only repack that breaks Vulkan prefill eligibility is not
accepted as the global representation; account for every staging/transition copy.

## Priority and execution order

1. **D-0 + D-B feasibility first:** record exact per-tensor zero counts, shapes,
   scales and layout costs; ISA/topology capabilities; profile current paths.
   This is the BITCOS priority gate, not automatic repacking. Reading the large
   model for statistics or running profilers waits for authorized target work.
2. **D-S + D-N are the primary implementation tracks:** bounded phase/core-plan
   search, then exact AVX2 LUT versus existing unpack/dot/reuse. Keep one process,
   one model, original sampler and independent fallback. Validate the existing v2
   candidate before attributing any incremental improvement.
3. **D-B prototype only if viable:** exact bitmap/sign unpack with optional BMI2,
   existing Q8 group boundaries and index accounting; compare L1-resident unpack
   against larger-than-LLC streaming. An instruction-bound case may win despite
   more bytes, but needs measured evidence; otherwise close with a recorded rejection.
4. **D-F/D-T/D-R adaptations:** targeted register/activation reuse and supported
   Vulkan prefill staging. Do not create a second competing tiling framework.
5. **D-Q and attention gaps:** joint MTP budgeting first; small tree and quantized
   split-KV/layout work only after profiling justifies it. No full tree/ISA port now.

## Required acceptance and fallback

Keep model/head/scales immutable and one full resident representation. Inspect
feature flags on every P/E core class; reject unsupported instructions. Include
conversion, lookup construction, repack startup, RSS/page cache, driver allocations,
context cache and recurrent checkpoints in memory/time accounting. No whole-model
FP16 or int8 expansion. The user's 5 GiB free-RAM limit applies only to the preparation
laptop; target reserve selection remains pending, not silently removed here.

Validate integer dots and boundaries first, then full-backend logits, greedy tokens
and agent actions. Test -128 activations, signed/zero scales, extreme/odd dimensions,
scale-group tails, strided inputs, unaligned buffers, format tails, all-zero/all-nonzero
blocks, BMI2 fallback and all MTP accept/reject/replay outcomes. Numerical reorder
is not accepted merely because it is mathematically equivalent.

Benchmark decode, prefill and verification separately and together; compare MTP
off/fixed/adaptive, cold startup, repeated turns and sustained thermal state. Use
paired repeated runs with noise/variation reported and no arbitrary minimum gain:
small repeatable gains count. Adopt only a correct, memory-fitting whole-workflow
improvement; retain baseline if a gain is indistinguishable from noise. Cache policy
keys include model/head hash, runtime/build, ISA/core class, shape, phase, thread
plan, KV type, context bucket and sampler constraints. Retune after changes; keep
tuning outside ordinary request latency and retain cancellation/ownership guards.

## Previous five papers: attribution boundary

The engine README now links all twelve sources. The earlier T-MAC-inspired source
candidate is our adaptation of locality/reuse, not a T-MAC port or a replicated
paper result. FlashAttention and Flash-Decoding describe mechanisms already found
in pinned Prism; no new attention kernel was added. FlashInfer informs future
layout/scheduling checks; its CUDA package was not imported. LLM in a Flash was
reviewed and its offload/predictor path rejected for this resident dense workload.
See the earlier review for source locations and limits. This revision adds plans,
references and compatibility decisions only; all new implementation gates stay open.


## Implementation increment: Intel decode v3

See [package README](../patches/prism-intel-decode/README.md) and checklist 4B.
The initial research-only revision above selected the structures; this increment
implements the exact pair-LUT/guarded matmul route, BITCOS codec/dot/cost analysis,
and explicit server phase/pool hooks plus offline topology/cost planning. It does
not implement a resident BITCOS GGML layout, a compiler autotuner, automatic device
migration or a new joint online MTP controller. Original model and head stay intact.

Seven Python tests and source/compile checks pass; no native code or model was run.
Earlier plan-only status is superseded only for these source components. Priority
and compatibility choices still require actual target evidence, not paper speedups.


## Additional source increment and research — 22 October 2026

Checklist4C and [additional review](ADDITIONAL_INFERENCE_REVIEW.md) cover four newly
investigated papers and explicit overlap/defer decisions. [Optional v3 delta](../patches/prism-intel-decode-increment/README.md)
adds E1 exact LUT local-work and E2 MTP owned hidden-row copy candidates. It changes
three source paths with independent disabled flags and exact prerequisite/hash
checks, preserving KV/recurrent rollback and existing Intel Vulkan routing. Four
Python oracle tests and compile/patch checks pass; native tests/full runtime/model
parity and target performance remain pending. New packed-layout conversion,
CPU/iGPU tensor partitioning and ReplaySSM reassociation are not implemented.


## Next selection priorities — five-paper review

[Detailed review](FIVE_PAPER_SELECTION_REVIEW.md) and checklist4D select F1 ordinary
decoding/off with complete cost accounting, F2 bounded suffix-chain lookup, then F3
joint cost-aware source/window choice using the existing MTP head. These consolidate
the earlier D-Q off/budget idea; no competing selector is proposed. Keep Sandwich
phase plans fixed during selection experiments. F4 is bounded cache/register PTQ1
tuning; F5 is an exact adjacent-normalization fusion audit. F6 tree execution stays
exploratory until recurrence/state/Intel compatibility is established. All are
design only; present1..4-window tuning still lacks off/source selection.
