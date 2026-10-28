# Xenolith engine components

Current optimization selection: conditional combinations C1–C8 (local development notes; excluded from source distribution)
(25 October 2026), replacing blanket either/or choices in earlier reviews.
Reuse shape dispatch and one cost controller; condition layouts/attention on actual
eligibility. This is a plan revision, not runtime activation or Intel acceptance.
Paper references and historical implementation boundaries below remain intact.

Two explicit routes:

| Route | Implementation | Current status |
|---|---|---|
| Native Gemma4 E4B | Existing C/Level Zero code, Gemma-specific profile and Q4_0 graph | Unvalidated; default gate retained |
| Bonsai 2 27B PTQ1 Lean | `launch.py` → owner-installed Linux PrismML llama.cpp → local `/v1` endpoint | Launcher/profiles implemented; target runtime untested |

Bonsai is not passed through the Gemma loader/template, and native snapshot/wire capabilities are not advertised for the process backend. Its PTQ1 graph, transformations and MTP are provided by Prism; there is no second full draft model in the launch command. See [backend status/runbook](docs/BONSAI_BACKEND_STATUS.md).

`python3 launch.py --profile profiles/bonsai27-linux-cpu.json --engine /opt/prism/bin/llama-server --model /models/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` prints a JSON plan only. Launch requires an explicit `--start` on the compatible Linux target; Windows refuses before opening engine/model files. No installer or startup hook runs here.

`make check-backends` uses Python-only no-model contract tests. `make` still builds only the native E4B engine and requires compatible Linux GCC, Level Zero and Intel ocloc. No C build, engine/model test or GPU workload was run in this Windows revision.

The active plan (local development notes; excluded from source distribution), checklist (local development notes; excluded from source distribution) and [E4B source status](docs/E4B_PORT_STATUS.md) distinguish present source, historical reference checks and pending target validation. Vision, native Bonsai kernels, full Hermes integration and physical KV eviction remain unfinished. Original licensing remains in [LICENSE.md](LICENSE.md).

## Intel CPU optimization increment — 22 October 2026

This increment supersedes the earlier blanket no-prefix-reuse configuration **only for the opt-in CPU candidate**. PTQ1 decoder/packing/Hadamard/matrix kernels remain deferred at the user's request. The original CPU MTP-off/prefix-off profile and live Hermes settings remain available and inactive here.

Implemented source: exact-prefix-friendly controlled-state serialization with full source/evidence/correction preservation, local HTTP keep-alive without request replay, separate prompt/decode/draft CPU threads, optional affinity masks and polling, complete-verification-microbatch guards, a CPU-only candidate profile and a dry-run24-plan tuning renderer. The cumulative cpu-mtp-v1 external-runtime patch adds buffer reuse, all-accepted hidden-copy elision, early enforcement of per-sequence draft limits and optional per-request timing-based MTP window selection (1..min(4,cap), four samples/window, a probe every32 valid cycles). It includes greedy-mtp; apply to the clean pinned source, never stack both patches.

Candidate: Linux CPU, device none and target/draft GPU layers0, one slot,8192 context, batch256/ubatch128, six prompt/decode threads, polling0, p_min0, adaptive cap3. Prefix cache is requested with128MiB prompt-cache budget, two additional context checkpoints and idle-slot archiving off. Live KV/checkpoints are outside that budget; >=5GiB MemAvailable safeguard remains. CPU backend sampling is independently selectable, not GPU placement. No CUDA switch is emitted. No affinity topology or fastest thread/batch choice is presumed.

Hermes owner CLI options are `--reuse-prefix --keep-alive`; both are opt-in. They do not activate ordinary Hermes or store/reinject prior reasoning. Prefix eligibility is exact rendered-token identity; hybrid state restoration/recomputation, changed user instructions and removed-reasoning tails need target parity acceptance. No persistent KV cache or automatic inference/tool retry is added. SSD pending-action writes/readback remain mandatory. Existing controlled-run reasoning-off/output1024 defaults are retained; the launcher's low/4096 request defaults are a separate production-provider configuration decision.

The actual C++ policy compiled and passed standalone tests. Python launcher/harness contracts, patch application/reverse/hash checks and dry-run tuning rendering were checked without a model. The full runtime remains uncompiled and unbenchmarked. Adaptive warmup can cost time on short answers and does not compare MTP-off internally. Compare whole-task latency, token/action parity, cache hits, recurrent replay, RAM/swap and sustained thermals before adoption. Do not remove synchronization or recurrent checkpoints on an assumption that CPU execution makes them unnecessary.

Implementation and target procedure: `engine/patches/prism-cpu-mtp/README.md` in Xeno; cumulative patch pinned to sudoingX ff414120c343e6e6cb868013c99f1dde52b27e70. Target compilation/validation and runtime activation remain pending. No model, engine service, GPU or WSL was loaded here; only the standalone policy test binary ran.


## PTQ1 AVX2 source candidate — 22 October 2026

`patches/prism-ptq1-avx2/` adds a source-only AVX2 fused trit-unpack/dot specialization for the external Bonsai Prism CPU backend, preserving SSE fallback, compact weights and four Q8 scales. This supersedes PTQ1-kernel deferral only for this initial candidate. VNNI, small-batch matrix reuse, prefill repacking and Hadamard fusion remain future work.

Patch applicability/reversal/hash checks and C/C++ compile-only checks passed. The kernel, comparison test, full runtime and model were not executed. Existing binaries/profiles were not changed; speed and numerical parity remain unverified. See the package README for exact source pin, independent composition with cpu-mtp-v1 and target acceptance. Native Gemma readiness guards are unchanged.

## Paper-guided CPU inference increment — 22 October 2026

Prepared `engine/patches/prism-ptq1-reuse/` (relative to project root), a cumulative
v2 replacement for the initial AVX2 patch, pinned to sudoingX ff414120. Never stack
both PTQ1 patches. The existing CPU-MTP patch is disjoint; combined runtime testing
is still pending. No installed runtime, model, Vulkan shader or live profile changed.

The optional AVX2 2–5-column matmul path shares each packed PTQ1 block's unpacking
across verification tokens, retaining original scales, per-column accumulation
order, bounded loads and fallback dispatch. It uses the existing head/batched
graph and changes neither draft acceptance nor KV/recurrent commit/rollback.
`GGML_PTQ1_ENABLE_BATCH` remains off by default. An exact-int16 LUT prototype and
baseline/AVX2/reuse/LUT comparison test and microbenchmark are supplied separately;
the scalar LUT prototype is not a tuned T-MAC implementation or runtime path.

Pinned CPU source already has tiled online attention and KV-split reduction for
eligible F16/F32 decode, but the Q4_0 KV profile does not select that split path.
Preserve Intel Vulkan prefill; automatic CPU-decode/iGPU-prefill migration is not
implemented. Do not add CUDA, approximate LUT sums, predictive dense-neuron skipping,
weight offload or a second expanded model. See engine/docs/INTEL_INFERENCE_PAPER_REVIEW.md
for all five papers, source evidence and separate CPU/iGPU decisions.

Patch/hash/reverse and C/C++ compile-only checks passed. Native parity tests,
guard pages/sanitizers, full runtime build, logits/tokens/actions, MTP replay and
target speed/DRAM measurements remain required before activation. Cache hits mean
less unpacking is not automatically less DRAM traffic. Reject parity failures or
regressions and retain fallback. No engine/new native test executable was run.

The 5 GiB free-RAM reserve applies to the preparation laptop per the user, not the
16 GB target. Existing profile/validator values still need a separately chosen
target setting; this increment does not remove safeguards or invent a new reserve.


## Decode research roadmap and paper references

The structure roadmap (local development notes; excluded from source distribution) explains mechanisms,
compatibility, selection and fallback; checklist section 4A (local development notes; excluded from source distribution)
contains the implementation gates. The initial research revision was documentation-only;
see the v3 source status below. Prioritize
BITCOS feasibility, exact bitnet.cpp lookup, and Sandwich phase scheduling. Keep PTQ1
as the baseline; these priorities do not establish a measured speed winner.

References below distinguish our source adaptations, existing upstream mechanisms,
and future work. They do not imply a complete paper implementation or reproduced gains.

| Paper | Relationship to this engine |
|---|---|
| [T-MAC](https://arxiv.org/abs/2407.00088) | Prior source-only adaptation of locality, bounded unpack/reuse and exact LUT comparison in `patches/prism-ptq1-reuse`; not a T-MAC port. Approximate table sums are excluded. |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | Pinned upstream tiled online attention was audited and preserved; no new attention kernel added. |
| [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | Existing upstream split-KV reduction audited for eligible F16/F32 CPU decode; current Q4_0 KV does not select that path. |
| [FlashInfer](https://arxiv.org/abs/2501.01005) | Layout/scheduling ideas inform pending checks; no FlashInfer dependency or new paged-cache implementation. |
| [LLM in a Flash](https://arxiv.org/abs/2312.11514) | Resident reuse considered; predictive loading/offloading rejected for this already-resident dense workload. No implementation claimed. |
| [bitnet.cpp](https://arxiv.org/abs/2502.11880) | D-N: v3 source implements exact AVX2 pair lookup and opt-in matmul dispatch; PTQ1/Q8 scales retained. Target comparison/parity pending. |
| [TriTera / TriRun](https://aclanthology.org/2025.acl-long.1294/) | D-R: division-free decode and fused staging review; existing extraction already has the former. CUDA kernels are not a portable Intel implementation. |
| [FairyFuse](https://arxiv.org/abs/2604.20913) | D-F: planned graph-compatible register/activation reuse audit; complex-model fusion and AVX-512 assumptions excluded. |
| [BITCOS](https://arxiv.org/abs/2609.16338) | D-B: v3 source includes cost analyzer, conditional codec and AVX2 dot; no resident layout hook or global bitmap conversion activated. |
| [Sandwich](https://arxiv.org/abs/2507.18454) | D-S: v3 source implements bounded topology plans and server phase/pool hooks; single model, target affinity/performance acceptance pending. |
| [Sequoia](https://arxiv.org/abs/2402.12374) | D-Q: planned hardware-cost-aware MTP budget including off; tree verification deferred pending hybrid-state correctness. |
| [T-SAR](https://arxiv.org/abs/2511.13676) | D-T: software tiling ideas only; custom TLUT/TGEMV ISA is not available on the target. |

The supplied Sequoia URL, arXiv:2402.06663, names a different wireless-security
paper. The corrected speculative-decoding reference above is used throughout.
Three priority adaptations now have inactive source candidates in `patches/prism-intel-decode`;
the other four remain research/checklist work. Target numerical, memory and performance
acceptance remains open for all new paths.


## Intel decode v3 source implementation — 22 October 2026

[Package and target procedure](patches/prism-intel-decode/README.md): exact row-tiled
AVX2 lookup with optional GGML dispatch; BITCOS cost tool/codec/dot comparison;
Sandwich-derived phase plans and real server pool-selection hooks. All are source
candidates, inactive here. BITCOS has no resident GGML layout hook; a second full
weight cache is not introduced. Existing Vulkan routing and profiles are untouched.

This cumulative package replaces PTQ1 v1/v2 and CPU-MTP v1/greedy-MTP when choosing
a future clean build; never stack them. Source hashes/apply/reverse and compile
checks passed, as did seven Python tests. C++ kernel/mock tests and benchmarks
were compiled as objects only. Full runtime, numerical parity, affinity/races,
RSS/DRAM and speed remain unverified; no model/native executable/GPU/WSL was run.
Native CPU pools (`GGML_OPENMP=OFF`) are required for this scheduler's affinity/
pause behavior. The deployment target's memory reserve remains a separate decision.


## Additional Intel inference increment — 22 October 2026

[Optional delta on v3](patches/prism-intel-decode-increment/README.md): exact compact
LUT integer accumulation/SIMD construction/scratch work and single-copy owned MTP
hidden-row capture. Each flag defaults off; no KV/recurrent checkpoint elision or
Intel Vulkan architecture change. Existing unpack/table reuse is not counted again.
Four Python oracles, patch integrity/reversal and compiler checks pass; native
tests are compiled only. No model/native engine run or measured speedup is claimed.

Additional sources and decisions ([full review](docs/ADDITIONAL_INFERENCE_REVIEW.md)):

| Paper | New mechanism considered; current disposition |
| --- | --- |
| [FLUTE](https://arxiv.org/abs/2407.10960) | Execution-aligned lossless packed layout; a future one-representation experiment. No resident conversion or second weight cache added. |
| [LUT-GEMM](https://arxiv.org/abs/2206.09557) | Fused lookup/accumulation overlaps existing packed paths; no BCQ requantization or duplicate kernel port. |
| [Ghidorah](https://arxiv.org/abs/2505.23219) | Contention-aware heterogeneous profiling; existing CPU phase planning retained, concurrent Intel CPU/iGPU tensor partitioning deferred. |
| [ReplaySSM](https://arxiv.org/abs/2610.13778) | Recurrent input replay/output-only verification; floating evaluation changes and backend rewrite need separate correctness work. Existing rollback retained. |

These four papers were investigated, not newly ported in full. The local LUT
engineering continues the prior bitnet.cpp/T-MAC source candidates; all earlier
paper references and their implementation-status limitations still apply.


## Draft selection and CPU papers — 22 October 2026

[Review and implementation order](docs/FIVE_PAPER_SELECTION_REVIEW.md), checklist4D.
The paper review records the design. An opt-in source adaptation now exists in
[prism-source-selection](patches/prism-source-selection/README.md); it remains
inactive and unvalidated in the full runtime.

| Paper | Selection for this engine |
| --- | --- |
| [SuffixDecoding](https://arxiv.org/abs/2411.04975) | Priority: bounded scored suffix-chain draft source complementing the community MTP head; compare existing n-gram first. Full tree method deferred. |
| [BanditSpec](https://arxiv.org/abs/2505.15141) | Priority: one cost-aware source/window selector including ordinary decoding. Replaces existing no-off window tuner; unequal-time adaptation needs its own validation. |
| [FlashNorm](https://arxiv.org/abs/2407.09577) | Limited adjacent-fusion audit; existing RMSNorm+scale fusion already present. No automatic PTQ1 weight folding or epsilon/rounding-sensitive norm removal. |
| [QIGen](https://arxiv.org/abs/2307.03738) | Select bounded cache/register-aware PTQ1 schedule tuning, not GPTQ format conversion or changed reduction order. |
| [STree](https://arxiv.org/abs/2505.14969) | Exploratory: diagonal-SSM tree scan is not a drop-in Gated DeltaNet/Intel kernel. Preserve current recurrent rollback. |

Without the new opt-in controller, the existing fixed-order n-gram/MTP dispatch
is not cost-aware source selection.
All-source processing and replay can incur costs outside the current draft timer;
off and source-switch accounting must handle them explicitly. No second draft
model, new training, model/head change or CPU/iGPU architecture change is selected.


## Source arbitration and optional QIGen schedules — 23 October 2026

[Implementation, switches and rollback](patches/prism-source-selection/README.md)
requires exact v3 plus increment-v1. It adds request-local warm-off/MTP/suffix
selection, full-cycle/replay/catch-up accounting and a bounded scored suffix-chain
source. Temperature0 and one slot only; existing verification/state hooks remain.
`LLAMA_SPEC_SOURCE_POLICY`, `LLAMA_SPEC_SUFFIX` and optional forced arm controls
enable independent tests. Warm off still maintains the head; true head-disabled
comparison remains necessary. The BanditSpec-inspired policy has no transferred
round-based regret guarantee; the SuffixDecoding adaptation is not its full tree.

QIGen-inspired scheduling has independent compile `GGML_PTQ1_ENABLE_QIGEN` and
runtime `GGML_PTQ1_QIGEN` switches plus a five-column schedule. It compares existing
exact dot/batch/LUT8/LUT16 paths without changing packed weights, four activation
scales or FP order. Offline recommendations need measured correctness and repeated
target timings; no online calibration or automatic activation. This is a bounded
schedule implementation, not QIGen's GPTQ generator. See the five cited papers in
the table above and package README for the precise attribution/deferred work.

Seven offline tests and source/compiler checks pass; native tests are compiled
only. Full runtime, numerical/model/action parity, memory and speed remain open.
No native engine/model/GPU/WSL run, binary/profile change, or Intel Vulkan routing
change occurred. Existing upstream license and all earlier paper references remain.


## Persistent draft-source learning - 23 October 2026

The [portable source profile package](patches/prism-source-profile/README.md) adds request-to-request cost learning,
profile export/import, frozen held-out evaluation and 24 Hermes-style workflows
(72 requests). It is applied in a separate pinned Prism source checkout and
supported by Xenolith's external Prism adapter; no installed binary is replaced.
The native Gemma graph does not consume this profile. Hardware/build/settings
mismatches require fresh calibration, rather than importing foreign timing priors.

This extends ideas from [BanditSpec](https://arxiv.org/abs/2505.15141) and
[SuffixDecoding](https://arxiv.org/abs/2411.04975), using a bounded deterministic
rate selector and prompt-local suffix chains, not the full published algorithms.
30 Python contracts pass and full speculation/server sources compile in syntax
checks. Native/runtime/model parity and Intel speed tests remain unrun. No learned
performance numbers or acceleration are claimed. Existing launch profiles remain
disabled/unmodified; the target RAM-reserve issue remains a separate open item.


User-authorized runtime update: [default-on source selector](patches/prism-source-default/README.md) retains
cost learning across eligible MTP requests without an enabling env switch.
The benchmark supplies explicit disk profile paths for restart persistence.
Runtime identities now include adjacent shared libraries. No Intel speed claim
is implied by a Windows/NVIDIA benchmark.

## Runtime lifecycle increment — 23 October 2026

The [runtime lifecycle package](patches/prism-runtime-lifecycle/README.md) adds
masked MTP hidden-row selection, optional cache-line CPU scheduling, bounded
aggregate telemetry and per-graph compute-buffer limits, and optional coalescing
of redundant waits across consecutive blocking input copies. It is applied to the
existing composed Prism development source; installed binaries are unchanged.
Existing PTQ1 kernels, batched verification, exact rollback and Intel iGPU placement
remain in place. Experimental scheduling, telemetry and wait coalescing default off.

All 18 packaged model-free execution/translation-unit compile checks pass against
the applied source. Hash guards and apply/reverse checks also pass. Real-model
parity, recurrent backend execution, Linux build and Intel speed/RSS measurements
remain required. Direct snapshot writes and alternative weight layouts are deferred
until lifetime and profiling evidence justify them. See the package checklist and
pinned source references before enabling experiments.

## Additional engine structures — 23 October 2026

The [additional engine review](docs/ADDITIONAL_ENGINE_REVIEW.md) inspects pinned
MLC-LLM, OpenVINO, ONNX Runtime, XNNPACK, KTransformers and mistral.rs sources.
The [suffix workspace increment](patches/prism-suffix-workspace/README.md) applies
bounded request-owned scratch reuse and a synchronous borrowed history view to
the external Prism suffix drafter. It is opt-in via LLAMA_SPEC_SUFFIX_WORKSPACE=1;
installed binaries and launch profiles remain unchanged. It composes after the
runtime-lifecycle delta and must be reversed first. Exact draft-vector parity
passes 836,772 cases; full modified source and request-lifecycle fixtures pass.
Warm helper allocation/copy work falls, but Intel/end-to-end gains remain unmeasured.

References: [MLC draft workspace](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/cpp/serve/draft_token_workspace_manager.cc),
[OpenVINO memory ownership](https://github.com/openvinotoolkit/openvino/blob/695d69efe543b9e6c75361bc5d3df9c866d9c6ed/src/plugins/intel_cpu/src/memory_control.cpp),
[ONNX memory planning](https://github.com/microsoft/onnxruntime/blob/84829e76d4fbb4a7258202f777fa4ff0876a3564/onnxruntime/core/framework/execution_frame.cc).
The independent implementation copies no upstream engine code. Other reviewed
structures either overlap existing code or need profiling/backend parity first.


## Additional paper review — 23 October 2026

[Paper review and decisions](docs/ADDITIONAL_PAPER_REVIEW_2.md): prioritize full-cycle
selector cost accounting and real-backend recurrent-state oracle tests. These are
pending research/validation items; no new runtime optimization is implemented here.
LeanAttention is conditional on an attention bottleneck. TreeWY remains exploratory
because its reported token streams are not bit-identical; existing ReplaySSM work
overlaps. Block Verification adds no benefit at temperature zero, and NoMAD's
approximate attention is excluded from the present correctness scope.

References: [SmartSpec](https://arxiv.org/abs/2406.14066),
[Decoding Speculative Decoding](https://arxiv.org/abs/2402.01528),
[GDN Tree-Scan](https://arxiv.org/abs/2609.23900),
[TreeWY](https://arxiv.org/abs/2608.20961),
[LeanAttention](https://arxiv.org/abs/2405.10480),
[Block Verification](https://arxiv.org/abs/2403.10444),
[NoMAD-Attention](https://arxiv.org/abs/2403.01273).


## Further CPU decode paper candidates — 23 October 2026

CPU paper comparison (local development notes; excluded from source distribution) records
[Vec-LUT](https://arxiv.org/abs/2512.06443) as a deferred small-column competitor
to existing PTQ1 reuse, and [AI-PC](https://arxiv.org/abs/2508.06753) operand-layout
ideas as a feasibility lead. Neither is implemented by this documentation update.


## AI-PC Intel instruction adaptation — 23 October 2026

Design and acceptance gates (local development notes; excluded from source distribution) map the
[AI-PC paper](https://arxiv.org/abs/2508.06753) to existing base-3 PTQ1. First
consider feature-gated VEX VPDPBUSD with exact unsigned-code correction, then
bounded lane staging only with evidence. The paper's signed-byte VPDPBSSD and
uniform int2 layout are not assumed available or directly compatible.
[Intel extension reference](https://cdrdv2-public.intel.com/843860/architecture-instruction-set-extensions-programming-reference-dec-24.pdf)
and [instruction reference](https://cdrdv2-public.intel.com/835757/325383-sdm-vol-2abcd.pdf)
are cited in the design. This revision changes documentation only; no SIMD kernel,
permanent weight layout or measured speedup is implemented.


## CPU hot-path increment - 23 October 2026

[Exact suffix search/voting and optional vector scaling](patches/prism-hotpath-increment/README.md)
are applied to the composed Prism source as a reversible two-header layer.
Existing installed binaries are unchanged. Suffix scratch mode remains opt-in;
vector scaling is disabled by default because synthetic results were mixed.
Model-free parity and timings are recorded in the package. MTP plus suffix are
currently selected per cycle; suffix-first/MTP-fallback is a documented candidate,
not an implemented hybrid arm. Intel laptop/model throughput remains untested.
