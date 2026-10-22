# Xenolith engine components

Two explicit routes:

| Route | Implementation | Current status |
|---|---|---|
| Native Gemma4 E4B | Existing C/Level Zero code, Gemma-specific profile and Q4_0 graph | Unvalidated; default gate retained |
| Bonsai 2 27B PTQ1 Lean | `launch.py` → owner-installed Linux PrismML llama.cpp → local `/v1` endpoint | Launcher/profiles implemented; target runtime untested |

Bonsai is not passed through the Gemma loader/template, and native snapshot/wire capabilities are not advertised for the process backend. Its PTQ1 graph, transformations and MTP are provided by Prism; there is no second full draft model in the launch command. See [backend status/runbook](docs/BONSAI_BACKEND_STATUS.md).

`python3 launch.py --profile profiles/bonsai27-linux-cpu.json --engine /opt/prism/bin/llama-server --model /models/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` prints a JSON plan only. Launch requires an explicit `--start` on the compatible Linux target; Windows refuses before opening engine/model files. No installer or startup hook runs here.

`make check-backends` uses Python-only no-model contract tests. `make` still builds only the native E4B engine and requires compatible Linux GCC, Level Zero and Intel ocloc. No C build, engine/model test or GPU workload was run in this Windows revision.

The [active plan](docs/Local_Bonsai27_Hermes_Xenolith_Memory_Plan_v5.md), [checklist](docs/PROJECT_FUNCTION_CHECKLIST.md) and [E4B source status](docs/E4B_PORT_STATUS.md) distinguish present source, historical reference checks and pending target validation. Vision, native Bonsai kernels, full Hermes integration and physical KV eviction remain unfinished. Original licensing remains in [LICENSE.md](LICENSE.md).

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

The [structure roadmap](docs/ENGINE_DECODE_STRUCTURE_ROADMAP.md) explains mechanisms,
compatibility, selection and fallback; [checklist section 4A](docs/PROJECT_FUNCTION_CHECKLIST.md#4a-bonsai-prism-paper-derived-decode-structures)
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
