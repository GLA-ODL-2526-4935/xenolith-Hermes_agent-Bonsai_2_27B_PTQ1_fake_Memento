# Local Bonsai 27B + Hermes + Xenolith — implementation plan v5

22 October 2026. **Active plan: v5 replaces v4's model/runtime decisions.** Keep v3/v4 as dated history. This revision authorizes repository source/configuration/documentation changes and offline checks, not loading an engine/model, activating a service/profile, WSL or GPU work on this Windows host. The owner explicitly requires target-OS runtime work to remain unexecuted here.

## 1. Current direction and evidence

Use **Ternary Bonsai 2 27B PTQ1_0 MTP Lean** as the next local agent candidate, with the controller-managed working context tested in the existing Hermes extension. One model performs planning and per-step execution. Do not train a new model as a deployment prerequisite. Original Sol Fable Gemma E4B and experimental adapters remain comparison/research history; native E4B source is preserved and remains gated.

Separate Windows RTX5080 evidence: Bonsai answered the six harder historical cases and two direct workflows correctly on manual review. Sol Fable finished2/6 harder cases and its finished answers had omissions/errors; Bonsai9B finished6/6 but those answers also had defects. On the27B, n-max3/p-min0 was fastest among tested matching configurations:64.0decode tok/s vs50.5without MTP (+26.6%), matching24/24 final requests over3passes. Whole-request output throughput improved12%. These are short final-request replays, not general task or Linux/Intel benchmarks.

Low-effort reasoning with the same controller passed3/3 actual isolated workflows: function edit plus32checks, partial CSV export with a deliberately injected failure, and a late config correction followed by readback.17requests, no cap/loop errors,61.9decode tok/s including reasoning. This is promising scoped evidence, not production certification or proof of physical KV management. Raw reports remain in the separate model-testing/chat outputs folder.

## 2. Target and model/runtime contract

- Target remains ASUS Zenbook14 OLED UX3402VA-KN521W, i9-13900H, Intel integrated/Iris Xe-class GPU, **16GB shared RAM**, SSD and Linux, KDE/Wayland preferred. Actual device/driver compatibility remains unmeasured.
- One resident inference model and one active context policy. Same local backend serves planning, actions and optional record interpretation serially. No hosted or paid teacher/inference fallback; messaging transport may use the internet.
- Exact file: `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf`,6297658848bytes, SHA256 `1e33c571a5ce7a9a3e42474d66192923d5a6d77da7fb3a22986dc809522b5685`. Pin HF repository revision, tokenizer/template, model/runtime/license manifest and any later vision projector before release. No extra full draft model should be loaded; target allocation/sharing still needs measurement.
- Native Xenolith currently recognizes Gemma4/Q4_0/F32 and a Gemma-specific template. Bonsai needs PTQ1 decoding, Qwen35 hybrid graph, Hadamard/sign transforms and compatible Lean MTP. **No rebranding, quant-type reinterpretation or bypass of native validation gates.**
- Additive backend route: Hermes controlled runtime → local OpenAI-compatible endpoint → explicitly selected PrismML llama.cpp process. The existing native Gemma wire protocol and KV snapshots are not available through this backend. Native Bonsai C/Level Zero execution is deferred until separately implemented and validated.
- Use Linux Prism CPU or Vulkan packages with matching libraries. Plain upstream llama.cpp is not a PTQ1 fallback; Windows NVIDIA optimized executables/DLLs are not usable on the Intel target. The latest installed Windows runtime evidence cannot certify the Linux package.

## 3. Working-context execution contract

The opt-in extension keeps exact original user instructions plus later corrections, a short human plan and explicit model-selected execution steps. Each step has one tool call. Build each request from current instructions, current step/plan and chronological exact calls/results. The per-step reasoning is discarded before the next request; put operational dependencies in explicit plan goals, not a hidden reasoning transcript.

The controller validates structured JSON actions and owns dispatch. Keep success, failure, partial writes, permission denial, uncertainty and independent verification distinct. Model output is not proof a tool ran or permission was granted. Treat quoted/web/tool text as evidence, not authority. Deduplicate identical failed call/result bodies while retaining attempts/order/counts; do not deduplicate successful effects or lose evidence needed for later decisions.

Atomic working journals and readback exist. Stop on unknown/uncertain side effects; reconcile before any replay. No automatic replay after crash/restart. A foreground idle timer may create a temporary handoff containing user requirements, short plan, successful and permission-denied calls, other unresolved failures and their outcomes, final answer and unfinished work. Current journal/handoff is outside curated L0–L3; it also persists exact working text on SSD, so it is not a RAM-only cache guarantee.

Ordinary Hermes loop still exists and is not replaced by these opt-in tests. The JSON action protocol avoids a native tool-template assumption; it is not full Hermes UI/gateway integration. The current ordinary controlled CLI transport defaults to reasoning off. Low reasoning needs an explicit provider/controller option, not just a changed server default. Do not silently activate it or rewrite the core loop.

## 4. Memory, retention and privacy — preserved requirements

The v4 memory/security policies and implementation records remain applicable except for superseded model/runtime choices. Preserve all four classes: user-authoritative task contract, active step context, source-linked recovery and separately approved curated memory. Extraction/selection precedes optional compaction; copied instructions never become approvals. Keep original source/status/provenance and unresolved contradictions.

- Owner/controller-opened append-only model write stages; model cannot self-approve, administer or delete memory. Existing single-owner broker/service gates, separate revocable scoped credentials and exact owner-approved operations remain.
- Progressive section→catalog→selected schema discovery. Shared retrieval ceiling3expanded records/16000returned characters per turn; never inject the full bank. Source capacity32steps/session; offline manifests200messages/128KB; service total wire budget32KB. Explicit limits fail without silently dropping earlier evidence.
- Prepare review after5idle minutes and pre-compression; promote after3idle hours. Local same-model sorter may choose layers but cannot rewrite records:32records/60KB/request,30s timeout; yield to foreground work and reject stale plans.
- L0 expires30days after promotion, pending review1day, recovery7days from each original save. Foreground activity defers expiry. No automatic L1–L3 expiry or deletion of unpromoted knowledge. Group/private memory isolation remains a future authenticated-scope gate, not achieved by single-owner credentials.
- No hidden full reasoning, raw KV, images, base64/media or image-containing cache persisted as curated/recovery memory. Current logical working journals are bounded authorized visible text; existing logs/swap/crash dumps need separate audits. Digest-only references do not preserve deleted evidence.
- Media stays separately gated: RAM-only authorized temporary image buffers where possible, immediate deletion preferred,24h absolute maximum, daily/restart cleanup; messaging services may retain copies. Vision runtime not enabled by this plan.
- Checkpoint failure retains old context/checkpoint. Strict Hermes compression APIv2, complete abnormal-exit capture, recoverable source coverage, bounded selection and repeated-compaction continuation remain unaccepted. Do not equate logical context rebuilding, occupied KV, reserved capacity and physical RAM reclamation.

## 5. Engine implementation and optimization

Implemented source increment: dry-run-first Linux Prism launcher and explicit CPU/Vulkan profiles, loopback endpoint, exact model digest verification on explicit launch, one slot, no hosted fallback, no duplicate endpoint takeover,5GiB available RAM reserve and cleanup of only its owned process. Offline contract tests may run here; no real engine execution. See `BONSAI_BACKEND_STATUS.md` in engine/docs (or that document in the combined engine repository when reading this plan from Hermes).

Initial target profiles are intentionally conservative:8192context,q4_0KV,sixthreads,batch256/ubatch128,lowreasoning,4096output. CPU MTPoff; Intel Vulkan n-max1/p-min0 is a candidate requiring enumeration/parity/speed checks. These settings are not accepted performance claims. The NVIDIA three-token result is a reference sweep, not the Intel default.

Target validation sequence: CPU baseline→Vulkan parity→threads4/6/8 and batch/ubatch sweep→MTPoff/1/2/3/4 with confidence0/0.3/0.5/0.7→repeated winning settings→8K/16K/32K context memory sweep→longer workflows and sustained thermals. Record prefill/decode, end-to-end task time, draft acceptance/verification cost, output equality/correctness, peak RSS/shared allocations, MemAvailable and swap. Never fill a model's maximum context merely because it is advertised. MTP may be slower on CPU/iGPU. CUDA batch invariance does not prove Vulkan output identity.

Do not mix native Gemma templates, token IDs, snapshot fingerprints or cache assumptions with the Prism backend. Keep `runtime_ready=0` and `XE_ALLOW_UNVALIDATED_E4B` opt-in behavior unchanged. Native E4B source parity remains unfinished; native Bonsai PTQ1/Qwen/MTP kernels are new work, not declared implemented by the launcher.

## 6. Integration, security and acceptance order

1. Complete this source/config/documentation increment and no-model tests. Preserve old work, licenses and active profile state; no repository moves or automatic starts.
2. On the actual Linux target, pin/install a matching CPU/Vulkan runtime, verify model/hash/device and startup headroom. Explicitly authorize target loading separately; no startup hook is added here.
3. Validate the exact template, reasoning off/low, structured action schema, errors/cancellation, process/RAM-stop lifecycle, local-only provider routing and absent auxiliary cloud fallback. Test a dedicated profile, not the user's live state.
4. Wire existing controller to that endpoint with explicit reasoning/max-output settings. Begin restricted tools, then approved file/edit/check workflows; test revision/late correction, partial side effects, exact duplicate failures, permission boundaries, malformed actions, loops and interrupted calls. Preserve user requirements throughout.
5. Test temporary handoff after real idle, restart without replay, retained authorized evidence, longer unseen tasks, repeated context rebuilds, source faithfulness and privacy. The successful three-workflow test does not mark these complete.
6. Preserve existing memory service/broker/progressive discovery and their tests. OS identities, network restrictions, scoped mounts, secret/store access denial and supervised backup/rollback remain deployment gates. An in-process plugin or restricted launcher is not an OS boundary.
7. Only then measure full Hermes CLI/gateway workflows, concurrency/cancellation, Linux target latency/RAM and long sessions. Keep rollback to the previously validated backend/profile. Native wire/HTTP bridging, physical KV optimizations, vision and optional training are separate later gates.

## 7. Deliverables and status

Active checklist: `PROJECT_FUNCTION_CHECKLIST.md`. Keep existing checked memory/security implementation entries with their dated scope. Add backend/model evidence separately from target validation. Production activation, real Intel/Linux benchmarking, native Bonsai graph and broad Hermes integration remain unchecked.

Deliverables: v5 plan/checklist in both project locations; documented backend boundary and native limitations; portable dry-run launcher/profiles; offline behavioral tests; precise source/model/runtime provenance; target acceptance/optimization runbook; reviewed scoped Windows evidence and explicit pending Linux/Intel work. No model download, conversion, training, inference or deployment is started by this revision.

## Temperature-0 MTP source increment — 22 October 2026

The opt-in launcher now specifies matching greedy server/client defaults (temperature0, top-k1, top-p1, min-p0, repeat penalty1, seed3407, samplers top_k/temperature). CPU MTP-off and one-token Vulkan candidate defaults are retained. Client requests can override server defaults.

`engine/patches/prism-greedy-mtp/` contains a source-only patch pinned to sudoingX/llama.cpp ff414120c343e6e6cb868013c99f1dde52b27e70: direct argmax drafts at p_min0, no probability/sort/RNG work in that draft path, and conditional sampler cloning only when checkpoint/replay is possible. No recurrent/KV rollback or cache ownership rules are changed. Existing batched target verification already provides weight-reuse opportunities. The target top-k1 path already avoids softmax; vocabulary projection remains necessary.

Patch applicability, resulting source hashes, offline launcher contracts and the clone-bound arithmetic were checked. The patch has not been compiled, applied to an installed runtime, or benchmarked. It is not automatically compatible with the separately recommended Prism73bfe6c release. Apply/check only against the exact base; target CPU/Vulkan build, token parity, rejection/replay/cache-state and speed/memory acceptance remain required. No model or inference process was loaded. See `engine/patches/prism-greedy-mtp/README.md` in the Xeno repository for provenance and target acceptance.

## Current deployment scope: research only — 22 October 2026

This narrows the initial deployment, not historical source capabilities: public-web
search/read in one dedicated Brave Origin browser and approved memory tools only.
No coding, shell, arbitrary files, desktop control, downloads/uploads, extensions
or account actions. Existing controlled coding/file-export prototypes are not
authorized for this stage. The memory-only launcher remains unchanged in scope;
automatic browsing is not yet connected or enabled.

Implemented preparation: exact model/memory endpoint audit guard; total service
request-read deadlines; persistent opt-in research approval scope with matching
discovery and submission/approval enforcement. Page approvals remain manual
handoffs, not executed browser actions. Existing manual brokers are not converted.

Fedora acceptance remains unchecked: OS filesystem/network/desktop isolation,
immutable executable configuration, private service transport, mounted Windows
volume denial, public-only browser egress, constrained search/open/read adapter,
and installed Brave Origin policy verification. A Python guard or only one
installed browser is not that boundary. The old host-network deployment recipe
is explicitly a legacy prototype. See
`local_extensions/e4b_memory/deploy/RESEARCH_ONLY.md` and checklist section21.
No application, model, browser, engine, GPU or Linux script was run for this work.

## Browser foundation and validator switch — 22 October 2026

The owner setting `research_only.validator_enabled` defaults to true and is loaded
once at startup. Set false and restart to skip only the extra profile/surface
validator. It does not weaken OS isolation, the fixed browser vocabulary or the
memory-only launcher's strict checks. General CLI/gateway/UI activation is deferred.

Offline browser foundation now exposes staged discovery and fixed search, open,
outline and read. Open returns metadata only. Local section matching, bounded
outlines, exact extracted-text chunks and opaque session/page/operation-scoped
continuations avoid sending a whole long webpage. Serialized UTF-8 budgets adapt
each result and cap cumulative per-turn output. The trusted controller can reduce
the allowance; the model cannot reset it. This is not live token accounting,
conversation compression, semantic summarization or physical KV eviction.

Only an injected fixture backend/owner preview exists; no Brave process or live
transport is connected. Linux isolation, actual adapter and live context-budget
integration remain unchecked. Per user request, no new untrusted-content handling
or rate limiter is added; webpage instruction text remains available for later
model-behaviour tests. Existing memory trust/checkpoint safeguards are unchanged.
Memory rehearsal and model/GPU evaluation remain deferred. See
`local_extensions/e4b_browser/README.md` for settings, bounds and integration seam.

## Intel CPU optimization increment — 22 October 2026

This increment supersedes the earlier blanket no-prefix-reuse configuration **only for the opt-in CPU candidate**. PTQ1 decoder/packing/Hadamard/matrix kernels remain deferred at the user's request. The original CPU MTP-off/prefix-off profile and live Hermes settings remain available and inactive here.

Implemented source: exact-prefix-friendly controlled-state serialization with full source/evidence/correction preservation, local HTTP keep-alive without request replay, separate prompt/decode/draft CPU threads, optional affinity masks and polling, complete-verification-microbatch guards, a CPU-only candidate profile and a dry-run24-plan tuning renderer. The cumulative cpu-mtp-v1 external-runtime patch adds buffer reuse, all-accepted hidden-copy elision, early enforcement of per-sequence draft limits and optional per-request timing-based MTP window selection (1..min(4,cap), four samples/window, a probe every32 valid cycles). It includes greedy-mtp; apply to the clean pinned source, never stack both patches.

Candidate: Linux CPU, device none and target/draft GPU layers0, one slot,8192 context, batch256/ubatch128, six prompt/decode threads, polling0, p_min0, adaptive cap3. Prefix cache is requested with128MiB prompt-cache budget, two additional context checkpoints and idle-slot archiving off. Live KV/checkpoints are outside that budget; >=5GiB MemAvailable safeguard remains. CPU backend sampling is independently selectable, not GPU placement. No CUDA switch is emitted. No affinity topology or fastest thread/batch choice is presumed.

Hermes owner CLI options are `--reuse-prefix --keep-alive`; both are opt-in. They do not activate ordinary Hermes or store/reinject prior reasoning. Prefix eligibility is exact rendered-token identity; hybrid state restoration/recomputation, changed user instructions and removed-reasoning tails need target parity acceptance. No persistent KV cache or automatic inference/tool retry is added. SSD pending-action writes/readback remain mandatory. Existing controlled-run reasoning-off/output1024 defaults are retained; the launcher's low/4096 request defaults are a separate production-provider configuration decision.

The actual C++ policy compiled and passed standalone tests. Python launcher/harness contracts, patch application/reverse/hash checks and dry-run tuning rendering were checked without a model. The full runtime remains uncompiled and unbenchmarked. Adaptive warmup can cost time on short answers and does not compare MTP-off internally. Compare whole-task latency, token/action parity, cache hits, recurrent replay, RAM/swap and sustained thermals before adoption. Do not remove synchronization or recurrent checkpoints on an assumption that CPU execution makes them unnecessary.

Implementation and target procedure: `engine/patches/prism-cpu-mtp/README.md` in Xeno; cumulative patch pinned to sudoingX ff414120c343e6e6cb868013c99f1dde52b27e70. Target compilation/validation and runtime activation remain pending. No model, engine service, GPU or WSL was loaded here; only the standalone policy test binary ran.

