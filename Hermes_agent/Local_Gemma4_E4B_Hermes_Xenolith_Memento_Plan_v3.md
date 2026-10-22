# Project: Local Multi-User Hermes Agent with Gemma 4 E4B, Memento, Xenolith, and Hierarchical Memory

> **Historical — superseded on 21 October 2026 by [the active v4 plan](Local_Gemma4_E4B_Hermes_Xenolith_Memento_Plan_v4.md).** Do not execute the instructions below as the current plan. v4 removes successful trained Memento as a prerequisite and preserves memory/security work.

> **Revised implementation brief — Version 3 | 19 October 2026**
>
> This document **replaces** the DeskForge/Qwen3.5 GUI-action-model plans and their associated two-model planner/action pipeline. It describes one Gemma 4 E4B reasoning/tool-calling model, Microsoft's Memento-style inference context management, a Gemma E4B port of Xenolith for the target Intel laptop, and a hardened Hermes Agent extension for group chat and four levels of scoped persistent memory.
>
> **Instruction to the implementation agent:** Inspect the actual repositories, versions, model files, hardware, interfaces, and licenses before coding. Implement and test working increments; do not stop at a proposal. Do not invent supported features, benchmark numbers, or model-training outcomes. Record blockers and use documented fallbacks when necessary.

## Current increment — 20 October 2026

Latest security-hardening follow-up: fixed ambiguous headers and duplicate JSON keys
in both custom service handlers, with strict bounded parsing shared between them.
Both original failing tests now pass unchanged; 34 additional parser cases were added.
Full selected run: 234 passed, zero failed; all 12 static checks passed. Memory policy,
owner approvals and discovery functionality remain intact. See
`OFFLINE_SECURITY_TEST_REPORT.md`; Linux isolation/live acceptance remain unchecked.

This status supersedes older local-only memory and discovery TODO statements below; historical phase descriptions do not imply completion. See `PROJECT_FUNCTION_CHECKLIST.md` for function-by-function status and `MEMORY_SERVICE_DISCOVERY_STATUS.md` for this increment.

- Implemented an opt-in separate memory service in the existing Hermes checkout, with a remote provider that never creates/falls back to an agent-local Store. Local mode remains the compatibility default; nothing is activated or migrated.
- Service owns text JSON memory, staged retrieval/appends, checkpoint review, recovery and idle maintenance. Distinct revocable model/controller credentials cannot approve, edit/delete or administer memory; owner commands stay local to the service host. This is still one owner's bank, not authenticated multi-user privacy.
- Implemented general registered-tool section-first discovery through Hermes' existing search/describe/call bridge: section names first, brief selected catalog second, one selected schema third. Session permissions, normal tool dispatch and approvals remain enforced. Four loop-native tools and custom non-registry provider tools remain direct when enabled.
- Passed 155 selected CPU-only tests on Windows Python 3.14.7 and all 12 static checks. No live service/application/browser/model/engine/GPU was launched or live profile changed.
- Still unfinished: Linux storage/process/network isolation, supervised live service integration and transport, migration/backup acceptance, actual E4B behavior, multi-user authorization, remaining direct-tool deferral and trained Memento context/KV reclamation. Discovery does not erase history or add tool permissions.

## 1. Why the architecture changed

The earlier design depended on **DeskForge-Qwen3.5-4B**, a next-action GUI model, plus an independent model or planner/controller. A trial on a more powerful CPU reportedly required approximately **50 seconds per GUI action**. Running another screenshot-reading planning model would further increase time, memory, and system complexity. This is unacceptable as the main path for a 16 GB Intel laptop.

**New strategy:** reuse Hermes Agent's established agent loop and tool framework rather than building a screenshot-based autonomous GUI agent from scratch. Deploy **one Gemma 4 E4B** model for reasoning, tool use, and, when correctly supported by the runtime, image understanding. Add four-level persistent memory to Hermes, and optimize one model's inference with Xenolith and Memento.

**Removed from the baseline:** DeskForge/Qwen3.5 weights, custom Qwen MTP head, separate planner LLM, screenshot-to-action control loop, Wayland mouse/keyboard automation, guaranteed multi-step desktop control, and Windows remote takeover. A lightweight GUI tool may be explored later, **only after** the text/tool agent meets performance targets. Do not silently carry old GUI acceptance criteria into this project.

## 2. Version 1 outcome and strict constraints

Deliver a **local, group-chat-capable general-purpose agent**, not a dedicated GUI-action model:

1. **Hermes Agent** provides conversation orchestration, planning, tool calling, progress and cancellation, messaging gateway, and session lifecycle.
2. **Gemma 4 E4B**, preferably an appropriate instruction-tuned checkpoint, is the **only required resident inference model**.
3. A functional **reference local inference backend** runs the model immediately; **modified Xenolith** becomes the preferred optimized backend once verified.
4. **Memento is trained and evaluated on a separate training machine**, and the deployment runtime uses learned reasoning-block boundaries and safe KV-cache eviction if supported and validated.
5. Multiple authenticated group-chat participants receive correct responses. The agent maintains **shared L0/L1** and **private per-user L2/L3** memories with mandatory code-enforced permissions.
6. Persistent memory remains **text-only**; images are confined to in-memory, access-controlled, automatically expiring temporary storage.
7. Model inference, reasoning summarization, memory search, and consolidation run **locally**; no cloud model endpoint or paid API may silently become the fallback. Third-party messaging transport itself may use the internet.
8. The system remains usable with **Memento switched off**, and with the reference backend if Xenolith is not ready.

### Hardware and separation of work

- Target: **ASUS Zenbook 14 OLED UX3402VA-KN521W**, Intel **Core i9-13900H**, Intel integrated graphics / Iris Xe-class iGPU, **16 GB shared RAM**, local SSD.
- OS: Linux, with KDE Plasma/Wayland preferred but **desktop control is not required in V1**.
- Train and merge/prepare checkpoints **on another suitable machine**, never on the target laptop.
- Model bundle includes base model revision, tokenizer, chat template, vision processor if used, Memento adapters/weights, quantization metadata, test examples, checksums and license metadata.
- Do not assume the exact Intel iGPU device ID or driver compatibility until tested on the machine.

### RAM expectation, not a performance guarantee

Google's Gemma 4 documentation gives approximately **4.5 GB** for E4B Q4_0 static model weights, excluding additional KV cache, runtime buffers, vision preprocessing, Hermes, OS and other applications. The “E” describes **effective** parameters; E4B uses per-layer embeddings and is not equivalent to a plain dense 4B model. Set and measure explicit peak-RSS, available-RAM, swap, startup, prefill, tokens/second and request-latency budgets; avoid making speed promises before benchmarking.

## 3. High-level architecture

```text
Telegram or Discord group / private chat
                 │
          Hermes messaging gateway
                 │  stable platform + group + participant identity
                 ▼
         Authorization and task policy
                 │
                 ▼
           Hermes Agent loop
        ┌────────┼──────────┐
        │        │          │
  Scoped memory  Tools   Task/session state
   retrieval       │
        │          │
        └─────┬────┘
              │  OpenAI-compatible chat + tool calls
              ▼
        Local inference adapter
            /         \
 Reference E4B       Ported Xenolith E4B
    backend               │
                     Memento-compatible
                     runtime KV eviction
              │
         Gemma 4 E4B
              │
       Text / approved media response

Separate stores:
  (a) Text-only L0–L3 persistent memory and provenance (SQLite + FTS5)
  (b) Durable text-only task/session state
  (c) RAM-only, short-lived image buffers (no persistent image objects)
```

Keep boundaries modular: `InferenceAdapter`, `HermesMemoryProvider` or carefully scoped Hermes fork hooks, `MemoryStore`, `AuthorizationContext`, `ConsolidationService`, `EphemeralMediaStore`, and `ChatMediaAdapter`. Avoid rewriting Hermes' existing tool system, group gateway or basic agent loop unless a proven limitation requires it.

## 4. Gemma 4 E4B model selection and baseline

Official family reference: <https://ai.google.dev/gemma/docs/core.md>.

1. Inspect and pin the **exact E4B instruction model** and tokenizer/chat-template revision. Check its support for native reasoning, function calls and images.
2. Download a compatible Q4_0 or QAT quantized artifact where feasible; verify model/processor compatibility and local license terms. Do not assume a quantized post-fine-tuning artifact is bit-identical to the original QAT checkpoint.
3. Run baseline evaluation on a **known working reference inference backend**, such as a confirmed-compatible llama.cpp build, before modifying Xenolith.
4. Test **single-model** chat, multi-turn history, schema-correct tool calls, tool results, reasoning settings, and error handling. Include group-chat instruction-following tests.
5. Measure both **text-only** and, separately, image-input requests. Vision remains a separate capability gate and is not inferred from successful text-only loading.
6. Confirm the reference backend exports an OpenAI-compatible `/v1/chat/completions` route, with streaming and tool calls in the format Hermes actually consumes.

**Do not** reuse the old DeskForge-specific MTP weights or GUI action prompt format. Gemma 4 has its own family of draft-model options, but MTP acceleration is **not needed** for the first functioning release and must not compete with Memento porting for priority.

## 5. Memento training and validation — separate training machine

Research implementation: <https://github.com/microsoft/memento>.

Memento means **intra-generation reasoning-block management**, not the four-level database memory system. A trained model emits delimiters for reasoning blocks and short continuation mementos. A compatible runtime then masks/evicts the completed reasoning blocks from the active KV cache while keeping the permitted summaries and the current block available.

### Training pipeline

1. Inspect the Microsoft dataset conversion pipeline and vLLM overlay, Gemma E4B training support, tokenizer/chat template and licensing. Its published implementation is **not a drop-in Gemma E4B/Xenolith integration**.
2. Establish a **non-Memento E4B baseline**: reasoning/problem-solving, multi-turn tool use, output validity and latency.
3. Prepare high-quality multi-block reasoning traces and compact block summaries appropriate to E4B, including local tool-calling scenarios. Avoid leaking private group content into the training data.
4. Map or add Memento delimiter tokens to the actual Gemma tokenizer and train with a clear compatibility manifest. Preserve ordinary system, user, assistant, tool and vision message conventions.
5. Choose and document full fine-tuning versus LoRA/PEFT, then export a **deployable merged/compatible checkpoint** if applicable. Confirm it can be quantized and loaded into the planned engine.
6. Evaluate summary correctness, multi-step reasoning, function-call JSON validity, safety instruction-following, refusal behavior and tool-call completion compared with the same checkpoint without Memento behavior.
7. Measure quality and peak KV allocation with the Memento-capable reference implementation where feasible. Report both **with physical eviction** and **summary-only without eviction**, not just model token outputs.
8. Supply model, tokenizer, added-token map, training recipe, checkpoints, hashes, prompts, held-out tests and a documented disable switch.

### Runtime contract

- Understand the exact delimiter sequence: `<|block_start|>`, `<|block_end|>`, `<|summary_start|>`, `<|summary_end|>` (plus any necessary thought wrapper conventions).
- Ensure the runtime does **not** evict an incomplete block, tool-call argument stream, uncommitted answer, currently needed system/tool schema, or evidence required to finish a task.
- Remove only the intended attention-visible positions; correctly update token positions, attention masks, RoPE/position handling and cache indexes. Verify no stale KV entry remains accessible incorrectly.
- Preserve correct tool-call and response formatting across Memento block boundaries.
- Implement a **safe escape hatch**: if output delimiters are malformed or compaction fails, fall back to non-Memento generation or a reliable session reset/rebuild.
- Memento summarization **never directly writes** private or shared long-term memories; external memory consolidation is a separate process with authorization and provenance.

**Caution:** Microsoft's reported Memento speed and memory gains are research results on other models/systems. They cannot be assumed for Gemma E4B on this Intel laptop.

## 6. Xenolith E4B port — a major explicit engineering milestone

Repository: <https://github.com/simoneiacomino/xenolith>.

### Implementation status — E4B execution replacement (20 October 2026)

**Implemented, not validated:** In the combined fork's `engine/` directory,
the active MoE/router/expert execution path and GPU kernels have been replaced
with the dense E4B text path. This includes strict tensor binding, Q4_0 matrix
repacking, memory-mapped PLE token rows, full PLE computation/residuals, dense
GELU FFN, independent global K/V, GQA and proportional/partial RoPE.
Context-sized KV storage now covers only 20 local and four global physical
donors. The final 18 layers reuse donor 22 (local) or 23 (global).
COW, cache mirrors, anchors, snapshots and layout fingerprints were adapted.

**Baseline prefill:** A new generic Level Zero projection kernel is wired in.
Prefill is token-serial with synchronized GPU projections and CPU attention;
optimized batching and asynchronous scheduling are not complete. CPU decode
and F32-activation diagnostic dump paths are present. The diagnostic graph
is shared with inference, not an independent reference implementation.

**Safety state:** `runtime_ready = 0` remains the default validation gate.
The separate validation environment can opt into the implemented path using
`XE_ALLOW_UNVALIDATED_E4B=1`. This does not certify it. Only static source
checks were done here; no compilation, model execution or GPU access occurred.
Correctness checks belong in the user's separate model-testing folder.

**Still unfinished:** Linux compile/link verification; deterministic parity
against HF/llama.cpp; long-context, cache/COW/snapshot and CPU/GPU checks;
optimized prefill and cancellation scheduling; actual RAM/latency measurements;
and the final checkpoint, QAT conversion manifest, tokenizer and chat-template
fingerprint. The supported baseline is Q4_0 matrices/F32 norms, not arbitrary
QAT exports. Vision, audio, OpenAI serving, Hermes, staged tool discovery and
Memento remain deferred.

Detailed status: [`E4B_PORT_STATUS.md`](E4B_PORT_STATUS.md).

**Current fork target:** Gemma 4 **E4B only**. The former model layout is rejected
during GGUF profile detection. E4B execution is implemented but remains gated
pending external validation. Xenolith's published wire protocol
is **text-only** and is **not an OpenAI `/v1/chat/completions` API**. The default
service also has **disk KV snapshot/cache behavior**, which must be evaluated and
constrained to meet the privacy rules.

### E4B architecture compatibility work

1. Pin Xenolith commit and inspect loaders, GGUF metadata requirements, tensor names, weight unpacking, tokenizer/chat template, kernels, layer definitions, and hardware device-ID checks.
2. Compile and validate the E4B port in the separate model-testing environment. Do not run this Linux/iGPU engine or use the GPU on the current Windows laptop; do not restore the removed large-model path.
3. Implement an explicit **E4B model-family adapter**, including the small-model **per-layer embeddings (PLE)**, actual attention layout, normalization, feed-forward layers, sliding/local/global attention behavior where present, positional encodings, tokenizer, embedding/output conventions, and quantization layout, based on the pinned E4B model configuration.
4. Build CPU correctness tests and compare E4B logits/token outputs with a reference runtime on deterministic inputs before GPU optimization.
5. Validate and optimize memory allocation, thread scheduling, Intel CPU decode, Level Zero iGPU prefill, data transfer and shared-memory contention. Check the actual iGPU PCI ID and permissions on the ASUS machine.
6. Add **Memento-aware KV block masking and physical reclamation** only after normal E4B KV management is correct. Test evict/rebuild, boundary states, multi-turn conversation, cancellation and malformed delimiters.
7. Add **OpenAI-compatible serving** through a *thin bridge or server adapter*: `/v1/chat/completions`, streaming deltas, roles, tool-call IDs/arguments, finish reasons, cancellation and error mapping. First verify Xenolith's native tool-call emission and wire semantics; do not pretend the native protocol already has this route.
8. Add vision/image input support **as a separately measured feature**, including the E4B vision encoder/processor and image-token formatting, **only if** complete reference correctness can be established. The current text-only wire protocol cannot transport images unchanged.
9. Make on-disk KV snapshots **disabled by default** (`--cache-budget 0` for stock Xenolith where applicable); audit custom code paths for snapshotting, crash dumps and accidental raw reasoning persistence. Restart by reconstructing from authorized text-only state.
10. Run comparative benchmarks against the same Gemma E4B checkpoint on the reference backend. Report when Xenolith is slower or less capable; retain the fallback.

### Required correctness gates

- E4B full forward correctness and text generation.
- Multi-turn context growth and cache reuse.
- Correct tool-call payloads accepted by Hermes.
- Stream and cancellation behavior under concurrent group requests.
- Memento delimiter detection + real KV memory release, not only shorter logical prompts.
- No cross-user/cache/session leakage and no persistent visual/KV artifacts.
- Peak total RAM and task latency on the actual **16 GB i9-13900H** laptop.
- Optional image-input parity separately from text-only parity.

The inherited engine's historical performance figures must **not** be repurposed as E4B estimates.

## 7. Hermes Agent integration — reuse first, customize only what is missing

Official: <https://github.com/NousResearch/hermes-agent> and <https://hermes-agent.nousresearch.com/docs>.

Hermes already offers an agent loop, tools, Telegram/Discord and other gateways, sessions, built-in bounded memories and optional memory providers. Its **custom provider** can point to a local OpenAI-compatible API. Therefore:

1. Clone/pin Hermes and run its existing automated tests before any fork.
2. Connect a known compatible Gemma E4B reference server using `model.provider: custom`, `model.base_url` and a local model identifier. Set **all auxiliary model slots** (compression, vision, background review, curator, etc.) to the local main model or disable them. Verify no silent fallback to hosted services.
3. Test real Hermes tool calls, streamed output, error handling, cancellations, and gateway message replies against the single Gemma model.
4. Replace the reference endpoint with the Xenolith bridge when ready, **without** changing Hermes' high-level agent behavior.
5. Keep one active inference model and queue/gate expensive requests to avoid multiple full model instances in 16 GB RAM. Do not automatically spawn model-heavy subagents or parallel model endpoints.
6. Extend the identity/routing layer to pass a trusted `platform_id`, `group_id`, `participant_id`, authenticated role and authorized scopes **into every memory read and write**.
7. Integrate a purpose-built scoped memory provider/tool implementation; do not rely on prompt wording to choose which users' memories may be read.
8. Review Hermes' built-in `MEMORY.md`, `USER.md`, session stores, auto-learning, memory provider hooks and FTS session search. The default built-in memory is **profile-wide** and can leak data in a shared-user deployment. **Disable or isolate those built-in writes/injections where they conflict with the new scoped store**. Do not merely add a database while the original global memory can still expose private information.
9. Preserve Hermes updates by keeping patches small, documented and covered by integration tests. Prefer a plugin/provider or well-defined fork hooks over broad rewrites.

### Security boundary

**20 October single-user access increment:** standalone `local_extensions/e4b_access/`
now supplies a default-disabled staged proposal tool, a separate-process
proposal/status-only broker implementation, owner-only exact-payload approval and
fixed-folder, no-overwrite plain-text exports. Browser/download/upload/form/message/
purchase/account requests are manual owner handoffs only; no automated browser is
connected. The memory-only launcher remains unchanged and rejects extra tools.
The broker is not deployed, does not isolate the existing memory store, and is not
an OS boundary when run under the same unrestricted account. Linux identities,
mount/network isolation, browser profile/egress/download policy, safe cross-namespace
transport and live adversarial tests remain required. See the functional checklist
section 15 and restricted-access status for completed versus deferred controls.

Hermes' own security documentation describes a **single-tenant trust model** and warns that in-process heuristics are not security boundaries. Multi-user group deployment is an **extension with extra responsibility**, not a supported security guarantee out of the box. Use operating-system/container isolation for untrusted tools, disable unnecessary host shell/code execution, restrict filesystem mounts and network, and enforce per-caller scope at the tool and database layers. An allowlist alone does not provide per-user capability separation inside one authorized group.

## 8. Four-level hierarchical persistent memory

### Single-user-first revision — 20 October 2026 (expanded implementation)

Build and validate for ONE local owner before multi-user integration. The Hermes
checkout now contains an opt-in standalone provider source at
`local_extensions/e4b_memory/`; Hermes has not been moved and no live profile has
been configured. This is implemented source, NOT a validated deployment.

The revised memory policy supersedes conflicting details below for this prototype:

- Both temporary and persistent text memory have L0–L3 sections. L0 contains
  selected verified evidence, not a raw transcript or every failed action.
- Preserve successes, confirmed facts/decisions/preferences and critical verified
  failure lessons; omit ordinary wrong tool calls, malformed arguments and guesses.
  Unfinished-task/error recovery belongs to separate task state, not success memory.
- Main-model access is on-demand section discovery, brief operation catalog,
  selected schema, then call. No automatic injection of the entire memory store.
- Main-model writes are append-only, require an owner/controller-opened expiring
  checkpoint and reference approved evidence. The model cannot open checkpoints,
  correct/delete records, or directly access memory files. Human edit/delete remain.
- Checkpoint review requests are now prepared automatically after five idle minutes
  and at Hermes' pre-compression hook, using bounded tool-result assertions only.
  Those are unverified candidates: the owner selects/approves evidence IDs before
  the model receives an append stage. Manual owner-supplied evidence remains supported.
- After three idle hours the provider promotes temporary records atomically.
  Optional semantic sorting calls the SAME explicitly configured loopback main model
  and accepts destination layers only, never replacement text. Foreground activity,
  owner edits, invalid replies or stale revisions prevent the commit. Default
  deterministic promotion remains available without any model call.
- Memory content is human-readable bounded JSON, rather than SQLite/FTS5 in this
  first increment. The owner can edit through controller commands or edit the file
  with Hermes stopped. Raw reasoning/KV/media are never copied into this store.
- Future multi-user behavior remains group-scoped L0/L1, owner-private L2/L3 and
  fully private temporary memory. Private-to-shared promotion requires approval.
  None of those multi-user authorization guarantees are implemented yet.
- Separate-laptop compaction is deferred. Existing main-model memories are not
  rewritten by consolidation; remote maintenance will need its own authorization.

Storage, staged tools, controller commands, automated review preparation, idle
sorting/promotion, separate recovery state and retention are implemented. Retrieval
is capped at three expanded records and 16,000 returned characters per turn.
Persistent L0 expires after 30 days from promotion; review requests after one day;
recovery state after seven days. Unpromoted memories are not silently discarded;
L1–L3 have no automatic expiry. All expiry defers while turns are active.

39 CPU-only contracts passed through Hermes' canonical runner on Windows Python
3.12; targeted lint, code-health and all 12 repository static checks passed. No Hermes application, model,
engine or GPU was run. These results do not certify the production Python 3.14
runtime, real E4B behavior or Linux deployment. A restricted profile generator/
launcher enforces the memory-only surface and loopback Python networking. An
optional Linux container recipe is supplied but has not been built/deployed.
Direct-storage denial with arbitrary work tools still requires an external store
or OS isolation; an in-process plugin alone cannot provide that boundary.
Task-specific truth verification, second-laptop maintenance, multi-user ACLs,
trained Memento and real-model/end-to-end acceptance remain unfinished.

The following table and multi-user details describe the later target, with the
revised L0 policy above taking precedence. Correction/deletion below are OWNER
operations, never tools available to the main model.

Use **SQLite + FTS5** as the initial local text-only database. Store records with explicit `platform_id`, `group_id`, `owner_user_id`, `scope`, `layer`, `label_path`, `content_text`, `evidence_ids`, `confidence`, timestamps, version and retention. A database constraint and application-side schema must prevent binary/image data in persistent memory records; audit file attachments and logs separately.

| Layer | Scope | Purpose | Default retention |
|---|---|---|---|
| **L0 — shared selected evidence** | One authorized group | Verified text evidence for successes, confirmed facts and critical failure lessons; timestamps and provenance, not a raw transcript | **30 days**, configurable; expiry not implemented in prototype |
| **L1 — shared durable knowledge** | One authorized group | Verified group decisions, shared project facts, confirmed results, useful group knowledge | Until superseded, corrected or deleted |
| **L2 — individual contextual memory** | One participant | Personal workflows, ongoing tasks, private project context and history summaries | Until obsolete or deleted |
| **L3 — individual long-term preferences** | One participant | Explicit preferences, persistent instructions, habits with sufficient evidence | Until revoked or deleted |

**Mandatory isolation rules:**

- Group A's L0/L1 is not automatically visible in Group B. Group-wide is **not** global across every group.
- A participant's L2/L3 is private to that participant, even when queried within a group thread.
- The model may use the requesting participant's private memory to personalize an authorized response, but **must not disclose the private fact itself to the group** without explicit owner permission. For highly sensitive private tasks, move processing/replies to an authenticated DM.
- Identity keys are platform-scoped. Do not equate Telegram ID `123` and Discord ID `123` without explicit account linking.
- **Labels, search snippets, existence checks and provenance** obey the same permission rules as full record contents.
- Admins may manage group memory according to configured group policy, but do not automatically gain access to individuals' private memories.
- Enforce permissions in SQL/query scopes, memory APIs and tests. Never put private memories into a general-purpose Hermes global prompt snapshot.

### Expandable labels and retrieval

Support `memory_list`, `memory_search`, `memory_expand`, `memory_trace`, `memory_correct`, `memory_delete`. Return only authorized category labels before text expansion. Start with 5–15 label hits, 1–3 expanded records and approximately 800–1,200 tokens of evidence selected per turn; make all limits configurable.

Use provenance and versioning, contradiction flags, evidence/retention invalidation, atomic writes and FTS5 indexes. Avoid requiring an embedding model, cloud vector DB or additional resident memory LLM in the first release.

### Hermes integration strategy

- A single local inference service may serve multiple **isolated conversation contexts**; isolation does not require loading a model per user.
- Treat Hermes' built-in memory tools as an integration point to **replace or restrict**, not as a ready-made L0–L3 store.
- Ensure Hermes auto-learning, skills, cron tasks, session search and auxiliary summarizers cannot read a different user's private records by bypassing the custom memory API. Add allowlists or disable incompatible behaviors until scoped.
- Provide an explicit migration/rollback plan and make it clear which old Hermes memory files are disabled, quarantined or group-scoped.

## 9. Two-stage memory consolidation and context maintenance

**Do not confuse these stages with Memento.** Memento changes the **currently generated reasoning KV context**. The following stages create and maintain **persistent text knowledge**.

### Stage 1 — safe checkpoint

**Revised prototype:** automatic review requests are wired at five minutes of idle
time and the pre-compression hook, not a separately calibrated 75% token threshold.
Tool success flags are review candidates, not automatic truth verification. Owner
approval opens the append stage. Verified lessons go to temporary L0–L3; unresolved
work and latest error stay in a separate, bounded session recovery record. This
provider does not claim the strict transcript-archiving checkpoint API v2 gate.

Trigger at roughly **5 minutes of relevant inactivity** or approximately **75% of the configured usable conversational context**. Wait for a safe tool-call boundary.

- Summarize current task status, user/group text events, verified facts, unresolved decisions and evidence IDs **only within the authorized scope**.
- Write an atomic, text-only checkpoint and independent task-state record before any context reset.
- Use Hermes' supported compaction hooks if safe; disable conflicting default compaction/replay paths if they can cross privacy scopes.
- Reset/rebuild chat context/KV safely. Do not repeatedly checkpoint unchanged state.
- **Never** promote everything straight to L1–L3. Do not persist hidden full reasoning traces or image buffers.

### Stage 2 — durable consolidation

**Revised prototype:** after three idle hours, the worker either promotes records
unchanged using their selected layers or optionally requests destination layers
from the same explicitly configured local main model. Text cannot be rewritten;
invalid/stale plans preserve temporary records. Classification is limited to 32
records/60 KB per request with a 30-second network timeout; oversized backlogs
require owner maintenance. Deduplication is exact, not semantic. Contradiction
resolution and second-laptop compaction remain deferred.

Trigger after approximately **3–4 hours of inactivity**, never while a task is active. Run on the **same local Gemma model**; queue it to avoid making foreground chat unusably slow.

- Inspect pending Stage 1 checkpoints and relevant original text evidence.
- De-duplicate, detect contradictions, classify evidence into permitted L1/L2/L3 categories and respect existing approvals.
- Require explicit consent or strong repeated evidence for significant L3 preferences.
- Preserve provenance/version history, mark processed checkpoints transactionally, and apply retention/deletion policies.
- Restart safely after crashes; make jobs idempotent.

L0 retains its configured raw evidence window. A purge may invalidate derived claims: preserve explicit provenance status rather than promising deleted original evidence remains inspectable.

## 10. Image and group-chat response policy

Keep the useful chat capabilities from the previous revision without reintroducing screenshot-driven GUI control.

- **Outgoing:** Hermes may reply with **text**, an **approved image/file already available from a tool**, or **text plus image/caption** when the chat platform supports it. Gemma E4B understands images but **does not by itself create image files**; generating or editing images requires a separate approved tool or application.
- **Incoming:** Image attachments may be read into RAM and passed to the Gemma vision processor **only after** a compatible local vision backend is tested. Until Xenolith supports E4B image input, use a local reference backend for vision or report vision as unavailable; never silently send an image to a cloud model.
- **Temporary images:** use bounded RAM buffers or explicitly RAM-backed storage with per-group/user access scopes, daily cleanup and an absolute maximum age of **24 hours**. Prefer immediate deletion after use; clear on restart. Keep image-specific crash reports, telemetry and swap/page-out risk under review.
- **Persistent memories:** only authorized, appropriately labeled **text descriptions** and source references are allowed. No saved images, thumbnails, base64, embeddings, raw image payloads or image-containing KV caches in L0–L3, checkpoints or durable logs.
- **Privacy:** Do not forward a private user's incoming image to a group, or publish desktop/private images, without appropriate explicit authorization. An image sent through Telegram/Discord may remain on that service even after local temporary media is deleted; document this distinction.

## 11. Tool use, group behavior and permissions

- Authenticate by stable platform account ID; explicitly authorize which rooms/channels and people can address the bot. Keep group and DM identities distinct.
- Queue model/tool work with configurable concurrency and deadlines. Only the permitted participant/role can cancel, approve or modify a task.
- Require confirmation for consequential tool actions: sending external messages, purchases, deletions, software installs, broad filesystem modifications, sharing private data and security changes.
- Treat quoted messages, documents, webpages, model output and retrieved memories as **untrusted** and never as authority to elevate permissions.
- Sandbox shell/code tools when enabled, and default-disable risky tools for ordinary group users. Restrict filesystem roots, secret access and network egress.
- Preserve durable structured task state (status, owner, evidence, last verified step, approval requests, retry count, recovery policy) independent of L0–L3.
- Keep one authenticated operator-controlled escape hatch: stop/cancel active work, pause gateway, disable a misbehaving plugin and restore the previous model bundle.
- Multiple simultaneous participants may share one inference engine but must never share unauthorized private prompt context or model cache contents.

## 12. Performance measurement and early feasibility gates

The first and most important goal is to **avoid the old two-model 50-second-per-GUI-action bottleneck**. Do not assume the new setup is fast merely because E4B is smaller.

Measure on the actual target (and label any development-host results separately):

- Cold and warm startup; static weights; peak RSS; system free RAM; iGPU shared-memory consumption; swap activity and memory fragmentation.
- Prompt processing tokens/sec, generation tokens/sec, **time to first token**, time to first correct tool call and complete tool-task wall time.
- Latency of short text chats, memory-augmented replies, 2–4-step tool tasks, and image-input requests where supported.
- Relative performance of reference backend vs Xenolith, Memento disabled vs summary-only vs real KV eviction, long-context tasks and after many conversational turns.
- Quality regressions after fine-tuning and quantization, schema-valid tool-call rate, retrieval quality and memory isolation tests.
- Sustained runs while normal desktop applications are open; thermal throttling and worst-case latency, not only the median.

**Go/no-go rule:** establish a working, responsive text/tool version on the laptop **before** committing major time to the Xenolith port or advanced trained Memento behavior. Define explicit acceptable latency and available-RAM thresholds from real user tests; if they cannot be met, reduce context limits, quantization and active background work, or reassess the chosen engine/model. Never present projected latency as measured throughput.

## 13. Implementation phases (new order)

### Phase 0 — inventory and feasibility

Pin repositories, inspect target/development hardware, verify E4B artifact, Gemma processor/tool template, Hermes gateway API, inherited Xenolith implementation, Memento research implementation and all licenses. Write a short risk list and benchmark plan. Identify what can be developed before the target laptop becomes available.

**Gate:** known working model/engine candidates, Hermes install verified, explicit unsupported-capability matrix.

### Phase 1 — one-model local Hermes baseline

Install Hermes in a reproducible isolated environment and connect it to a functional **local E4B reference backend**. Disable external LLM fallbacks and auxiliary models. Test chat, tools, streaming, cancellation and group delivery.

**Gate:** authenticated group message → single local model → valid tool execution/result → group response, with measured RAM and latency.

### Phase 2 — security and four-level memory

**Current increment:** single-owner CLI provider implemented, not activated in a
live profile. Temporary/persistent L0–L3, bounded staged retrieval, append-only
checkpoint writes, owner maintenance, automatic review preparation, recovery,
retention and optional local sorting are present. All 39 CPU-only contracts pass;
targeted lint, code-health and all 12 repository static checks pass. Real-model/CLI acceptance, Linux sandbox
deployment and then multi-user authorization remain required before this phase passes.

Implement the SQLite/FTS5 schema, identity model, access filtering, group L0/L1, per-user L2/L3, provenance, correction, delete and tests. Restrict or disable incompatible Hermes built-in memory/auto-learning behavior. Keep the selected Gemma E4B inference and Hermes orchestration unchanged.

**Gate:** malicious and accidental cross-user, cross-group and metadata read/write attempts fail in code; shared knowledge remains available to authorized participants.

### Phase 3 — persistent state and consolidation

Implement atomic Stage 1 checkpointing, idle Stage 2 consolidation, hierarchical retrieval, bounded context rebuilding, restart and task recovery. Consolidation uses the same E4B model during idle slots.

**Gate:** memory survives restart, unrelated private records do not appear, no raw image bytes persist.

### Phase 4 — group media and usability

Implement text+image outgoing replies using existing Hermes chat adapters, RAM-only temporary image storage with daily/24-hour purge, provenance-aware image descriptions, permission-based sharing. Enable incoming image understanding via a validated **local reference backend** if supported.

**Gate:** authorized text+image response works, and deletion/storage-audit tests pass.

### Phase 5 — Memento training (separate compute)

Generate/curate training traces, fine-tune E4B, validate baseline reasoning/tool-call parity, quantize and export an artifact bundle. Carry out this work in parallel with Hermes/memory development where practical.

**Gate:** repeatable checkpoints and held-out tests, including correctly emitted block/summary boundaries; documented training shortcomings.

### Phase 6 — Xenolith E4B correctness port

Implement Gemma E4B tensor/PLE/attention/quantization support and tokenizer/template compatibility. Pass deterministic CPU/reference comparisons before altering GPU kernels.

**Current phase status:** `partial — architecture foundation only`. Model
profiles and E4B metadata recognition are implemented with a fail-closed runtime
gate. Tensor binding, PLE, dense forward execution, shared KV, CPU correctness,
quantized loading, and reference comparisons remain unfinished. This phase has
not passed its gate.

**Gate:** complete correct E4B text generation with quantized model weights and bounded memory.

### Phase 7 — Xenolith on Intel i9/iGPU

Validate GPU device ID and Level Zero, optimize prefill/decode and memory use for 16 GB shared RAM, and benchmark reference vs Xenolith on the exact laptop.

**Gate:** stable long-running service with measured comparative latency and RAM, or documented reason to retain reference backend.

### Phase 8 — Memento-compatible Xenolith cache

Implement runtime delimiter handling, logical masking and **physical** KV eviction; verify numerical and semantic behavior against a Memento-capable reference. Handle malformed blocks, tool calls, cancellations, reset and memory leaks.

**Gate:** demonstrable memory release and acceptable quality/latency; fallback disables Memento if this cannot be validated.

### Phase 9 — Xenolith↔Hermes bridge and final hardening

Expose a tested OpenAI-compatible local endpoint for Xenolith and integrate Hermes streaming/tool calls, single-model concurrency, model bundling, upgrades/rollback, all authentication and audit tests. Enable optional E4B vision support on Xenolith **only if independently correct**.

**Gate:** multi-user end-to-end Linux acceptance tests and benchmark report with no cloud model inference.

### Phase 10 — optional later extensions

Only after reliable performance: optional audited GUI helper tools, advanced Gemma-native draft/MTP acceleration, manual second-laptop memory maintenance with approval-only changes, live remote desktop, and more platforms. **None** of these are Version 1 blockers.

## 14. Suggested repository/fork layout

```text
local-hermes-e4b/
├── vendor/                     # pinned upstream refs (or git submodules)
│   ├── hermes-agent/
│   ├── xenolith/
│   └── memento/                # research references; do not blindly merge
├── patches/
│   ├── hermes/                 # minimal gateway/memory integration patches
│   └── xenolith/               # E4B architecture, KV, service patches
├── integration/
│   ├── inference_adapter/
│   ├── hermes_bridge/
│   ├── identity_auth/
│   ├── memory/                  # schema, FTS5, provenance, CRUD, ACL
│   ├── consolidation/
│   ├── media/                   # RAM-only storage and approved outgoing image
│   └── task_state/
├── training/
│   └── memento_e4b/            # separate-machine recipe/config/eval
├── tests/
│   ├── inference_correctness/
│   ├── memento_cache/
│   ├── privacy_acl/
│   ├── memory_retention/
│   ├── hermes_gateway/
│   └── end_to_end/
├── benchmarks/
├── config/
├── scripts/
├── docs/
└── README.md
```

The structure is illustrative. Use supported upstream extension mechanisms if available and change only what is necessary. Pin versions and make forks/reproducible patch application explicit.

## 15. Required deliverables and completion criteria

**Deliverables:**

1. Working local Hermes Agent configured to use one Gemma 4 E4B model, with no cloud-model fallback.
2. Recorded baseline latency/RAM and comparative results on actual target hardware once available.
3. Tested authorization/identity handling for group and DM use.
4. Fully tested SQLite/FTS5 four-layer, provenance-backed, privacy-scoped memory and text-only persistence.
5. Memory retrieval, correction, deletion, retention, Stage 1 checkpoint and Stage 2 consolidation.
6. Text/image group replies and controlled in-RAM media retention, with explicit backend-capability status for incoming-image understanding.
7. Memento training recipe, artifact/version manifest and model-quality evaluation.
8. A serious, tested implementation of Xenolith E4B architecture support, correctness checks, iGPU optimization and Memento block-mask/eviction; document any part still blocked and preserve a functioning fallback.
9. Thin Xenolith-to-Hermes compatibility bridge or a documented reason the reference inference backend remains in use.
10. Installation, backup/restore and rollback instructions, reproducible tests, performance report, security/privacy review and limitation list.

**Intermediate functional milestone:** From an authorized group chat, participants can ask the same local Gemma model to answer questions and execute allowed tools; the system keeps group knowledge shared, individuals' context/preferences private, persists **text-only** memories with evidence, survives a restart, and does not require a second concurrently loaded planner model. The reference local backend keeps the agent usable *during development*; it is not a substitute for attempting the requested Xenolith/Memento work. Label every uncompleted integration unambiguously.

**Full project target:** The same Memento-trained Gemma 4 E4B runs through modified Xenolith on the 16 GB Intel laptop, with correct Hermes tool calling, measured performance, and physically verified KV eviction. This target is a **required engineering workstream**, not an assumed existing capability or a reason to block the usable intermediate milestone. If it proves infeasible on the hardware, provide reproducible evidence, the specific blocker, and the closest functional fallback.

## 16. Working instructions for an implementation agent

- Start by inspecting the actual source repos, model config, tool schemas and development hardware. If target laptop is not available, do not claim to have run target benchmarks.
- Provide a concise implementation plan, then begin Phase 0 and Phase 1. Run tests and resolve failures at every phase; continue to implement working changes rather than stopping at architecture diagrams.
- Maintain a running capability matrix: `working`, `partial`, `unsupported`, `not yet tested`; annotate the exact repository commit/model checkpoint used.
- Keep inference, chat framework, external persistent memory and Memento separate and replaceable. Never introduce a second resident LLM merely for auxiliary summaries or planning.
- Use security-by-construction and OS isolation for multi-user access, not only prompt rules or Hermes' single-tenant defaults.
- Prefer a correct, fast, modest local group assistant over an impressive set of unverified model/inference features.

## 17. Primary upstream references (verify current versions)

- Google Gemma 4 model overview and E4B quantization: <https://ai.google.dev/gemma/docs/core.md>
- NousResearch Hermes Agent: <https://github.com/NousResearch/hermes-agent>
- Hermes local/custom model providers: <https://hermes-agent.nousresearch.com/docs/integrations/providers>
- Hermes default memory: <https://hermes-agent.nousresearch.com/docs/user-guide/features/memory>
- Hermes security posture: <https://github.com/NousResearch/hermes-agent/blob/main/SECURITY.md>
- Xenolith source: <https://github.com/simoneiacomino/xenolith>
- Xenolith text-only wire protocol: <https://github.com/simoneiacomino/xenolith/blob/main/PROTOCOL.md>
- Microsoft Memento method and implementation: <https://github.com/microsoft/memento>
- Microsoft Research Memento explanation: <https://www.microsoft.com/en-us/research/articles/memento-teaching-llms-to-manage-their-own-context/>

---

**Version history**

- v1: DeskForge GUI action model and custom multi-user controller.
- v2: added image replies, temporary visual storage, Wayland adapter and explicit Xenolith/vision port.
- **v3: replaces DeskForge and separate planner with Gemma 4 E4B + Hermes; keeps four-level memory and group media; adds E4B-specific Xenolith port and Memento runtime requirements; desktop automation deferred.**
