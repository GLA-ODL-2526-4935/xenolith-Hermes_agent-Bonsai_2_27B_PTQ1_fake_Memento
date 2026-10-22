> Superseded model/runtime direction: see [active v5 plan](Local_Bonsai27_Hermes_Xenolith_Memory_Plan_v5.md). This v4 file is retained as dated implementation history; its memory/security requirements remain in force where v5 preserves them.

# Local Gemma E4B + Hermes + Xenolith — implementation plan v4

21 October 2026. **This replaces v3 as the active plan.** Preserve v3 as historical context, not a competing implementation instruction. The user authorized this revision and work specifically on memory; this does not authorize activation, repository moves, engine work or model workloads here.

## 1. Decision and scope

Build a local agent around the original Sol Fable Gemma 4 E4B instruction model and Hermes' existing agent/tool loop. Make the system manage bounded working context through source-linked task/step records. A successful Memento fine-tune is **no longer a deployment prerequisite**. Learned intra-generation Memento remains a deferred research option, requiring fresh evidence of benefit without agent regressions.

The separate six-case synthetic Hermes-style diagnostic found more failures with the Stage 1 adapter, while the original also fabricated state. This motivates testing the harness approach; it does not certify the original model or prove all training harmful. Do not deploy the experimental adapter as a default.

Keep the existing checkout and core structure. Prefer supported context-engine/provider extensions over rewriting the loop. Preserve the modified memory service, access broker, progressive tool discovery and security controls. Do not overwrite other local work or silently activate profiles.

Removed from the baseline: DeskForge/Qwen action weights, Qwen MTP, two resident planner/action models, screenshot-driven desktop control and remote takeover. Bonsai/PQ2 runtime work, GUI automation, MTP acceleration and remote memory maintenance remain deferred.

## 2. Hardware, local-only and model contract

- Target: ASUS Zenbook 14 OLED UX3402VA-KN521W, i9-13900H, Intel integrated/Iris Xe-class graphics, **16 GB shared RAM**, local SSD, Linux; KDE/Wayland preferred. Exact PCI device/driver compatibility requires measurement.
- Only **one resident inference model**. Planning, execution and optional record interpretation use that same service, serially where necessary. No silent hosted inference, paid teacher or remote auxiliary fallback. Messaging transports may use the internet.
- Original Sol Fable is the candidate, not an accepted production model. Pin its model revision, quantization, tokenizer, chat template, licenses, hashes, runtime revision and all auxiliary routes. Keep vision behind a separate processor/runtime correctness gate.
- The existing working reference tests used Q4_K_M. The Xenolith port documents Q4_0 matrices/F32 norms. Do not assume interchangeability or deploy a differently quantized artifact without independent compatibility/quality checks.
- Reference inference remains usable while Xenolith is validated. Train/export only on separate suitable compute if later authorized; do not launch models, engines, applications or GPU workloads in this checkout/on the restricted development laptop.
- Establish startup, prefill/decode latency, total task time, peak RSS, available RAM, swap and thermal budgets on the actual target. Static model size is not total process memory; no throughput or RAM promise is established here.

## 3. Architecture and boundaries

```text
Hermes gateway/CLI → trusted session identity and authorization → existing agent loop
    ├─ existing staged tool discovery and approved tool dispatch
    ├─ scoped curated memory provider/service (L0–L3)
    └─ task/step recovery state → opt-in context-engine policy
                                   ↓
                  validated local reference API / future Xenolith bridge
                                   ↓
                          one original Gemma E4B
```

Separate four state classes:

1. Task contract: original requirements, exact values, current decisions, permissions/prohibitions, supersession and dependencies. Source-linked model proposals never acquire user authority.
2. Active context: current step, necessary evidence, recent complete tool exchanges and needed schemas. Keep the ordinary Hermes prompt/cache prefix stable within a segment. The user-authorized controlled-step prototype below deliberately rebuilds requests after each completed action; measure that explicit exception separately.
3. Recovery: source-linked visible events, controller-reported outcomes, attempted/failed actions, uncertainties and pending work. Routine failures belong here, not curated success memory.
4. Curated memory: approved text evidence and durable L0–L3 knowledge, governed by the existing owner gates and retention. A saved recovery step is neither verified knowledge nor a memory-write approval.

Ordinary Hermes work steps can include several tool calls. The user-authorized controlled workflow instead permits one tool action per step. Neither is a hidden reasoning block. Use explicit task transitions and complete tool-result boundaries; preserve outstanding call/result pairs, partial streams and relevant approval state.

## 4. Memory: current implementation and required increment

Existing opt-in source includes temporary/persistent L0–L3 JSON memory, owner administration, a separate memory service, bounded staged retrieval, review preparation, recovery and idle maintenance. It is single-owner source, not certified live or multi-user deployment. No actual memory is migrated by this revision.

### Structured recovery records — first increment

Add controller-only atomic step submission with session binding, expected revision, immutable step ID, exact visible source events, quoted selected facts, controller-reported status and readback receipt. The record must preserve reported success separately from verification. Reject unknown fields, hidden-reasoning/assistant event roles, media envelopes, forged verification fields, unsupported statuses and invalid quote references.

Keep supplied event text intact; do not silently truncate it. Explicit record/count limits fail without dropping older records. A quote proves correspondence to the supplied source, not truth, completeness, authority or that every important event was supplied. A controller is trusted to select authorized visible evidence; content checks alone cannot detect all encoded media or mislabeled reasoning. OS isolation and controller provenance remain necessary.

Provide staged, session-scoped list/read access under the existing discovery and retrieval limits. Models cannot submit, overwrite, approve or delete step records. Distinct service credentials remain role-scoped and client-prefixed. File import is owner/controller administration only, with strict JSON parsing and no execution of embedded commands.

Security hardening (21 October): validate stored fixed trust labels and structure on
recovery list/read/retry, including metadata integrity for newly written records.
Requirement/decision facts must quote a complete visible user event; tool results
remain observations/proposals, never user authority. Returned source roles and
reference-only/no-permission markers are controller-derived. They are not a claim
that a model will always ignore injection or that the controller's source identity
is independently authenticated by this schema.

Refuse recognizable credential/private-key patterns before saving a recovery step,
using generic errors and no silent rewriting. The controller must select an
authorized non-secret source excerpt instead. This heuristic is not a general
secrets/privacy detector and does not sanitize existing task state/logs/backups.
Step expiry now uses each record's original save time, not the session's newest
step; foreground work still defers deletion. Legacy records violating the stricter
capture rules are refused on read and need owner review, not automatic migration.

Expose this as an explicit API/owner workflow first. Do not silently hook it to every turn, advertise strict checkpoint API v2 or enable context removal merely because disk persistence exists.

### Extraction and checkpoint gate — next increments

1. Build records from typed tool events and exact user clauses; use model proposals only for interpretation that cannot be obtained deterministically.
2. Evaluate requirements, values, status, unresolved dependencies and failure retention against independently authored expected facts. Keep extraction separate from compaction.
3. Run shadow mode without deleting history. Record omissions, inventions, cap/loop failures and unnecessary copying. Test longer unseen histories, not only the current six diagnostic cases.
4. Design strict checkpoint-required compression using the existing core capability contract. Complete event coverage, pending work, evidence availability, readback, concurrency and crash recovery must be proven before declaring v2 support.
5. On checkpoint failure retain the old context and checkpoint. On restart reconcile side effects; never automatically replay tools because an action's status is uncertain.
6. Enable bounded context rebuild only after extraction and checkpoint gates pass. Preserve system prompt stability, tool-call validity, session leases, normal sanitizer/approval paths and rollback.

### Curated memory rules — preserved

- Main-model writes remain append-only, owner/controller-opened expiring stages referencing approved evidence. Model cannot self-approve or administer records. Review candidates are not verified facts.
- Staged section → catalog → selected schema → operation; at most **three expanded records and 16,000 returned characters per turn**, shared with recovery expansion. No full-bank injection.
- Prepare review after **five idle minutes** and pre-compression. Current provider is legacy/best-effort; strict checkpoint-required compression v2 is unfinished.
- Promote after **three idle hours**. Optional same-local-model sorting chooses layers only, cannot rewrite text, rejects stale plans and yields to foreground activity. Limits: **32 records / 60 KB per sorter request**, **30-second timeout**. Large backlogs require owner maintenance.
- Persistent L0 expires **30 days after promotion**; pending review **one day**; recovery **seven days**. No automatic L1–L3 expiry or silent deletion of unpromoted memories. Expiry defers while foreground turns are tracked active.
- Keep exact deduplication; semantic rewrite, contradiction resolution, remote compaction and SQLite/FTS5 memory search are later work. The credential SQLite database is not memory search.
- Selected verified evidence, confirmed facts/decisions/preferences and critical verified lessons may enter curated memory. Routine malformed calls/failures remain recovery-only.

## 5. Context-engine and physical cache distinction

Use `agent/context_engine.py` and the normal supported compression lifecycle. One engine is active. Compose required behavior; do not run competing independent compressors. `select_context` is a request-only replacement seam and does not mutate persisted history; cache-prefix changes must be measured. `on_turn_complete` is best-effort and skips some failures, so cannot be the only recovery capture.

The system prompt/tool surface stays stable during a segment. Dynamic state uses supported message/compression paths, not fabricated user turns or system-prompt rewrites. Existing discovery reset is not evidence of KV reclamation or permission changes.

Rebuilt requests can reduce future token processing, but may require prefill recomputation. Measure logical token counts, occupied KV tokens, reserved cache capacity, physical allocations and process RAM separately. Do not claim a shortened request releases fixed allocated cache memory.

This does not reproduce Memento's trained intra-generation attention masks and cannot stop a loop inside one uninterrupted response. Keep output/liveness/cancellation limits. Learned delimiters, masked continuation and physical block reclamation need a separate model/runtime research gate if pursued later.

## 6. Storage, privacy and scope

Do not persist hidden full reasoning, raw KV, images, thumbnails, base64/media payloads or image-containing caches into memory/recovery/logs. Do not equate full session storage with a reviewed authorized archive. Store only authorized visible text evidence and bounded recovery records; inspect existing replay/log paths before making guarantees.

Evidence references obey retention and authorization. A digest is not recoverable evidence after deletion. Essential facts need an authorized retained source or an explicit unavailable-evidence status. No blanket permanent archive is introduced.

Keep the service bank and owner administration outside the model tool surface. The default-disabled broker provides exact owner-approved manual browser handoffs and fixed-folder no-overwrite text exports; no automated browser is connected. Preserve one-hour exact-payload approval expiry and existing request parser hardening.

The restricted launcher and an in-process plugin are not OS security boundaries. Linux identities, read-only/minimal mounts, network egress, safe cross-namespace transport, disposable browser profile and actual denial of direct store/secret access remain mandatory before broader tools. No shell/file/browser tool is enabled by this memory increment.

Future group scope: authenticated platform/group/participant IDs, group-local shared L0/L1, private per-user L2/L3 and private temporary memory; explicit consent for private-to-shared disclosure. Enforce identical access rules for content, labels, snippets, existence, provenance, session search and caches. Administrators do not automatically gain private-user access. None of this is certified by client credentials for one owner's bank.

Media remains a separate gate: approved outgoing media only; incoming vision through a validated local runtime. RAM-only authorized temporary image buffers, immediate deletion preferred, absolute **24-hour** maximum, daily cleanup and restart clearing. Messaging services may retain their own copies; document swap/crash-dump risks separately.

## 7. Xenolith and serving — preserved work, no implementation in this increment

E4B tensor binding, PLE, dense execution, shared local/global donor cache layout, Q4_0 loading and generic CPU/Level Zero source exist but are unvalidated. Keep `runtime_ready = 0`; no inherited performance claim becomes an E4B measurement. Compile/parity/device validation occurs separately. Refer to `E4B_PORT_STATUS.md` for source details; do not restore the removed large-model/MoE path.

Keep deterministic reference logits/tokens, long positions/ring rollover, shared donors, COW/rewind/snapshots, CPU/GPU parity and cancellation gates before optimization. Add measured batching, scheduling and actual iGPU compatibility later.

The native protocol is text-only, not an OpenAI chat-completions API. A tested thin bridge must handle roles, tool IDs/arguments, streaming, finish reasons, errors and cancellation. Vision is separately gated. Persistent KV snapshots default disabled; audit crash dumps and rebuild from authorized text after restart.

## 8. Revised execution order and acceptance

1. **Inventory and baseline:** pin original Sol Fable/reference runtime/configuration. Test chat, exact tool formatting, error/cancel and no hosted auxiliary fallback in the separate validation environment. Keep profile activation supervised.
2. **Memory foundation (current authorized work):** versioned recovery record contract, atomic controller saves, scoped staged retrieval, restart/failure/idempotency/retention tests. Preserve curated-memory gates.
3. **Extraction/shadow mode:** source-faithfulness first. Require no invented consequential outcomes/approvals/values or dropped hard constraints on the reviewed acceptance set; expand to unseen longer histories before reliability claims. Whole-transcript copying is not successful bounded selection.
4. **Checkpointed context rebuild:** prove coverage and persistence/failure gates, then compare continuation against ordinary original-model context. Include delayed dependencies, repeated compactions, changed requirements, partial side effects, cancellation and crash recovery. Answer accuracy alone is insufficient.
5. **Security/deployment:** supervise live single-owner service integration, Linux isolation, backup/restore/migration/rollback and realistic allowed-tool acceptance. Broaden to authenticated groups only after code-enforced scope gates.
6. **Engine correctness/performance:** preserve Xenolith port and bridge milestones; compare with exact model/reference backend on the target 16-GB laptop. Retain the fallback when faster or more correct.
7. **Later optional research:** compact already faithful records; narrowly targeted training with agent regressions if needed; learned Memento masking/eviction, MTP, remote maintenance and audited GUI helpers only under their own acceptance gates.

Training is not launched by this plan. Existing offline test results are historical scoped checks, not production certification. No live inference or deployment gate is considered passed here.

## 9. Required deliverables

- Reproducible original-model/runtime/configuration manifest and usable reference fallback.
- Documented step/event schema, source/status distinctions, bounded retrieval and atomic recovery protocol.
- Direct extraction, agent regression, checkpoint/restart and later continuation test reports with raw outputs and limitations.
- Curated memory service preserving owner approvals, retention, isolation design and rollback.
- Validated local-only Hermes tool workflows; later measured Xenolith E4B bridge and Intel performance, or precise blockers/fallback evidence.
- Capability matrix distinguishing implemented source, offline tests, live acceptance and deferred features.

The mandatory v3 target of a Memento-trained E4B with physical intra-generation block eviction is explicitly superseded. The new required context target is reliable harness-managed recovery and bounded working context, with measured cache behavior. Keep trained-Memento artifacts/results as research history, not a release requirement or claimed capability.

## Offline foundation increment — 21 October 2026

The existing Hermes extension now has default-off automatic visible-text shadow
capture through supported local/service provider hooks, immutable source steps,
readback-validated manifest candidates, ordered source-linked task views and
owner-declared statuses/supersession. It also provides an exact-transcript offline
checkpoint gate and a lossless rebuild preview. Whole-transcript copying remains
only a baseline, not faithful bounded selection or a token-saving claim.

No live context replacement is enabled and Hermes strict checkpoint API v2 is not
advertised. Unsupported/oversized transcripts, changed prefixes and incomplete
source coverage cannot produce a successful matching checkpoint. Actual model
selection/continuation, abnormal-exit integration, strict compression handoff and
shorter working contexts remain acceptance gates. Shadow source capacity is the
existing 32-step/session limit; library manifests accept at most 200 messages and
128 KB, while service requests retain their smaller 32-KB total wire budget.

Scripted/injected connection contracts, owner inspection/annotation and curated
record-only backup/empty-bank restore commands, configuration opt-in and source
SHA256 inventory are present without starting a model or application. This does
not constitute the native Xenolith HTTP/SSE bridge, a GUI, a complete service
backup/migration tool, a complete release/license inventory or Linux deployment.
See `OFFLINE_FOUNDATIONS_STATUS.md` and checklist section 18 for precise scope.

## Status documents

`PROJECT_FUNCTION_CHECKLIST.md` is the active function checklist; `STEP_RECOVERY_STATUS.md` tracks this increment. `SINGLE_USER_MEMORY_STATUS.md`, `MEMORY_SERVICE_DISCOVERY_STATUS.md`, `RESTRICTED_ACCESS_STATUS.md`, `E4B_PORT_STATUS.md` and `OFFLINE_SECURITY_TEST_REPORT.md` retain their dated scope. This v4 decision takes precedence over their references to a required trained model.


## Controlled-step increment — 22 October 2026

The user explicitly authorized implementing and testing anchored user requirements + model plan + one tool call per step, dropping previous per-step reasoning from subsequent requests. The opt-in `controlled-run` owner command now consumes this workflow inside the existing extension without changing the normal core loop or activating profiles. Separate local model tests were authorized and ran from the chat workspace. See `local_extensions/e4b_memory/CONTROLLED_STEPS.md` for the implementation, limits and results.

This working-context journal is independent of curated layers and prior recovery selection. It persists exact calls/results, preserves corrections, deduplicates identical failure bodies, stops uncertain actions and supports an explicit foreground idle handoff. The initial selected tool surface is read-only within an owner-selected directory. Native tool formatting is bypassed using validated JSON action envelopes.

The two direct current-task tests and one CLI-consumer probe passed, but only two of six harder historical tasks completed; their final answers still had omissions/errors. Do not activate this as the default or claim model acceptance, physical KV reclamation, general tools, full UI/gateway integration or unattended restart. These new tests do not mark earlier unchecked production/security/runtime gates complete.
