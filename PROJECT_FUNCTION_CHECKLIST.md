# Bonsai 27 + Xenolith + Hermes — functional checklist

Version5 — updated22October2026. Active plan: Local_Bonsai27_Hermes_Xenolith_Memory_Plan_v5.md. Bonsai 2 27B PTQ1 Lean plus the controlled-step harness is the new model candidate; Linux Prism CPU/Vulkan is the additive backend route. Earlier Gemma entries below retain their historical/native-source scope and are not Bonsai implementation claims. Prior checked memory/security items are preserved; target runtime/deployment acceptance remains separate. No inference, model loading, Linux application, GPU or WSL work is authorized on this Windows host by this revision.

**How to read this:** `[x]` means that specific implementation or documentation exists. It does **not** mean the complete feature is deployment-ready. Runtime acceptance tests are separate unchecked items. `[ ]` means unfinished, deferred or not yet verified. Existing upstream Hermes capabilities are not counted as completed project integration.

## 1. Project scope and repository organization

- [x] Focus the engine fork on Gemma 4 E4B; remove the former large-model/MoE execution path.
- [x] Copy the engine into the combined project repository and provide a brief project README.
- [x] Keep Hermes in its existing checkout while its changes are developed.
- [x] Separate inference, Hermes orchestration, curated text memory, task/step recovery and deferred trained Memento responsibilities.
- [x] Make the first memory implementation single-user rather than claiming multi-user security.
- [x] Keep model-specific correctness testing for a separate validation environment.
- [x] Preserve the current restriction: no engine/model/application launch or GPU use on this laptop.
- [ ] Move/integrate the modified Hermes checkout into the combined repository after changes are ready and the owner authorizes the move.
- [ ] Finalize reproducible repository pins, patch/export process and complete dependency/license inventory.

## 2. Historical E4B model bundle and reference backend — conditional, not current baseline

- [ ] If the historical Gemma route is revived, pin and accept original Sol Fable; its Stage1 adapter remains experimental. This is not a prerequisite for the new Bonsai route.
- [ ] Pin model revision, tokenizer, added tokens, chat template and any vision processor.
- [ ] Supply base-model provenance, quantization metadata, checksums and license information; training manifests are needed only for optional trained artifacts.
- [ ] Verify the actual QAT export and its conversion into the engine-supported format.
- [ ] Establish a working local reference backend in the separate model-testing environment.
- [ ] Validate chat, multi-turn history, reasoning settings, tool calls and tool-result formatting.
- [ ] Validate streaming, finish reasons, cancellation and error responses.
- [ ] Validate an OpenAI-compatible local API against what Hermes actually consumes.
- [ ] Verify the reference fallback remains usable with Xenolith and Memento disabled.

## 3. Xenolith: preserved native E4B loading and text execution

These checks describe **implemented source only**, not proven numerical correctness.

- [x] Add E4B-only profile/metadata recognition and reject the former model layout.
- [x] Replace active router/expert execution with the E4B dense text path, including per-layer embeddings (PLE).
- [x] Implement strict tensor-name/type/shape binding and rejection of unsupported formats.
- [x] Implement the baseline Q4_0 matrix/F32 norm loading and repacking path.
- [x] Implement mapped PLE token rows, PLE projection/normalization/mixing and residual computation.
- [x] Implement dense GELU-gated FFN, normalization, output projection/tied embeddings and logit handling.
- [x] Implement separate K/V projections, GQA and local/global attention layouts.
- [x] Implement proportional/partial RoPE handling.
- [x] Implement shared-KV donor layout and context-sized physical cache allocation.
- [x] Adapt COW, cache mirrors, anchors, snapshots and layout fingerprints to E4B.
- [x] Add baseline CPU execution and generic Level Zero projection code.
- [x] Add diagnostic layer/logit dumps and descriptor-test source.
- [x] Keep execution fail-closed by default with `runtime_ready = 0`.
- [ ] Compile/link the engine on compatible Linux hardware.
- [ ] Run descriptor tests and independently validate the full E4B graph.
- [ ] Compare deterministic forward states, logits and generated tokens against the exact model in a reference runtime.
- [ ] Validate CPU/GPU, prefill/decode, shared donors, long positions and ring rollover.
- [ ] Validate COW, rewind, snapshot restore, multi-turn sessions and cancellation.
- [ ] Verify the final tokenizer/template/export manifest against the trained model.

## 4. Xenolith: performance, serving and cache privacy

- [x] Provide a basic token-serial prefill path; synchronized projections and CPU attention are present.
- [ ] Optimize batched prefill, attention kernels, transfers and shared-memory contention.
- [ ] Implement/validate asynchronous scheduling, progress and timely cancellation.
- [ ] Validate the actual Intel device ID, Level Zero driver and permissions.
- [ ] Build the Xenolith-to-Hermes API bridge: roles, tool-call IDs/arguments, streaming, finish reasons and errors.
- [ ] Verify tool calls emitted through the bridge are accepted by Hermes.
- [ ] Queue expensive requests while keeping only one inference model resident.
- [ ] Disable persistent KV snapshots by default and audit snapshot/crash-dump paths for privacy.
- [ ] Reconstruct sessions from authorized text state rather than relying on persistent raw KV/reasoning.
- [ ] Measure target-laptop startup, prefill/decode speed, latency, RAM, swap and thermal behavior.
- [ ] Compare Xenolith with the same model on the reference backend; retain the fallback if needed.

## 5. Hermes: local-model configuration and integration

- [x] Implement an opt-in standalone memory provider without rewriting the core agent loop.
- [x] Provide a fresh-profile generator; refuse to overwrite an existing profile.
- [x] Configure the generated restricted profile for one explicit local main-model route, disabled hosted fallbacks and pinned auxiliary routes.
- [x] Require incompatible built-in memory writes/injections to be disabled before this provider activates.
- [x] Refuse gateway/subagent activation in the current single-owner prototype.
- [x] Provide a restricted memory-only launcher with tool/provider checks and a single-owner profile lock.
- [ ] Activate a test profile in the separate validation environment.
- [ ] Verify actual E4B chat/tool use, streaming, errors, cancellation and session recovery through Hermes.
- [ ] Connect Hermes to the validated Xenolith bridge.
- [ ] Expand beyond memory-only tools after storage isolation and tool sandboxing are ready.
- [ ] Verify all real runtime/auxiliary paths cannot silently call hosted model services.

## 6. Progressive tool discovery and context limits

- [x] Expose a small stable memory bridge rather than every memory operation's full schema upfront.
- [x] Let the model discover memory sections, then a brief operation catalog, then the selected operation schema.
- [x] Require schema discovery before calling the selected memory operation.
- [x] Retrieve selected memory on demand; do not inject the entire store automatically.
- [x] Bound memory retrieval to three expanded records and 16,000 returned characters per turn.
- [x] Reset memory discovery/budget state at the supported compression boundary.
- [x] Provide a second small staged bridge for restricted browser/file approval proposals; no full upfront operation schemas.
- [x] Add opt-in section → tool names/descriptions → selected schema discovery through Hermes' existing tool bridge, scoped to current session permissions.
- [x] Bound catalog/schema replies and preserve normal execution hooks and approval checks.
- [ ] Defer the remaining four agent-loop-native tools and custom non-registry provider schemas; these remain direct when enabled.
- [ ] Validate general tool selection and schema adherence with the trained E4B model.
- [ ] Safely remove no-longer-needed tool schemas/results from actual conversation context.
- [ ] Validate checkpointed context rebuilding and measure logical/physical cache behavior; discovery-state reset alone does not release KV. Trained intra-generation eviction is deferred.

## 7. Memory: layers, storage and access policy

- [x] Implement temporary L0/L1/L2/L3 and persistent L0/L1/L2/L3 for one owner.
- [x] Store curated memory as bounded, human-readable text JSON rather than binary KV or media.
- [x] Preserve record labels, evidence references, timestamps and revision information.
- [x] Implement atomic replacement, locking and restart-safe writes.
- [x] Make main-model memory writes append-only; expose no model correction/deletion tools.
- [x] Restrict appends to expiring session-bound stages referencing owner-approved evidence.
- [x] Keep owner inspection, editing, deletion and export separate from model tools.
- [x] Exclude ordinary wrong tool calls, malformed arguments, guesses, assistant reasoning and media from curated evidence.
- [x] Support successes, confirmed facts/decisions/preferences and selected critical-failure lessons through the approval gate.
- [x] Implement an opt-in separate memory service and remote provider with no local-store fallback.
- [x] Separate revocable model/controller credentials from owner approval/edit/delete administration.
- [ ] Deploy OS permissions/mounts that physically isolate memory files from the agent; a separate service alone does not enforce this.
- [ ] Validate that arbitrary enabled file/shell/code tools cannot bypass memory permissions before enabling them.
- [ ] Add scalable search/provenance operations; SQLite/FTS5 is a later target, not the current store.
- [ ] Add semantic contradiction flags and derived-evidence invalidation where required.

## 8. Memory: checkpoint capture and task recovery

- [x] Prepare review requests after five idle minutes and at Hermes' pre-compression hook.
- [x] Extract bounded tool-result assertions as unverified review candidates, not automatic proof.
- [x] Allow owner-supplied evidence and owner approval of selected candidate IDs.
- [x] Open append stages only through the owner/controller, never by the model itself.
- [x] Keep unfinished tasks/latest errors in separate bounded session recovery state, not curated success memory.
- [x] Provide owner task completion and stopped-Hermes crash recovery commands.
- [x] Avoid repeatedly preparing the same unchanged idle review request.
- [ ] Add task-specific automatic verification if human approval is to be reduced.
- [ ] Implement/validate strict checkpoint-required compression API v2 and the intended transcript-archiving contract.
- [ ] Calibrate any explicit usable-context threshold, such as the original approximately 75% target.
- [ ] Validate safe context reset/rebuild without losing required evidence, active tool calls or unfinished tasks.
- [ ] Validate real interrupted-task recovery end to end with the trained model.

## 9. Memory: idle consolidation, cleanup and maintenance

- [x] Promote temporary records atomically after three idle hours, while no tracked foreground turn is active.
- [x] Keep deterministic promotion available without calling a model.
- [x] Add optional same-local-model destination-layer classification; never rewrite existing memory text.
- [x] Reject stale/invalid sort plans and preserve temporary memory when sorting fails.
- [x] Deduplicate exact records; remove successfully promoted entries from the temporary set transactionally.
- [x] Expire persistent L0 after 30 days by default, pending reviews after one day and recovery state after seven days.
- [x] Avoid automatic L1–L3 expiry and silent deletion of unpromoted memory.
- [x] Provide owner sorting, consolidation and expiry commands.
- [ ] Validate optional sorting against the actual trained model rather than scripted replies.
- [ ] Improve large-backlog processing beyond the current 32-record/60-KB sorter request limit.
- [ ] Add semantic deduplication/contradiction handling without granting the main model unrestricted rewrite authority.
- [ ] Implement authorized second-laptop maintenance/compaction and approval of proposed changes.
- [ ] Implement and test the future remote-maintenance link; no such link exists now.

## 10. Multi-user memory and group authorization — later

- [ ] Authenticate stable platform/group/participant identity and authorized roles in every read/write request.
- [ ] Share L0/L1 only within the authorized group, not globally across groups.
- [ ] Keep L2/L3 private per participant.
- [ ] Keep all temporary memory private to the requesting participant.
- [ ] Require explicit approval for private-to-shared promotion/disclosure.
- [ ] Apply identical permissions to labels, snippets, existence checks, search and provenance.
- [ ] Isolate users' prompts, task state, logs, caches and active sessions.
- [ ] Prevent built-in session search, auto-learning, skills, cron and auxiliary routes from bypassing authorization.
- [ ] Test cross-user, cross-group, group/DM and platform-ID confusion attacks.
- [ ] Ensure administrators do not automatically gain access to private participant memory.

## 11. Harness-managed context and deferred learned Memento

Task/step recovery is separate from approved L0–L3 memory. Raw KV/hidden reasoning are not copied into JSON. See section 17 for the memory increment; all live context removal remains disabled until its gates pass.

- [ ] Accept original-model chat/tool behavior and direct extraction on unseen longer agent histories.
- [ ] Validate task contracts, source-linked records, supersession and pending dependencies without model self-verification.
- [ ] Run shadow records before enabling any context replacement.
- [ ] Prove strict checkpoint coverage, readback, interruption/restart recovery and reset refusal on save failures.
- [ ] Integrate the policy through the existing context-engine interface, keeping prompts/tool dispatch and permissions intact.
- [ ] Preserve active tool-call/result pairs, needed schemas/evidence, incomplete streams and unfinished answers.
- [ ] Test repeated context rebuilds and continuation separately from extraction; include cancellation and rollback.
- [ ] Measure input tokens, rebuild/prefill cost, occupied/reserved KV, actual allocations, RAM and task latency separately.
- [ ] Keep context records unable to bypass external memory approval/authorization.
- [ ] Optional later: justify targeted training with held-out agent regression gates before another run.
- [ ] Optional later: validate learned delimiters, masks, position/RoPE/cache updates and actual physical block eviction; no such capability is claimed.

## 12. Messaging, tools and media — later integration

- [ ] Validate authenticated Telegram/Discord group and private-chat routing with the single local model.
- [ ] Queue work with deadlines and participant/role-scoped cancellation and approval.
- [ ] Require confirmation for consequential actions and reject untrusted content as permission authority.
- [ ] Enable approved work tools with filesystem, secret, network and subprocess restrictions.
- [ ] Provide operator stop/pause/plugin-disable/model-rollback controls.
- [ ] Validate outgoing text, approved image/file and text-plus-image replies.
- [ ] Implement bounded RAM-only image storage, scoped access, immediate cleanup where possible and a 24-hour maximum age.
- [ ] Audit logs, checkpoints, crashes, caches and swap exposure for media persistence.
- [ ] Validate incoming image understanding through a compatible local reference backend.
- [ ] Implement Xenolith vision support only after independent correctness checks; current engine is text-only.
- [ ] Require authorization before sharing private images or descriptions into groups.

## 13. Security, verification and release readiness

Latest selected offline foundations/regression run: **325 passed, zero failed**
across 14 files; all 12 static checks pass. See `OFFLINE_FOUNDATIONS_STATUS.md`.
The prior v4 recovery-hardening run passed 289 cases across 12 files;
see `V4_MEMORY_SECURITY_FIXES.md`.
The prior parser-hardening run passed 234 cases. The earlier
198-pass/two-failure run exposed the parser gaps below; both original failing tests
now pass unchanged. See `OFFLINE_SECURITY_TEST_REPORT.md`.

- [x] Add/run 45 offline adversarial cases; both initial failures now pass after fixes.
- [x] Reject duplicate security-sensitive headers in both custom memory and access endpoints.
- [x] Reject duplicate JSON field names, including nested objects, before service dispatch.
- [x] Add 34 shared-parser cases covering both services, incomplete bodies, nonstandard numbers, depth limits and valid requests after refusal.

- [x] Pass 39 CPU-only memory tests, including provider discovery, owner CLI, gates, recovery, sorting and retention.
- [x] Pass 17 additional access-broker tests (56 total), including real plugin discovery, owner CLI, request-only HTTP handling, exact approvals and file-export restrictions.
- [x] Pass targeted Ruff lint, code-health checks and all 12 repository static checks.
- [x] Provide a restricted launcher that rejects unexpected tools/providers and non-loopback Python networking.
- [x] Supply an optional non-root/read-only-root Linux container recipe with documented limitations.
- [x] Document that neither the launcher nor an in-process plugin is a complete OS security boundary.
- [ ] Build/deploy and audit actual Linux isolation; the container recipe is not tested deployment.
- [x] Pass 155 selected CPU-only tests on Windows Python 3.14.7: 56 prior project, 16 service, 11 sections and 72 upstream discovery tests.
- [ ] Validate the complete production environment and Linux Python 3.14 runtime, beyond these selected Windows tests.
- [ ] Complete real-model, gateway, long-running, interruption and Linux end-to-end acceptance tests.
- [ ] Verify no cross-user leakage or unintended reasoning/KV/media persistence.
- [ ] Define acceptable RAM/latency budgets and measure them on the actual 16-GB target laptop.
- [ ] Benchmark reference versus Xenolith and ordinary versus checkpointed rebuilt context; separate logical occupancy from physically released cache. Learned eviction comparison is deferred.
- [ ] Provide final installation, backup/restore, migration, rollback and upgrade instructions.
- [ ] Publish the supported-capability matrix, benchmark report and remaining limitations.

## 14. Optional extensions — keep deferred

- [ ] Audited lightweight GUI-helper tools, only after reliable text/tool performance.
- [ ] Gemma-native draft/MTP acceleration, only after core E4B/Memento correctness.
- [ ] Live remote-desktop functionality, only with separate authorization/security design.
- [ ] Additional messaging platforms and other nonessential extensions.

## 15. Browser-only access and narrowly controlled files

Implemented foundations are opt-in and **not activated**. No automated browser is connected. A separate process is not OS containment unless its identity, mounts and network permissions are restricted.

### Capability and approval controls — built now

- [x] Add a default-disabled access bridge without modifying core Hermes or enabling existing system tools.
- [x] Accept only an explicit vocabulary of browser handoff requests and plain-text export requests.
- [x] Expose submit/status only to the model-facing service; no owner approval, inbox content listing or system-execution API.
- [x] Require a client credential outside model messages; bind transport to numeric loopback and refuse client proxies/redirects.
- [x] Keep owner decisions in separate local administration commands, never model tools.
- [x] Bind each decision to the exact reviewed payload digest with a one-hour expiry and no reusable blanket approval.
- [x] Reject unknown actions, unexpected payload fields, model-supplied file paths and oversized requests.
- [x] Provide manual owner handoffs for opening pages, downloading, uploading, forms, messaging, purchases and account changes.
- [x] Perform no automatic browser/download/network action for those handoffs, even when the owner approves a request.
- [x] Screen proposed URLs for HTTPS, credentials, private/local IP literals, local hostnames and ambiguous forms.
- [x] Create approved plain-text exports only in the broker's dedicated output folder using generated `.txt` filenames.
- [x] Refuse overwrites and expose no arbitrary file read, edit, delete, execution or installation operation.
- [x] Bound inbox/text sizes and retain decision status transactionally for restart/recovery.
- [x] Test these controls without launching a browser, live service, application, model or GPU.
- [x] Preserve the memory-only launcher's exact tool restriction; add optional service-memory profile configuration without enabling access tools.

### Linux isolation and automatic browser controls — later, NOT deployed

- [ ] Run agent, browser and privileged broker under appropriately isolated OS identities/containers.
- [ ] Restrict agent filesystem mounts so it cannot reach owner administration, broker/memory stores, browser profile or host secrets.
- [ ] Deploy read-only runtime, minimal writable scratch, resource limits and no host-control/Docker socket mounts.
- [ ] Enforce network egress outside the agent; allow only the reviewed inference/broker/browser transports.
- [ ] Provide a safe broker transport across isolated network namespaces without exposing the service publicly.
- [ ] Build and connect the limited browser adapter; no arbitrary JavaScript/CDP/evaluation or OS command surface for the model.
- [ ] Use a clean disposable browser profile with no personal credentials, saved sessions, extensions or normal user-profile sharing.
- [ ] Block browser local-file access, uploads and downloads by default through browser and OS policy.
- [ ] Intercept navigation, redirects, popups, downloads, service workers and subrequests without bypass channels.
- [ ] Enforce DNS/IP public-network policy including redirects, rebinding, local network and metadata-service access.
- [ ] Bind browser action approval to the exact page/action/content; invalidate it when the relevant page state changes.
- [ ] Require approval for submissions, uploads, messages, purchases, account changes and disclosure of private information.
- [ ] If automatic downloads are desired, add exact one-shot grants and isolated quarantine; never auto-open/install/execute a downloaded file.
- [ ] Permit the alternative manual download workflow without giving the model general download permission.
- [x] Implement a controlled memory service independently of the access broker; service mode opens no agent-local memory bank.
- [ ] Deploy and adversarially verify the memory service's separate OS identity, storage permissions and transport.
- [ ] Add a new reviewed launcher with an exact browser/file/memory tool allowlist and fail-closed activation.
- [ ] Validate file symlink/hardlink/race protection under the final Linux permissions and mount layout.
- [ ] Test malicious webpage injection, data exfiltration, local-network access, download bypasses and sandbox escape attempts.
- [ ] Validate real Hermes/browser/file workflows and document remaining risks before enabling the deployment.

## 16. Memory-service deployment and discovery acceptance

- [x] Keep local memory compatible by default; service mode is explicit and not activated in live profiles.
- [x] Enforce service-side discovery, retrieval budgets, append stages, checkpoint capture and idle maintenance.
- [x] Keep credential issuance/revocation and memory approval/edit/delete/export in owner-local administration, never the HTTP model surface.
- [x] Reject unknown backend settings and refuse local owner-store commands in service mode.
- [x] Document explicit owner setup, service limits and session/client scoping.
- [ ] Run a supervised live service and real Hermes integration in the separate validation setup.
- [ ] Test foreground latency during serial idle sorting and choose production scheduling/timeouts.
- [ ] Validate populated-bank migration, backup/restore and rollback; no actual memory was migrated.
- [ ] Add authenticated multi-user isolation; client credentials currently address one owner's shared bank.
- [ ] Validate section-first discovery with the original E4B model and real approved tools.

## 17. Source-linked step recovery — v4 memory increment

These items cover memory/recovery source only; no model, live service, profile activation or context reset is included.

- [x] Define bounded visible user/tool events, exact source-linked facts and controller-reported statuses; reject unsupported fields and verification claims.
- [x] Keep source text intact; refuse oversized records and full recovery banks without silently dropping evidence.
- [x] Save immutable steps atomically with expected revisions, content digests, idempotent retries and readback receipts.
- [x] Keep step recovery separate from curated records, approval stages and verified evidence.
- [x] Add controller-only service submission and owner-local strict-JSON file import; no model write operation.
- [x] Add staged session-scoped step list/read under the existing shared retrieval budgets.
- [x] Apply seven-day recovery retention and defer expiry while tracked foreground work is active.
- [x] Validate stored trust labels, structure, content and ordering before recovery list/read/retry; new records also check metadata integrity.
- [x] Permit requirement/decision facts only from complete visible user events, never tool output or a negation-dropping substring.
- [x] Return derived source roles and reference-only/no-permission markers; saving or reading recovery cannot recreate an approval.
- [x] Refuse recognizable credential/private-key patterns before step persistence with generic errors; preserve accepted source text exactly.
- [x] Expire each recovery step by its own save time rather than renewing old steps when new ones arrive; preserve foreground deferral.
- [ ] Complete typed-event capture exclusions for secrets/unrelated private output; pattern checks cannot guarantee detection of all sensitive text.
- [ ] Validate automatic consumers against negation, supersession, forged source roles and approval replay; context rebuilding stays disabled.
- [x] Complete offline recovery/security tests and existing extension regressions: 23 recovery cases plus 143 existing extension cases passed; targeted lint and all 12 static checks passed (health rerun with writable temporary storage).

Security hardening follow-up (21 October): see `V4_MEMORY_SECURITY_FIXES.md` for
the 289-pass selected run, 32 new recovery-security cases and compatibility notes.
Legacy records that fail the stricter
source/secret rules require owner review; no live bank was migrated or rewritten.
- [ ] Validate complete event capture and task-contract coverage; supplied quotes do not establish completeness or truth.
- [x] Wire opt-in bounded visible-text shadow capture through existing local/service memory-provider hooks; no live context replacement.
- [ ] Complete automatic abnormal-exit/rewind reconciliation and live shadow acceptance; identical offline retries alone are not complete reconciliation.
- [ ] Implement and prove strict checkpoint API v2; do not advertise it for this increment.
- [ ] Accept recovery with actual original-model/Hermes workflows in the separate validation environment.
- [ ] Enable bounded context replacement only after all persistence/extraction gates pass.

## 18. Offline foundations — no model/Linux required

See `OFFLINE_FOUNDATIONS_STATUS.md` and the extension's `OFFLINE_FOUNDATIONS.md`.

- [x] Add exact-source shadow capture, immutable event steps and readback-validated checkpoint manifests; default off.
- [x] Preserve complete user clauses and reported tool outcomes; do not infer semantic success, dependencies or supersession.
- [x] Add owner-declared task statuses/explicit later-user supersession with revision checks and scoped staged model reads.
- [x] Add an offline exact-transcript checkpoint gate with refusal on absent/stale source and pending tool calls; all receipts deny context replacement.
- [x] Add a lossless context-rebuild preview preserving accepted tool-call/result pairs, source text and assistant replies.
- [x] Add injected/scripted chat-stream bridge contracts for selected schemas, IDs/arguments, finish/errors and cooperative cancellation; no transport/server.
- [x] Add owner overview, task inspection/annotation, preview and curated-record-only backup/empty-bank restore commands.
- [x] Add profile-generation shadow opt-in, owner workflow/installation notes and SHA256 source inventory; no new dependencies.
- [x] Add 36 offline behavioral cases across two new test files.
- [ ] Complete actual native Xenolith/HTTP/SSE transport and Hermes connection acceptance.
- [ ] Connect/prove strict Hermes compression API v2 and normalized-evidence/transcript-archiving coverage.
- [ ] Implement validated semantic selection, shorter context and live replacement; lossless copying is not context compression.
- [ ] Validate actual E4B extraction/tool behavior, repeated continuation and physical cache/performance effects in the separate model environment.
- [ ] Complete GUI, full profile/service backup, populated-bank migration/rollback, full release/dependency/license inventory and Linux setup.

## 19. Controller-selected working context — 22 October prototype

- [x] Add an explicit opt-in owner CLI workflow; preserve the core loop and live profiles.
- [x] Anchor exact user instructions/corrections; generate plans; allow one host-selected action per step.
- [x] Rebuild requests from current state and exact chronological outcomes, excluding prior per-step reasoning.
- [x] Use strict JSON action decoding plus controller validation; no native tool-call compatibility claim.
- [x] Persist pending calls before dispatch and read back outcomes; refuse implicit replay after uncertainty.
- [x] Share identical failure bodies while retaining every attempt; preserve changed outcomes.
- [x] Add separate temporary handoff and configurable foreground idle timer; no curated-layer promotion.
- [x] Provide bounded local-only inference and owner-folder read-only tooling for isolated validation.
- [x] Pass 29 new behavior tests and 91 affected regression cases; targeted Ruff passes.
- [x] Run original Sol Fable using only the controlled system: two direct workflows and a CLI-consumer probe pass.
- [ ] Pass harder sequential agent acceptance: only two of six historical tasks completed, with answer omissions/errors.
- [ ] Diagnose output-cap/protocol behavior, repeated plan steps and native template compatibility.
- [ ] Integrate normal interactive CLI/gateway sessions, authorized general tools, concurrent steering and hard cancellation.
- [ ] Implement owner-reconciled restart/resumption, working-journal cleanup and unattended idle scheduling.
- [ ] Measure actual KV/cache costs and gains; do not infer physical savings from request rebuilding.

See [controlled workflow details](local_extensions/e4b_memory/CONTROLLED_STEPS.md). Previous offline-only scope statements describe their dated increments; local model tests for this new opt-in workflow occurred in the separate validation workspace with explicit user authorization.

## 20. Bonsai 27 backend and controller direction — active v5 increment

- [x] Make Bonsai 2 27B PTQ1 Lean the next candidate; preserve Gemma source/history and validation gates rather than relabeling its architecture.
- [x] Record exact Lean filename,6297658848byte size, SHA256 and parent/runtime provenance requirements.
- [x] Record separate Windows evidence: eight reasoning-off answers reviewed correct, final-request MTP sweep, and three actual low-reasoning file workflows. These are not Intel/Linux certification.
- [x] Add explicit Linux Prism process backend, dry-run default, exact model hash validation only on explicit start, loopback/one-slot routing, no occupied-endpoint takeover and cleanup of only owned process.
- [x] Add conservative CPU/Vulkan profiles with8192context,q4_0KV,6threads,256/128batch,4096output and loweffort. CPU MTPoff; Vulkan1token is an unvalidated candidate, not a measured optimum.
- [x] Refuse launch on Windows before accessing model/engine files; retain no-load restriction.
- [x] Add5GiB available-RAM guard, explicit device selection and no hosted/backend fallback.
- [x] Expose structured-action client defaults; document that existing controlled CLI defaults to thinkingfalse and cannot be changed by server defaults alone.
- [x] Pass11Python no-model contract tests; inspect CPU/Vulkan dry-run JSON without invoking a runtime, C build, model, WSL or GPU.
- [x] Update active v5 plan/checklist and engine docs; preserve prior memory/security implementation scope.
- [ ] Pin final HF revision and Linux CPU/Vulkan archive/executable hashes, dependency and license inventory; verify actual Intel device/driver names.
- [ ] Load/validate on the compatible Linux target with explicit authorization; verify actual RAM reserve/peak RSS/swap and PTQ1/Hadamard/Lean MTP behavior.
- [ ] Benchmark target CPU vsVulkan, threads/batches/cache/context and MTPoff/1/2/3/4 plus confidence thresholds; repeat winners and compare outputs, acceptance, total task time and thermals.
- [ ] Expose explicit low/off reasoning in the owning Hermes provider/controller transport, keep profile opt-in, and verify full CLI/gateway consumers. No core loop change is implied by the isolated adapter tests.
- [ ] Expand unseen adaptive workflows, longer histories, duplicate failures, real-idle handoff, restart/uncertain effects, concurrency/cancel and memory-stop acceptance.
- [ ] Validate prompt reuse/physical KV handling separately from logical context selection; keep native snapshots unsupported on the Prism backend.
- [ ] Implement a native Bonsai PTQ1/Qwen35 graph, Intel kernels or native MTP only as a separately reviewed future port. Current process backend does not complete this item.
- [ ] Activate production profile/service only after independent model, target OS/device, memory, security/isolation and rollback gates pass.

## 20a. Temperature-0 MTP candidate — 22 October 2026

- [x] Match greedy server/client defaults and verify no-model launcher contracts; retain CPU MTP-off and one-token Vulkan candidates.
- [x] Prepare an exact-commit/hash-checked external-runtime source patch for argmax drafts at p_min0 and conditional sampler cloning. Patch applies and reverses cleanly to pinned source fixtures; no runtime binary is changed.
- [x] Preserve existing batched verification and hybrid recurrent/KV rollback rules; verify the clone-elision arithmetic bound for every acceptance position in the checked range.
- [ ] Compile patched runtime and run upstream sampling/backend tests on the Linux CPU/Vulkan target. Patch is uncompiled and not a native Xeno port.
- [ ] Validate exact token IDs and recurrent/cache continuations for every rejection position, all-accepted, EOS/cancel, checkpoint/replay, suppressed/tied logits, grammar, reasoning limits and nonzero confidence thresholds.
- [ ] Measure original versus patched latency, acceptance, sampling/verification/rollback time and shared RAM/thermals before choosing settings or activating the candidate.

## 20b. Intel CPU optimization candidates — 22 October 2026

- [x] Add opt-in prefix-friendly state serialization and dedicated-slot exact-prefix requests; preserve complete sources/corrections/evidence and exclude prior reasoning.
- [x] Add optional loopback HTTP connection reuse with close-on-exit/error, no redirects/proxies and no automatic request replay; retain SSD action journaling.
- [x] Add separate CPU decode/prompt/draft thread controls, explicit optional affinity/polling, complete-verification microbatch validation, CPU-only candidate and24 dry-run tuning plans.
- [x] Prepare cumulative cpu-mtp-v1 source patch for draft bitmap reuse, all-accepted hidden-copy elision, early per-sequence draft caps and bounded timing-based adaptive windows; keep PTQ1 kernels unchanged.
- [x] Compile/run the actual standalone C++ policy tests and run offline launcher/harness/CLI tests plus patch application/reverse/hash checks. This does not certify the full runtime.
- [ ] Compile full CPU runtime, run upstream suites and verify all rejection/replay/cache continuations and token/action parity against original/MTP-off.
- [ ] Validate exact-prefix reuse versus recomputation after corrections, plan changes and removal of reasoning; verify cache hits and account for live/recurrent checkpoint memory separately from the prompt-cache budget.
- [ ] Measure CPU threads, affinity, polling, backend sampling, batch/microbatch, fixed/adaptive/off MTP and prefix/HTTP reuse separately and together; select by repeated whole-task time and sustained RAM/thermal measurements.
- [ ] Activate a CPU candidate only after target acceptance. Efficient PTQ1 decode/packing/Hadamard/matrix kernel work remains explicitly deferred.

## Changes from the original plan

- **v4 supersedes trained-model dependency:** original Sol Fable plus explicit task/step state is the first path. Training and intra-generation KV eviction are deferred research, not completed requirements. The core agent structure and existing memory/service permission gates remain intact.

- **Single user first:** group L0/L1 and per-user L2/L3 are still the target, not implemented guarantees.
- **JSON memory bank first:** the separate service still uses the text JSON memory store; SQLite/FTS5 memory search remains later. A small SQLite credential database is not that search implementation.
- **Append-only main model:** correction/deletion are owner operations, never main-model tools.
- **Selective evidence:** ordinary failures are excluded from curated memory; approved critical lessons may be retained. Task recovery is separate.
- **Human review gate:** automatic checkpoint preparation does not mean automatic truth verification.
- **Idle classification, not rewriting:** optional same-model sorting selects layers; it does not semantically compact existing records.
- **E4B execution source exists:** the old Phase 6 statement that tensor binding/PLE/forward execution are all unfinished is stale. Their implementation exists; compilation and numerical validation remain unfinished.
- **Retention exists now:** the old table's statement that prototype L0 expiry is unimplemented is stale. Single-user expiry is implemented; multi-user retention is not validated.
- **Progressive discovery now covers general registered tools:** optional section-first discovery extends the existing Hermes bridge. Four loop-native tools and custom non-registry provider tools remain direct when enabled. This does not grant permissions or remove conversation/KV history.
- **Memory service implemented, isolation not deployed:** an opt-in remote backend separates memory operations from the agent process. Linux filesystem denial, transport and live acceptance remain unfinished.
- **Browser/file restriction foundations added:** a default-disabled proposal broker and owner-approved text exports exist. Automated browsing, browser download interception and Linux OS containment are not deployed.
- **No deployment claim:** tests and static checks passed, but no Hermes application, model, engine or GPU was run here.

## Source status documents

- [Active v5 plan](Local_Bonsai27_Hermes_Xenolith_Memory_Plan_v5.md)
- [Historical v4 plan](Local_Gemma4_E4B_Hermes_Xenolith_Memento_Plan_v4.md)
- [Historical v3 plan](Local_Gemma4_E4B_Hermes_Xenolith_Memento_Plan_v3.md)
- [Step recovery status](STEP_RECOVERY_STATUS.md)
- [Single-user memory status](SINGLE_USER_MEMORY_STATUS.md)
- [E4B engine port status](E4B_PORT_STATUS.md)
- [Restricted access status](RESTRICTED_ACCESS_STATUS.md)
- [Memory service and discovery status](MEMORY_SERVICE_DISCOVERY_STATUS.md)
- [Offline security test report](OFFLINE_SECURITY_TEST_REPORT.md)
