# Controlled working context — opt-in prototype, 22 October 2026

This implements the user's one-tool-action-per-step workflow as a standalone consumer in the existing Hermes memory extension. It does not activate a live profile, replace the normal AIAgent loop or use L0-L3/recovery memory. The user explicitly authorized local model validation; it ran from the separate chat testing workspace.

## Request and action lifecycle

1. Anchor the complete original user request and every later correction, in order.
2. Ask the same local model for a short human-readable plan and explicit step goals. Model plans grant no authority.
3. Send a fresh request containing the stable controller prompt, anchored user text, current plan/step and chronological action/outcome history. Previous assistant messages and per-step reasoning are excluded.
4. Accept exactly one strict JSON action: call the current step's selected tool, revise the remaining plan, or finish. Decoder constraints enforce shape/current step; controller validation remains authoritative.
5. Atomically save and read back a pending call before dispatch. Execute only a host-selected callback. Save exact arguments and its reported outcome, then advance. The next model request is reconstructed from that state.
6. Keep every attempt. Exactly identical failed calls and outcomes share one result body, with explicit references for repeats. Changed outcomes and successful repeats remain distinct.
7. An uncertain outcome or exception after dispatch stops for owner reconciliation. No implicit retry/restart/replay is provided. Capacity, invalid response, output cap and round-limit failures retain the journal.

The controller does not save full hidden reasoning. This is controlled text context, not selective KV-block eviction. Rebuilding may incur prefill work and does not prove lower allocated VRAM or shorter total task time.

## Runnable Hermes edge

The active extension's existing owner CLI now registers:

```text
hermes e4b_memory controlled-run TASK.txt --session unique-run --workspace APPROVED_FOLDER --base-url http://127.0.0.1:18780 --model MODEL_ID --idle-handoff-after 300
```

Use a fresh isolated owner home/profile for testing. Existing memory-provider CLI discovery rules still apply; this work does not enable the provider in any existing profile. Service-backed memory configuration refuses this local owner command rather than falling back.

The supplied tool is `read_text(path)`: complete bounded UTF-8 reads below the explicitly selected workspace, with separate missing-file/permission outcomes. No shell, write, upload, browser, curated-memory or external messaging tools are granted. This is a conservative path boundary, not an OS sandbox against hostile filesystem races or a general Hermes tool-grant system. Broader tools require their ordinary authorization and dispatch integration before being connected.

LocalJSONModel uses one explicit loopback-IP HTTP route, disables proxies/redirects, has a request timeout, and never selects a hosted fallback. Strict JSON action envelopes deliberately bypass the unvalidated native tool-call template; they are a controller protocol, not proof that Gemma emits native Hermes function calls.

Files live under `OWNER_HOME/controlled_runs/SESSION/working-context.json`, independent of the memory bank. This disk journal mirrors logical working context for crash inspection; raw KV is never saved. Hash/readback checks detect accidental corruption, not a malicious writer able to recompute hashes. A fresh run refuses an existing session directory's journal. Owner reconciliation/resumption is not automated.

## Idle handoff

`--idle-handoff-after N` keeps this explicit CLI invocation in the foreground after termination, then saves a separate revision-named temporary handoff. Omitting it exits without a timer. The process must remain alive; no detached scheduler or four-layer promotion is installed. Allowed delay: 0-86,400 seconds; 300 is the usage example, not a new global retention policy.

The handoff includes anchored user requirements, short plan/current step goals, exact calls and all outcomes, final answer and unfinished work. It retains ordinary unresolved failures as well as permission failures. It is reference-only, unverified and grants no permission. Active/pending/uncertain actions refuse idle handoff. Source journals remain available if handoff fails. Separate retention/owner cleanup is still needed; no existing bank's seven-day or L0 policy is silently applied.

## Limits and remaining integration

- 16 selected tool schemas, 16 plan steps, 24 model rounds by default.
- Working request: 24,000 characters; user text 16,000 characters; tool arguments 8,000; output 16,000; journal 2 MB. Limits stop rather than truncate. These are character/byte guards, not tokenizer estimates.
- Model output budget 1,024 tokens; reference test server context 8,192. Liveness/output failures never dispatch partial JSON.
- User corrections are accepted between action/model boundaries. Concurrent interactive steering, hard in-flight cancellation and normal CLI/gateway session integration are not implemented.
- Replanning can still be poor, repeat reads or attempt duplicate step IDs. The controller refuses invalid IDs; it does not establish semantic plan completeness or task success.
- Strict compression API v2 and the original provider's context-replacement prohibition are unchanged. This is a separate explicit workflow, not an enabled replacement policy in ordinary sessions.

## Tests and actual model result

29 new controller behavior cases pass, plus 91 affected existing memory tests (120 distinct cases total). Covered: user corrections, deterministic outcome retention, duplicate failures, changed outcomes, crash/no replay, uncertain calls, bounded context, cancellation, local endpoint selection, integrity/disk failure, CLI registration, decoder schema and idle handoff. Targeted Ruff passes.

Original Sol Fable Q4_K_M, no adapter/MTP, reasoning requested off, official llama.cpp b11780:

- Two current-task workflow tests completed correctly: a 10-to-8-ms mid-run user correction; and actual missing-file failure plus out-of-workspace permission denial with pending SSD recovery.
- The registered CLI parser and actual run_controlled handler also completed a separate two-file inspection correctly, including a temporary handoff with zero test delay. This does not test full profile bootstrap or the default UI loop.
- Six harder historical agent tasks: only two completed. Three stopped at the output cap and one on a reused plan-step ID. The two final answers still contained omissions/an incorrect prohibition, so completion is not correctness. This is not production acceptance.
- Earlier unconstrained and schema-only iterations are preserved separately. No baseline, curated bank, summary model or memory-retrieval system supplied hidden help to the final controlled-loop runs.

Detailed artifacts and reviewed results are in the separate chat output folders `sol-fable-controlled-loop-final-2026-10-22`, `sol-fable-controlled-loop-edges-2026-10-22` and `sol-fable-controlled-cli-2026-10-22`. Further protocol/runtime diagnosis and longer task acceptance are required before default activation.

## Intel CPU optimization increment — 22 October 2026

This increment supersedes the earlier blanket no-prefix-reuse configuration **only for the opt-in CPU candidate**. PTQ1 decoder/packing/Hadamard/matrix kernels remain deferred at the user's request. The original CPU MTP-off/prefix-off profile and live Hermes settings remain available and inactive here.

Implemented source: exact-prefix-friendly controlled-state serialization with full source/evidence/correction preservation, local HTTP keep-alive without request replay, separate prompt/decode/draft CPU threads, optional affinity masks and polling, complete-verification-microbatch guards, a CPU-only candidate profile and a dry-run24-plan tuning renderer. The cumulative cpu-mtp-v1 external-runtime patch adds buffer reuse, all-accepted hidden-copy elision, early enforcement of per-sequence draft limits and optional per-request timing-based MTP window selection (1..min(4,cap), four samples/window, a probe every32 valid cycles). It includes greedy-mtp; apply to the clean pinned source, never stack both patches.

Candidate: Linux CPU, device none and target/draft GPU layers0, one slot,8192 context, batch256/ubatch128, six prompt/decode threads, polling0, p_min0, adaptive cap3. Prefix cache is requested with128MiB prompt-cache budget, two additional context checkpoints and idle-slot archiving off. Live KV/checkpoints are outside that budget; >=5GiB MemAvailable safeguard remains. CPU backend sampling is independently selectable, not GPU placement. No CUDA switch is emitted. No affinity topology or fastest thread/batch choice is presumed.

Hermes owner CLI options are `--reuse-prefix --keep-alive`; both are opt-in. They do not activate ordinary Hermes or store/reinject prior reasoning. Prefix eligibility is exact rendered-token identity; hybrid state restoration/recomputation, changed user instructions and removed-reasoning tails need target parity acceptance. No persistent KV cache or automatic inference/tool retry is added. SSD pending-action writes/readback remain mandatory. Existing controlled-run reasoning-off/output1024 defaults are retained; the launcher's low/4096 request defaults are a separate production-provider configuration decision.

The actual C++ policy compiled and passed standalone tests. Python launcher/harness contracts, patch application/reverse/hash checks and dry-run tuning rendering were checked without a model. The full runtime remains uncompiled and unbenchmarked. Adaptive warmup can cost time on short answers and does not compare MTP-off internally. Compare whole-task latency, token/action parity, cache hits, recurrent replay, RAM/swap and sustained thermals before adoption. Do not remove synchronization or recurrent checkpoints on an assumption that CPU execution makes them unnecessary.

Implementation and target procedure: `engine/patches/prism-cpu-mtp/README.md` in Xeno; cumulative patch pinned to sudoingX ff414120c343e6e6cb868013c99f1dde52b27e70. Target compilation/validation and runtime activation remain pending. No model, engine service, GPU or WSL was loaded here; only the standalone policy test binary ran.

