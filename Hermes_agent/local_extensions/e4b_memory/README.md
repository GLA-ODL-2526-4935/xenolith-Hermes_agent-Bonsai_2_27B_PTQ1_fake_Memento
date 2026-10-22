# E4B single-user memory

The active project plan is v4. Controller-only source-linked step recovery is now
available separately from curated memory; see `../../STEP_RECOVERY_STATUS.md` for
the payload, limits, owner `save-step` command and unfinished integration gates.
Opt-in automatic visible-text shadow capture, source-linked task inspection,
checkpoint candidate gates and lossless previews are now available; see
`OFFLINE_FOUNDATIONS.md`. Semantic extraction, strict Hermes checkpoint API v2
integration and live context removal remain disabled/unvalidated.

Service-backed storage is now available as an explicit opt-in; see `SERVICE.md`.
The original local backend remains the compatibility default. Neither backend
is automatically activated, and Linux file-access isolation is not deployed.

Opt-in standalone Hermes provider. Hermes stays in its existing checkout; nothing
is installed/activated automatically. No model weights or new runtime dependencies.

## Available now

- Temporary/persistent L0–L3, text-only JSON, append-only model access.
- Sections → brief catalog → selected schema → call, with per-turn retrieval limits
  (three expanded records, 16,000 returned characters across memory calls).
- Automatic five-minute-idle and pre-compression review requests from bounded tool
  results. Tool success flags are candidates, NOT proof; an owner approves them.
  Routine failures stay only in separate recovery state. Hidden reasoning/media
  and assistant claims are not archived as evidence.
- Session-scoped unfinished-task recovery, owner completion and crash recovery.
- Three-hour-idle atomic promotion, deduplication, and optional same-model semantic
  layer sorting. Sorting cannot rewrite text; stale/malformed results are rejected.
- Retention: persistent L0 expires 30 days after promotion; pending reviews expire
  after one day; recovery state after seven days. Unpromoted memory is never
  silently discarded. L1–L3 do not auto-expire. Expiry waits while turns are active.
- Owner inspection/edit/delete/export commands; the model has none of those writes.
- Restricted profile generator and launcher, plus an untested Linux container recipe.

## First test later, in the separate validation environment

From the Hermes checkout, with its normal runtime prepared:

```
python -m local_extensions.e4b_memory.restricted init NEW_PROFILE --base-url http://127.0.0.1:8000/v1 --model YOUR_MODEL_ID
python -m local_extensions.e4b_memory.restricted run NEW_PROFILE
```

Use a NEW directory outside the plugin source. Creation refuses existing paths,
copies only this plugin, disables built-in memory, pins auxiliary routes locally,
and exposes only the memory toolset. The launcher checks the final tool surface
before every turn, allows one launcher per profile, and rejects unexpected tools,
altered profiles and non-loopback Python socket connections. It is a memory-only
test harness, NOT an OS security boundary or a general work-agent sandbox.

Add `--semantic-sort` at profile creation to enable optional idle classification
using the SAME configured local main model. Otherwise promotion preserves the
already selected layer. Model/network errors retain temporary records. Sorting
is bounded to 32 records/60 KB per request and has a 30-second network timeout;
larger backlogs require owner maintenance. No second model is loaded by this code.

For a normal opt-in directory installation, copy this folder to the selected
profile's `plugins/e4b_memory/`, select `memory.provider: e4b_memory`, explicitly
set `memory.memory_enabled: false` and `memory.user_profile_enabled: false`, and
set `platform_toolsets.cli: [memory]`. Primary single-owner CLI sessions only.
Never enable arbitrary shell/file/code/MCP/delegation tools with this in-process
store. See `deploy/README.md` for the Linux isolation recipe and remaining limits.

## Owner workflow

Use the owner's terminal with this profile selected (not a model-exposed shell):

- `hermes e4b_memory requests` / `tasks` / `status` / `export`.
- `hermes e4b_memory approve SESSION EVIDENCE_ID ...` after reviewing candidates.
  This opens a five-minute, eight-append checkpoint. Ask the model to inspect its
  checkpoint and save useful verified lessons; it cannot approve its own evidence.
- Alternatively `checkpoint SESSION evidence.json` with an owner-verified list:
  `[{"id":"proof","kind":"success","text":"Owner verified restore checksum."}]`.
- `edit ID replacement.txt`, `delete ID` (permanent except owner backups),
  `task-complete SESSION`, `consolidate`, `expire`.
- `sort plan.json` accepts an owner-prepared idle plan with `revision` from export
  and `assignments: [{"id":"...","layer":"L2"}]`; every temporary record is required.
- After a crash, stop ALL Hermes processes for this profile, then
  `recover --hermes-stopped` to clear abandoned activity markers and revoke stages.

The authoritative file is `<profile>/e4b_memory/memory.json`. It is human-readable;
use owner commands for live edits, or edit content with Hermes stopped. Do not
manually alter control fields. Keep backups outside model-accessible paths.

## Verification and remaining work

39 CPU-only contracts passed through Hermes' required test runner on Windows
Python 3.12, including real provider discovery, owner CLI, failure/restart paths,
review gates, sorting validation, retrieval budgets and restricted-launch checks.
These are not real-model, Linux, Docker or full Hermes CLI acceptance tests.

Still later: trained E4B/tool-template and Xenolith integration; learned Memento;
actual Linux isolation deployment; task-specific automatic truth verification;
second-laptop compaction; multi-user authorization. Strict checkpoint-required
compression API v2 is not claimed: this provider uses the legacy pre-compression
hook to receive tool evidence. Opt-in shadow mode archives only its explicitly
accepted bounded visible-text transcript, not every possible Hermes message form.
