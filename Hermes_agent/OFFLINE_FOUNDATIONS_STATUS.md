# Offline foundations — 21 October 2026

Implemented in the existing Hermes checkout; no engine/model, live application,
server, browser or GPU was run. Existing profiles and actual owner memory were not
activated, migrated or edited. No new dependencies or model permissions.

| Area | Implemented now | Still later |
|---|---|---|
| Automatic shadow capture | Opt-in local/service lifecycle wiring; exact bounded visible user/tool sources and transcript baseline | Arbitrary message formats, automatic abnormal-exit reconciliation, real Hermes/model acceptance |
| Task state | Ordered whole user clauses/tool reports, pending defaults; owner statuses and explicit later-user supersession; staged model reads | Semantic requirement extraction, dependency inference, automatic proof of completion |
| Checkpoints | Versioned manifest, source receipts/readback, exact-transcript gate, idempotent retry and partial-write reconciliation | Hermes strict checkpoint API v2/transcript-archiving integration and completeness proof |
| Rebuilds | Lossless preview preserving all accepted messages and tool envelopes; pending-call reporting | Shorter working context, live replacement, repeated model continuation and KV measurements |
| Connection scaffold | Injected/scripted stream adapter with selected tools, IDs/arguments, completion/errors and cooperative cancellation | Actual HTTP/SSE/native Xenolith transport and model/engine integration |
| Owner management | Overview, source-linked task inspection/annotation, previews and curated-record backup/empty-bank restore | GUI, full service/profile backup/migration/rollback |
| Packaging | Configuration opt-in, installation/limits guide, source SHA256 manifest command, existing dependency locks retained | Complete release pins, license/dependency inventory and Linux deployment |

See `local_extensions/e4b_memory/OFFLINE_FOUNDATIONS.md` for commands and limits.
The shadow transcript is a lossless baseline, **not successful bounded selection**.
There is no token-saving/performance claim. All checkpoint receipts explicitly
deny live context replacement and the provider still advertises checkpoint API v1.

## Verification

36 new CPU-only behavioral cases cover capture/restart/retry/refusal, partial writes,
pending tool envelopes, source expiry, owner supersession, local/service provider
hooks, scoped discovery/budgets, configuration generation, scripted streams,
cancellation, curated-record backup/restore and source manifests.

The first wider regression run exposed a legacy pre-compression test path with an
uninitialized optional setting. Fixed by keeping the new shadow toggle default-off
before initialization; the original test was not changed. Final results: **325 passed,
zero failed across 14 files** (289 prior cases plus 36 new cases), using the required
runner with the isolated Windows Python 3.14.7 test environment. All **12 static
checks passed**, including code health (42 files, zero blocking/advisory findings).
Targeted Ruff passed and `git diff --check` was clean.

Limits remain deliberate: 32 total recovery steps/session, 200 messages/128 KB local
transcript, smaller 32 KB total service request, exact prefix requirement, generic
capture refusal, no authority from tool reports, no semantic contract completeness.
Legacy review/log paths are not sanitized by refusing a new shadow archive. Owner
review and separate future Linux/model acceptance remain required.
