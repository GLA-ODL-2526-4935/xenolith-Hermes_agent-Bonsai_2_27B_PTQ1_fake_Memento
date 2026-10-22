# Offline recovery foundations

Opt-in, single-owner, visible-text **shadow** capture. Nothing here launches a
model, starts a server, changes the live context, or enables another toolset.

## Capture and checkpoints

Set `memory.e4b_memory.shadow_capture: true` in a future test profile, or add
`--shadow-capture` when generating a NEW restricted profile. Restart to load the
setting. Default remains off; no existing profile is changed by this work.
Local `sync_turn` / `on_pre_compress` and service-backed controller hooks attempt
capture. The async hook reports generic refusal without interrupting normal turn
completion. `shadow_status` is an in-process diagnostic, not a success claim.

Accepted transcripts have 1–200 messages and at most 128,000 UTF-8 JSON bytes.
User/tool/system/developer text is at most 4,000 characters per message; assistant
visible text and serialized tool arguments at most 16,000. Only supported plain
text messages and function-call envelopes are accepted. Unknown fields (including
reasoning), media, obvious credentials, orphan/duplicate tool results and malformed
ordering are refused. Accepted text is exact, never silently shortened or redacted.
These are conservative format/pattern checks, not complete sensitive-data detection.
Service requests retain the existing **32,000-byte total wire limit**, so their
effective transcript budget is smaller. Reaching any bound does not remove context.

Each user/tool event becomes one immutable, source-linked recovery step. The
existing **32 steps per session** limit applies, including manually submitted steps.
User messages are whole source clauses, not semantically extracted requirements;
tool success/failure is reported, never independently verified. Assistant prose is
retained in the visible transcript baseline, never treated as success evidence.

Checkpoint manifests are committed only after all step receipts succeed, then
validated through readback. Identical retries are idempotent; interruptions can
leave orphan steps but cannot create a successful manifest. Retry the same source
to reconcile without rerunning a tool. A changed/rewound prefix is refused; do not
silently reset it. Existing owner crash recovery clears activity/stages only, and
does not invent a complete checkpoint. Fresh session IDs are needed for deliberate
new/branched work. Seven-day source-step retention also invalidates dependent
manifests; foreground expiry deferral is unchanged.

`checkpoint.require_checkpoint()` checks a current exact transcript and refuses
missing/stale checkpoints or pending tool calls. It is an **offline candidate
gate**, not Hermes' strict compression API v2 integration. The provider still
advertises API v1. Every receipt says `context_replacement_allowed: false`.
`preview_rebuild()` returns a lossless baseline, including tool envelopes and
assistant replies; no shortening, tokenizer counts, KV release or speedup is claimed.

## Owner commands

From the selected owner profile, use `hermes e4b_memory` followed by:

| Command | Purpose |
|---|---|
| `overview` | Inspect curated labels, pending reviews, tasks and shadow sessions. |
| `capture-shadow SESSION transcript.json` | Explicit offline transcript import. |
| `task-state SESSION` | Inspect full source clauses/reports and owner statuses. |
| `annotate-task SESSION change.json` | Append an owner-declared status/supersession. |
| `preview-context SESSION` | Inspect the lossless rebuild baseline. |
| `backup-records` | Print a portable curated-records-only JSON archive. |
| `restore-records archive.json` | Restore into a fresh empty bank, never overwrite. |
| `source-manifest` | Print source-file SHA256 inventory without installation. |

Service owner equivalents are
`python -m local_extensions.e4b_memory.service SERVICE_ROOT owner COMMAND ...`.
Use scoped service session IDs shown by owner inspection. None is a model write
operation. Existing owner approve/edit/delete/export commands remain available.
Save printed exports through the owner's normal interface, outside agent access.
Backups contain potentially private memory and require owner protection.

Annotation JSON example (replace IDs from `task-state`):

```json
{"id":"SOURCE_ID","status":"cancelled","superseded_by":"LATER_USER_SOURCE_ID","note":"Owner replaced this requirement.","expected_revision":0}
```

Statuses: pending, in_progress, completed, blocked, cancelled. Supersession must
point to a later user clause, never a tool result. Original source stays intact;
no semantic completion/supersession is inferred and no annotation grants permission.
The model discovers `recovery` → `task_items` (brief list) → `task_item` (one source)
through existing staged discovery, sharing the three-record/16,000-character budget.

Backups intentionally exclude approvals, active stages, credentials and recovery
transcripts. They are **not a complete service/profile backup**. Restore only into
an empty bank with relevant applications/services stopped. Full populated-bank
migration, credential restoration and release rollback remain later work.

## Bridge and packaging

`bridge_contract.py` prepares selected-schema chat requests and consumes scripted
stream chunks via an injected transport. It assembles tool IDs/arguments, refuses
unknown tools, duplicate JSON keys, truncated/incomplete/error replies, and checks
cancellation between chunks. It never dispatches a tool or opens a network socket.
Real HTTP/SSE/native Xenolith transport, schema-to-argument validation by the normal
dispatcher, hard cancellation/timeouts and model formatting remain integration work.
Unsupported reasoning/media/provider extensions are refused, not silently accepted.

No new dependencies. Use Hermes' `pyproject.toml` and `uv.lock` as dependency
authority, its package manager for environments, and `scripts/run_tests.sh` for
tests. Copy the entire extension directory (not only a new module) when installing
later, since existing modules reference siblings. `source-manifest` inventories the
source selected by extension. Full reproducible release/dependency/license inventory,
Linux installation and model pins remain unchecked in the main checklist.
