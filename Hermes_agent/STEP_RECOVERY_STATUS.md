# Step recovery — v4 memory increment, 21 October 2026

Active architecture: [plan v4](Local_Gemma4_E4B_Hermes_Xenolith_Memento_Plan_v4.md).
Changes are restricted to the standalone memory extension, documentation and offline
tests. No core loop, engine, tool grants, live profile, actual memory or credentials
were changed or activated.

## Implemented source

- `step_recovery.py`: controller-submitted visible user/tool events and exact quoted
  facts; status is controller-reported, `verified: false`. Source text is intact.
  Only supplied evidence is retained: no automatic extraction or completeness claim.
- Immutable IDs, expected session revisions, atomic Store transactions with
  fsync/replace, idempotent retries, digest/readback receipts and integrity checking
  on individual retrieval. Digests are not authentication or semantic verification.
- Limits: 32 events and 32 facts per step; 4,000 characters per source event/quote;
  12,000 serialized UTF-8 bytes per record before metadata; 32 steps per session bank.
  Oversize/capacity failure preserves prior state, without truncation or rolling loss.
- Staged `steps` and `step` reads in `recovery`; individual step reads share the
  three-expanded-record and 16,000-character per-turn memory budget.
- Service `save_step` requires a controller credential and active begun instance.
  The service supplies the client/session prefix. Model credentials cannot submit.
  Abandoned activity requires existing owner reconciliation after restart.
- Local owner/controller `save-step SESSION FILE` imports bounded strict JSON.
  It is not a model tool and refuses service-mode local fallback.
- Recovery bank expiry uses the configured seven-day recovery retention, measured
  from each step's own save time after the security follow-up. A new step no longer
  renews older steps. Foreground activity defers expiry. Saving recovery creates
  no curated records, approved evidence or append stages.

## Security follow-up — 21 October 2026

- Stored trust labels, structure, revision ordering and content are checked before
  list/read/retry; fixed `verified: false` and controller-reported semantics cannot
  be replaced by edited "approved" values. New records add a full metadata
  integrity hash alongside the stable content digest used for idempotency.
- Older records without the metadata hash remain readable only if validation passes;
  their old digest does not cover timestamps/revisions. No hash is authentication:
  an attacker who can rewrite the bank can also recalculate hashes. OS denial remains required.
- Requirement/decision quotes require the whole visible user event. The earlier
  substring convention remains for observations, not authoritative user clauses.
  Returned `source_roles` and `use_policy` are derived and not caller-writable.
  All recovery remains reference-only and grants no permission.
- Recognizable password/API-key/token/Bearer/private-key patterns are refused before
  save; errors do not echo content and accepted source text is never silently redacted.
  This reduces accidental leakage, not every possible secret or private disclosure.
- Legacy content rejected by the new capture policy can still expire normally if its
  structural/integrity checks pass. Reading/retrying unsafe legacy content requires
  owner reconciliation; no actual bank was migrated, scrubbed or activated here.
- The two existing tests with substring requirement fixtures were updated to use full
  user events, reflecting the deliberately stricter contract. Original persistence,
  disk-failure, role, scope and retrieval-budget assertions remain.
- See `V4_MEMORY_SECURITY_FIXES.md` for final offline results and remaining gates.

## Payload example

```json
{
  "step_id": "save-check-1",
  "expected_revision": 0,
  "status": "blocked",
  "events": [
    {"id": "u1", "role": "user", "content": "Do not restart training.", "tool_call_id": "", "outcome": "unknown"},
    {"id": "t1", "role": "tool", "content": "Checkpoint 128 save failed: read-only SSD.", "tool_call_id": "save-128", "outcome": "reported_failure"}
  ],
  "facts": [
    {"category": "requirement", "event_id": "u1", "quote": "Do not restart training."},
    {"category": "observation", "event_id": "t1", "quote": "Checkpoint 128 save failed"}
  ]
}
```

Statuses: proposed/in_progress/blocked/completed/failed/cancelled. Even completed is
a controller report, not verified success. Fact categories: requirement/decision/
observation/uncertainty/next_action/rejected. Events: user/tool; outcomes:
unknown/reported_success/reported_failure. User text must use unknown. Tool results
require a tool-call ID; matching it to a real authorized call is the controller's job.

Amend history using a new step ID at the current revision. Conflicting IDs and stale
revisions fail; an identical retry returns its original receipt. A readback failure
can follow a successful commit: reconcile/retry the record, never replay the tool
action merely because its acknowledgement failed. Callers must retain working
context on any persistence failure. Longer tasks need reviewed rollover before
production; the capacity limit currently requires explicit owner maintenance.

## Verification

Latest hardening run: **289 passed, zero failed** across 12 selected files,
including the 23 recovery cases, 32 new recovery-security cases and the prior
234 memory/access/discovery cases. All 12 static checks passed, code health zero
blocking/advisory findings, and targeted Ruff passed. The results below are the
historical initial recovery increment, not the current full selected suite.

Canonical offline checks: **166 distinct tests passed, zero failed**: 143 existing
extension cases plus 23 new recovery cases. The first full seven-file run passed
163 cases; after adding exact-quote, readback/integrity and aggregate-budget checks,
the final recovery-file rerun passed all 23. Used `scripts/run_tests.sh` with the
existing isolated Windows Python 3.14.7 test interpreter, no bare pytest.

Targeted Ruff passed. All **12 repository static checks passed** across the original
11 passing guards and a successful code-health rerun. The initial health attempt
hit Windows sandbox temporary-folder permissions, then passed using a writable
task-local temporary directory: 33 files, zero blocking and zero advisory findings.
This is scoped offline validation, not a live deployment certification.

Tests use disposable banks and in-process dispatch, no live socket, Hermes app,
model, engine or GPU. Historical reports retain their original dated scope.

## Offline foundation follow-up — 21 October

Opt-in local/service shadow capture, source-linked task views, owner annotations,
readback-validated manifests, offline checkpoint gates and lossless rebuild previews
now exist. See `OFFLINE_FOUNDATIONS_STATUS.md` and the extension's
`OFFLINE_FOUNDATIONS.md`. They do not establish semantic extraction, source
completeness, full abnormal-exit reconciliation or safe context shortening. The
provider remains API v1 and no live context is replaced.

The final offline foundations/regression run passed all 325 cases across 14 files,
including 36 new cases; all 12 static checks passed. This is not live acceptance.

## Not implemented or certified

- Arbitrary typed-event capture, semantic task-contract completeness, automatic
  supersession/dependency inference, extraction fidelity and actual model/Hermes acceptance.
- Automatic compaction lifecycle integration. Provider checkpoint API v1 is unchanged;
  strict checkpoint API v2 is not advertised and no context is removed.
- Reasoning segmentation, context rebuild/KV eviction, performance gains or trained
  Memento behavior; Linux deployment, migration/rollback and multi-user privacy.

Unknown fields, assistant-reasoning roles and obvious inline-media envelopes are
refused, but schema checks cannot detect every encoding or mislabeled reasoning
inside arbitrary text. Controllers must supply authorized visible evidence. Quoted
text remains untrusted data and grants no permission. Owner review is still required
for curated memory. No new tool permissions or OS containment are claimed.
