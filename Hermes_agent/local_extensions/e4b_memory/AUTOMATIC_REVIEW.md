# Automatic drafts and same-layer conflict review

Implemented source, **opt-in and not activated**. Temporary memory remains L0–L3.
No checkpoint API v2, model inference, Linux deployment or live profile change
is part of this increment.

## Workflow

1. Existing hooks collect bounded, **unverified** tool-report candidates and a
   bounded user-goal excerpt. At five idle minutes the same configured local
   model can draft up to four temporary records from one session's candidates.
   No tools, filesystem or approval capability is supplied to that model pass.
2. Drafts enter an owner-only proposal queue, not the readable memory bank.
   Inspect the exact proposed text AND sources. Explicit owner acceptance adds
   temporary L0–L3 records; no second evidence-approval round is necessary for
   these drafts. This does not open a model append window.
3. A later idle pass compares one temporary record with **all current persistent
   records in its own layer**. The model proposes keep, rewrite or discard with
   source IDs and an explanation. It cannot change the bank itself. Existing
   persistent memory is not necessarily correct: dated/scoped changes and new
   user corrections must be surfaced, not silently rejected.
4. Inspect and accept/reject that conflict proposal. Acceptance of a rewrite or
   discard is an owner-authorized edit of temporary memory only. Its original
   content remains in the proposal audit. Pending/rejected proposals do not edit,
   discard, approve, promote or become model-readable curated memories.
5. After three idle hours, only records with an accepted, still-current conflict
   check can promote. Any change to their content/evidence or compared persistent
   layer invalidates approval. Promotion of one record changes that layer, so
   other records of that layer need a fresh comparison. Persistent text is never
   rewritten by this workflow. Existing L0 expiry and owner administration remain.

This replaces the old idle sorter **only when enabled**, retaining each temporary
record's current layer. Legacy evidence approval/stage-gated append remains.
Enabling marks existing temporary records for conflict review. Disabling stops
automatic generation but does not remove review requirements from marked records;
neither ordinary consolidation nor owner sort can bypass those requirements.

## Configuration and owner review

Local profile: set the following in its existing e4b_memory settings and restart.
The main model must already be explicitly configured as a custom loopback model.
No endpoint, model or profile is selected/activated by this code change.

```yaml
memory:
  e4b_memory:
    automatic_review: true
    review_timeout_seconds: 120
```

For a **new** restricted local test profile, the generator accepts
`--automatic-review`. Old profiles/configurations remain compatible. Service
users enable the same optional settings via the existing owner-side `configure`
command with the service stopped, keeping its required settings and explicit
local model/base_url. Do not set model-maintenance options on the remote agent.

Owner CLI (not model tools):

- `hermes e4b_memory review-memory`: inspect proposals, before/after text, source
  records, conflict IDs, reasons and generic failure notices.
- `hermes e4b_memory accept-memory PROPOSAL_ID`: accept exactly the inspected
  proposal. It refuses stale sources, active foreground work and repeat decisions.
- `hermes e4b_memory reject-memory PROPOSAL_ID`: reject without touching records.
- `hermes e4b_memory retry-memory-review`: explicitly retry failed model/budget
  checks; successful/pending/rejected proposals are not automatically regenerated.
  Changed sources create a fresh job. Reject a stale pending proposal during review.

Service owner equivalents use the existing service-host
`python -m local_extensions.e4b_memory.service ROOT owner COMMAND` interface.
Neither model nor controller bearer credentials can approve proposals.

## Bounded work and limitations

- One job per maintenance tick (normally 30 seconds), outside the store lock.
  The service uses a background worker so its HTTP request loop is not occupied
  by review inference. Foreground activity/revision changes discard in-flight
  results. This is cooperative yielding, **not hard cancellation** of inference
  already running on the model server. One-slot runtime scheduling and latency
  still require target tests. No second model or inference retry is introduced.
- The configurable socket timeout is 1–300 seconds, default 120 for slow local
  inference. It is not a whole-job/server cancellation deadline. Output cap 1,536
  tokens; input scope 18,000 serialized bytes and HTTP request 24,000 UTF-8 bytes;
  response 16,000 bytes. An incomplete/oversized reply retains originals.
- Full same-layer comparison is limited to 32 persistent records and the input
  budget. Overflow blocks that check; it never labels a partial search complete.
  Large-bank indexed/chunked coverage and true preemption are future work.
- Up to 128 stored proposal audits and 256 attempted source signatures. Capacity
  stops work, not silent eviction. Owner export/offline maintenance is currently
  required at these limits; no automatic proposal deletion or cleanup UI exists.
- Generation uses the existing last-eight bounded tool candidates and optional
  bounded goal excerpt, **not a complete day/session archive**. Ordinary failures
  remain recovery-only. User preference/decision drafts require a user source;
  that structural rule does not prove that the interpretation is correct.
- Structural checks and secret-pattern refusal are not semantic verification or
  complete secret detection. Manual review is essential, including source truth,
  misleading tool outputs, hallucinated lessons and conflict decisions. Drafts
  explicitly remain unverified and never become permissions.
- Proposal audits contain potentially private source text, including discarded
  temporary records. Full owner export includes them; curated-record backups do
  not. Protect service storage/backups. Linux OS isolation remains unaccepted.
- No automatic live-context deletion, hidden reasoning/KV capture, persistent
  memory compaction, authenticated multi-user access or new model tools is added.

Offline scripted tests cover generation/acceptance, conflict keep/rewrite/discard,
same-layer scope, stale/busy/disk failures, promotion/sort gates, bounded refusal,
owner CLI and service authorization. Actual model behavior and target performance
remain unchecked.

Verification: the full 15-file offline extension run passed 348 cases. After
adding six malformed-response cases, the final review-file rerun passed all39
cases (354 distinct cases across those runs). Targeted Ruff and the existing
Hermes health rules applied to all10 changed Python files pass, with zero health
findings. The stock health CLI assumes Hermes is the repository root and cannot
locate its pyproject in this combined layout; this was a scoped rules run, not a
claim that every repository CI guard ran. Tests used scripted model outputs and
disposable banks only. No model, GPU, engine or live service was started.
