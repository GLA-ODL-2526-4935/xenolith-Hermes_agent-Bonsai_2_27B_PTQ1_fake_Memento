# V4 recovery security hardening — 21 October 2026

## Result

Implemented in the existing Hermes checkout. **289 selected offline CPU-only tests
passed, zero failed**, across 12 files on Windows Python 3.14.7. All 12 repository
static checks passed; code health had zero blocking/advisory findings and targeted
Ruff passed. No new dependencies.

No live service, Hermes application, model, engine, browser or GPU was started.
Only disposable test banks were used. No actual memory, secrets or live profile
was migrated, edited or activated. Hermes remains in its existing location.

## Implemented changes

### Trust metadata

- Validate recovery structure, fixed `verified: false` / `controller_reported`
  semantics, revision ordering, timestamps, content and digests before list/read/
  retry. A damaged record cannot return a success receipt just because its payload
  digest still matches.
- New records include an additional integrity hash covering stored metadata as
  well as content. The original content digest remains stable for idempotency.
- Listings and detailed reads refuse invalid labels rather than silently declaring
  the record verified. Derived role/policy labels are generated on return.
- Legacy records without the additional hash still require strict fixed trust
  semantics and validation. Their old digest does not protect timestamps/revisions.
  A hash is not authentication against someone able to rewrite the bank and hashes.

### Source authority and negation

- Requirement/decision facts must quote an entire supplied visible user event;
  a tool event cannot be categorized as either, and extracting an affirmative phrase
  from a negative user clause is refused.
- Observation/proposal quotes may remain bounded exact substrings with their original
  source events retained. Detailed reads include a derived source-role map and a
  reference-only/no-permission policy. Source/fact text is not silently rewritten.
- Recovery cannot create a live approval or curated-memory append stage. Existing
  model/controller role separation and retrieval budgets remain in force.
- This is not authentication of the controller's claimed source identity or proof
  of completeness/truth. Automatic extraction/context replacement stays disabled.

### Sensitive source text and retention

- Before saving a step, reject recognizable password/API-key/token assignments,
  Bearer credentials, selected vendor-token forms and private-key envelopes.
- Refusal errors do not echo the credential. The whole step fails atomically;
  exact evidence is never silently redacted. The controller must provide an
  authorized non-secret source excerpt and retain old context on save failure.
- Expiry uses each step's own original save timestamp, not its session's latest
  addition. New steps no longer renew old ones. Foreground activity still defers
  deletion; remaining revision counters are preserved for a partially expired bank.
- Older sensitive/source-policy-invalid records are blocked from model reads but
  may still expire normally when structurally valid. No populated bank was scrubbed.

## Compatibility and tests

The original 23 recovery tests pass. Two test fixtures that used partial requirement
quotes now use the full user source, reflecting the intentionally stricter contract;
disk-failure, persistence, scope, revision and budget assertions are retained.

Added 32 security cases covering list/detail metadata edits, retry/readback corruption,
legacy labels, tool-as-requirement/decision, negation dropping, inert recalled approval
claims, credential refusals, exact non-secret values, canonical roles and per-step
expiry including legacy content. The initial 31-case pre-fix run exposed 30 failures
and one passing control; an expiry-compatibility case was added subsequently.

Final selected suite = 234 prior memory/access/discovery cases + 23 recovery cases
+ 32 recovery-security cases = **289 passing**. Tests use the canonical runner:

```text
scripts/run_tests.sh tests/local_extensions/ tests/tools/test_tool_search_sections.py tests/tools/test_tool_search_security.py tests/tools/test_tool_search.py tests/tools/test_tool_search_multiquery.py -j 1 --file-timeout 180
```

Production changes are confined to `local_extensions/e4b_memory/step_recovery.py`,
`maintenance.py` and new sibling `step_policy.py`. Core agent execution, model routes,
tool permissions and curated-memory approval rules were not changed. Active plan,
checklist and recovery status were updated in the Hermes folder.

## Still required

- Recognizable-pattern screening is not a general secrets/private-data detector;
  encoded/unlabeled secrets can be missed and legitimate examples may be refused.
  Typed capture allowlists, controller provenance and authorized source selection
  remain required before automatic capture. Existing task state/logs/backups are
  outside this new step-specific screening; do not claim they were sanitized.
- Owner reconciliation is required for older records violating stricter capture
  rules. No silent migration/rewrite or forged-source repair was introduced.
- Linux identities, mounts, file permissions and network egress must still deny
  the agent direct access to bank/owner credentials. Encryption/backup/log policy
  and live transport acceptance remain deployment work.
- Automatic consumers need independent tests for source identity, supersession,
  omitted restrictions, pending actions and approval replay before deleting history.
- Model prompt-injection behavior, multi-user isolation and real-model/live-service
  acceptance are not certified by these offline tests.

These changes address the reproduced gaps without claiming that recalled text is
trusted or that the deployment is fully contained.
