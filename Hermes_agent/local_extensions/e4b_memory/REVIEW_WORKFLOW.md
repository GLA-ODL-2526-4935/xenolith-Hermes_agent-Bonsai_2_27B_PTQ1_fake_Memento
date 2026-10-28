# Readable memory review and archival

Implemented owner-side, not enabled in a live profile. No model tool, automatic
acceptance, browser permission or system access is added.

## Markdown review

`hermes e4b_memory review-memory --id ID` now prints Markdown by default. Redirect
it to a private `.md` file to read in your preferred viewer. `--format json`
retains the machine-readable view. The report contains numbered drafts, layers,
memory text, evidence, and the full unchanged audit snapshot. Conflict reports
show the current temporary record, proposed action/replacement, explanation and
all persistent records that were compared in its layer. Untrusted text is quoted
with dynamically sized code fences; it cannot escape those fences to create
rendered links/HTML in a conforming Markdown viewer. This is presentation, not
semantic prompt-injection protection for the model.

Owner commands below use the selected existing Hermes profile. For service-backed
storage, run the corresponding `python -m local_extensions.e4b_memory.service
/absolute/service-root owner COMMAND ...` on the owner/service host instead.

```text
hermes e4b_memory review-memory --status pending
hermes e4b_memory review-memory --id ID --format markdown
hermes e4b_memory accept-memory ID --drafts 1 3
hermes e4b_memory reject-memory ID
hermes e4b_memory memory-history --status partially_accepted
hermes e4b_memory memory-history --status stale
```

Selecting drafts 1 and 3 accepts **only those entries and rejects all remaining
drafts in that proposal**, in one atomic decision. The original result/evidence
remains in its audit with accepted/rejected numbers. The proposal then closes;
there is no later acceptance of its remaining entries. Omitting `--drafts`
accepts the whole proposal as before. Selection is generation-only: a conflict
keep/rewrite/discard proposal remains one indivisible decision. Existing current
source checks, busy-state refusal and conflict-before-promotion gates remain.

Discord previews now use `memory-proposal.md`. After reading the full attachment,
append comma-separated numbers to the bound approval command:
`!memory accept ID CODE 1,3`. Unselected drafts are rejected. Omitting numbers
accepts all. Existing owner-DM, delivery, one-use and five-minute checks still
apply. Privacy/export consent is unchanged; see [DISCORD_REVIEW.md](DISCORD_REVIEW.md).
No remote archival/configuration commands or automatic notifications are added.

## History and durable archival

History lists pending, stale, accepted, rejected and partially accepted proposals.
Stale means its source or temporary/persistent comparison no longer matches; it
is computed without changing the original audit. Archiving is **owner-only** and
allowed for closed proposals or stale pending proposals, never current pending
ones. Preview and confirm explicitly:

```sh
hermes e4b_memory archive-memory-preview ID1 ID2 > archive-plan.json
# Inspect that plan before confirming; do not run while foreground work is active.
hermes e4b_memory archive-memory archive-plan.json
hermes e4b_memory memory-archives
hermes e4b_memory review-archive ARCHIVE_FILENAME.json > archived-review.md
```

The archive operation requires the exact preview revision and selection digest.
It writes a private full JSON audit backup under the bank's `proposal-archives/`,
flushes it and verifies readback **before** removing live queue entries. Memory
records and their evidence are not removed. JSON stays the authoritative storage
format; Markdown is a readable view, not an editable authorization document.
Archived reports are read-only and never restore pending approvals.

A failed backup leaves live proposals untouched. A failure committing the bank
may leave both the backup and original; inspect state before repeating. Archives
are not automatically deleted. Protect them like the original private bank.
The live 128-proposal capacity is freed, and matching attempt entries are moved
to compact source-signature tombstones so rejected/accepted sources do not
immediately regenerate. Tombstones have a separate 4,096-signature safety limit;
that boundary still needs deliberate owner maintenance. Failed-job retry remains
a separate explicit action. Archived sources can become eligible again only when
their source/comparison basis changes, not merely because audits were archived.

## One-switch automatic generation/checks

The combined launcher's `automatic_memory_workflow` boolean controls automatic
draft generation and same-layer conflict checks for the next start. It defaults
to **false**. Acceptance remains **manual**, including when this switch is true.
No automatic-acceptance mode is implemented by this change.

For an existing generated local profile or service config, stop its owning
Hermes/service first, then use the owner helper from the Hermes environment:

```text
python -m local_extensions.e4b_memory.workflow_switch /absolute/profile/config.yaml on --kind local --stopped
python -m local_extensions.e4b_memory.workflow_switch /absolute/service/settings.json on --kind service --stopped
```

Use `off` to disable, then restart. The helper supports the generated JSON-subset
profile format, preserves other settings, validates the local endpoint/config and
atomically saves the switch. Service settings require an explicit valid local
model before enabling. Service-backed agent profiles must not enable owner-side
inference; switch it in the service settings instead. `--stopped` is your explicit
confirmation, not detection/termination of another process.

Turning the switch off does not erase drafts, evidence, archives or existing
review requirements. It never permits already marked temporary records to bypass
conflict acceptance. Already authorized idle promotion and normal retention can
still run; this is not a freeze-all-storage switch.

Actual model fidelity, manual workload, Linux isolation and live Discord rendering
remain validation tasks. Full memory and browser behavior is not certified by
scripted offline checks.

Verification for this increment: all 18 offline local-extension files passed,
441 cases total, including 30 review-usability and 28 scripted launch cases.
Targeted Ruff passed. The existing Hermes health engine scoped to 12 changed
nested Python files passed with no blocking/advisory findings (the standalone
owner-token exception is retained). The standard health wrapper was attempted
but still assumes Hermes is the Git root and fails to find the combined root's
pyproject; no checker or CI guard was weakened. The root start.py passed targeted
Ruff. No real inference subprocess, model, GPU, Discord connection or Linux setup
was launched. Live process-group/readiness/desktop behavior remains target work.
