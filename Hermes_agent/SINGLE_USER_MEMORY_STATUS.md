# Single-user memory status — expanded implementation, 20 October 2026

> 21 October update: [plan v4](Local_Gemma4_E4B_Hermes_Xenolith_Memento_Plan_v4.md) is active. See [step recovery](STEP_RECOVERY_STATUS.md) for the new separate controller records. Verification below is historical. Required trained-Memento deployment is superseded, not completed.

Hermes remains at its existing checkout. The opt-in provider source is in
`local_extensions/e4b_memory/`; no live profile has been changed or activated.

## Implemented

- Temporary/persistent L0–L3, text-only JSON and append-only main-model writes.
- Staged section/catalog/schema retrieval, capped at three expanded records and
  16,000 returned characters per turn. Discovery state resets at compression.
- Automatic checkpoint-review preparation after five idle minutes and before
  compression. Only bounded tool-result assertions become review candidates;
  assistant claims, hidden reasoning, media and routine failures are excluded.
- Owner approval opens an expiring, session-bound append stage. The model cannot
  self-approve evidence, edit/delete memory or open its own write stage.
- Separate session task/recovery state with the latest error; owner task completion
  and crash recovery. Failed attempts are not promoted into curated memory.
- Atomic promotion after three idle hours. Optional semantic layer classification
  uses the same configured loopback model, not another resident/remote model.
  Model output cannot rewrite text; invalid/stale plans leave temporary data intact.
- Retention: persistent L0 30 days after promotion; pending review one day; recovery
  seven days. No automatic L1–L3 expiry or silent deletion of unpromoted memories.
  Maintenance/expiry pauses while tracked foreground turns are active.
- Owner CLI inspection, approval, editing/deletion, sorting, expiry and export.
- New-profile generator and restricted launcher: memory tool only, built-in memory
  disabled, local routes pinned, one launcher per profile and non-loopback Python
  networking refused. Existing profiles are never overwritten.
- Linux non-root/read-only-root/container deployment recipe supplied, NOT deployed.

## Verification

39 CPU-only tests passed through `scripts/run_tests.sh tests/local_extensions/`
on Windows Python 3.12. Coverage includes real provider discovery, owner CLI,
permissions, restart/failed writes, concurrency, review gates, recovery, retention,
sorting/stale results, retrieval limits, profile creation and launcher policies.
Targeted Ruff checks, the repository code-health ratchet and all 12 repository
static checks passed.

No Hermes application, model, Xenolith engine or GPU was run. No model weights
were downloaded. Disposable test/lint environments were prepared through Hermes'
package manager outside the checkout; no project dependencies were changed.

## Still later / not certified

- Real E4B/tool-template and Hermes CLI end-to-end tests; production Python 3.14
  and Linux target checks. The optional sorter transport was tested with scripted
  replies, not an actual trained model.
- Actual OS isolation deployment. The restricted launcher is a model-tool/network
  guard, NOT an OS boundary; arbitrary host shell/file/code/MCP tools remain unsafe
  with this in-process store. Direct physical store separation needs a sidecar or
  isolated backend before expanding the work-tool surface.
- Task-specific automatic truth verification. Review preparation is automatic;
  approval remains human-controlled, and model summaries still need review.
- Trained Memento, Xenolith serving, multi-user authorization and second-laptop
  memory maintenance/compaction. No remote-maintenance link was added.
- Strict checkpoint-required compression API v2 and complete transcript archival
  are not claimed. Sorting classifies layers only; it does not semantically merge,
  resolve contradictions or rewrite existing memories. Large sort backlogs need
  owner maintenance (32 records/60 KB per sorter request).
