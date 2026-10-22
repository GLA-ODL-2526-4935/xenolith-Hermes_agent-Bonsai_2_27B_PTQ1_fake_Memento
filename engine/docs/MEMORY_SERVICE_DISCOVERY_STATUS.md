# Memory service and general tool discovery — 20 October 2026

## Implemented, opt-in and not activated

### Separate memory service

- Service owns the existing JSON memory bank and its approvals, retention,
  recovery, per-turn retrieval budget and idle maintenance.
- `e4b_memory` can select `backend: service`. The remote provider opens no local
  Store and has no local-memory fallback on failures. The local prototype remains
  available for compatibility; no live profile was changed.
- Owner-issued model/controller credentials are distinct, revocable and hashed
  at rest. Model calls use only the existing memory bridge. Lifecycle calls only
  prepare unverified review candidates and task state; neither role can approve,
  edit/delete, export the bank or change maintenance settings.
- Owner administration remains on the service host, not in a model tool or HTTP
  approval endpoint. Session IDs are prefixed with owner-issued client IDs.
- Restricted new-profile generation supports service memory without adding any
  filesystem, shell, browser or other work tools.
- Ordinary local owner commands refuse service mode, preventing accidental
  creation of a second local bank. No actual user memory was migrated/deleted.

### General tool discovery

- Extend Hermes' existing bridge, preserving its normal execution, hooks and
  approval path instead of introducing a parallel dispatcher.
- Opt-in setting: `tools.tool_search.discovery_mode: sections`, with search enabled.
- Initial bridge lists section names only; the selected section returns tool names
  and descriptions capped at 160 characters; `tool_describe` returns one selected
  schema; `tool_call` invokes the chosen tool through normal Hermes dispatch.
- The catalog is rebuilt from the current session grant. Globally registered but
  ungranted tools cannot be discovered/called through this bridge.
- Most registered core/plugin/MCP tools can be deferred. Four agent-loop-native
  tools (`memory`, `todo_list`, `session_search`, `delegate_task`), when enabled,
  stay direct in this increment; custom non-registry provider tools also remain
  direct (the current memory provider already exposes one small staged bridge).
- Oversized sections/schemas return a budget error, not an uncontrolled dump.
  Dynamic hosted connector discovery is not enabled by sections mode.
- Tool schemas stay stable after discovery. No previous messages or KV entries
  are deleted; trained Memento/context eviction remains a separate workstream.

Example future config, applied only when starting a new validation session:

```yaml
tools:
  tool_search:
    enabled: "on"
    discovery_mode: sections
```

This changes disclosure, NOT permission. Only enable reviewed tools; section
discovery must never be used as a reason to turn on shell/code/file/plugin tools.

## Verification

Subsequent offline hardening run: **234 passed, zero failed**. The initial adversarial
run exposed duplicate Authorization headers and duplicate JSON keys being accepted;
both are now strictly rejected in memory and access service handlers. Neither initial
test demonstrated authentication bypass. Both original failing contracts now pass,
alongside 34 added parser cases and all prior tests. All 12 static checks pass.
See `OFFLINE_SECURITY_TEST_REPORT.md`; Linux isolation and live acceptance remain pending.

155 CPU-only tests passed on Windows Python 3.14.7: 56 prior project tests,
16 new service tests, 11 new section tests and 72 upstream discovery regressions.
All 12 repository static checks passed. The suite covers owner CLI, actual plugin
selection, service-role denial/revocation, checkpoint gates, recovery and no-local-
fallback behavior, plus real tool dispatch, denied approvals and session scoping.

No live server, Hermes application, browser, model, engine or GPU was launched.
No model weights were downloaded. Test dependencies were prepared in an isolated
environment through Hermes' package manager; project dependencies were unchanged.

## Still unchecked

- Linux users/ACLs/mounts that actually deny agent access to the service bank.
- Safe cross-namespace transport, process supervision and live service integration.
- Real E4B/Hermes end-to-end behavior and foreground latency during idle sorting.
- Multi-user/group authorization; this is one owner's bank, not private tenants.
- Automatic migration and full backup/restore/rollback of a populated local bank.
- Section deferral of remaining agent-loop-native tools and actual Memento KV removal.
- Browser sandbox, download interception and other previously deferred Linux work.
