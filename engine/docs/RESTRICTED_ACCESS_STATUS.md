# Restricted browser/file access — 20 October 2026

## Implemented, not activated

New standalone extension source is in the existing Hermes checkout at
`local_extensions/e4b_access/`. No engine or existing memory behavior was changed.

- Staged proposal tool: section names, brief operation catalog, selected schema.
- Separate broker process implementation with a client-credential-protected,
  proposal/status-only loopback API. No approval API is available to the model.
- Owner-only inspect/approve/reject commands, exact-content digests, one-hour
  request expiry, bounded queues and transactional state.
- Approved text-export creation in one fixed folder, generated `.txt` filenames,
  exclusive creation and no model-controlled paths, overwrite or arbitrary reads.
- Browser/download/upload/form/message/purchase/account requests return manual
  owner handoffs only. They do not launch a browser or perform the requested action.
- Conservative URL screening, no proxy/redirect broker client and a default-disabled
  Hermes plugin. No shell, process execution, general file or browser-code tools.

56 CPU-only tests passed: 17 access tests and 39 memory tests. Targeted lint,
code-health and all 12 static repository checks passed. The new HTTP handler was
tested in memory without opening sockets. No browser, live service, Hermes agent,
model, engine or GPU was launched. No live profile was modified.

## Linux/deployment work remains unchecked

The separate-process implementation is not an installed security boundary.
Linux OS users/containers, mount restrictions, browser isolation, egress controls,
download interception/quarantine, safe transport between namespaces, actual
browser adapter and a new explicitly restricted launcher still need deployment
and adversarial testing. The current memory-only launcher intentionally refuses
the new access tool.

The compatibility memory backend remains in-process, but an opt-in separate memory
service and no-local-fallback provider are now implemented. See
`MEMORY_SERVICE_DISCOVERY_STATUS.md`. The access broker does not migrate memory;
Linux permissions must still deny the agent direct access to service storage before
arbitrary work tools are considered. The earlier 56-test result above is historical;
the expanded service/discovery suite now passes 155 selected CPU-only tests.

Browser URL screening is not DNS-rebinding or subresource protection. Even after
deployment, browser content can mislead the model; permissions limit consequences,
not the truthfulness of its answers. Browser automation is not claimed complete.

See the functional checklist for individual completed and deferred items.
