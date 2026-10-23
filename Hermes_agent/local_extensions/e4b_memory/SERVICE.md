# Service-backed single-owner memory

Optional remote owner acceptance: [DISCORD_REVIEW.md](DISCORD_REVIEW.md).
The dedicated bot runs owner-side against the existing bank; model/controller
HTTP credentials still cannot accept or reject proposals.

Optional automatic drafts/conflict checks with manual owner decisions:
[AUTOMATIC_REVIEW.md](AUTOMATIC_REVIEW.md). Configure on the service host, not the
remote agent; model/controller credentials gain no acceptance capability.

Implemented source, opt-in, NOT activated or Linux-isolated. No live service,
Hermes application, model, browser, engine or GPU was launched during development.

## Boundary

The service owns `memory/memory.json`, review/approval stages, retrieval budgets,
recovery, idle promotion/sorting and retention. The remote Hermes provider opens
no local Store and never falls back to one if the endpoint or credential fails.
The original local backend remains available for compatibility, explicitly marked
`backend: local` (also the legacy default). No existing live profile was changed.

The client API has two owner-issued credentials:

- **Model:** section/catalog/describe/read and stage-gated append through the
  existing memory bridge. No full-store export, approval, edit/delete or maintenance.
- **Controller:** bounded task start/finish, unverified tool assertions,
  pre-compression capture, explicit immutable `save_step` recovery and session close. Cannot approve evidence or rewrite
  memory. New model-side sessions must first be established by the controller.

Neither credential provides owner administration. Owner-issued client IDs namespace
session/checkpoint IDs. This remains ONE owner's memory bank; it is NOT multi-user
ACL enforcement. Different client credentials are not different authenticated users.
Persistent memories remain available across the owner's sessions by design.

The v4 step-recovery path requires an actively begun controller instance and
returns a disk-readback receipt, never a verified-memory grant. Models can list/read
only their session's steps through the staged recovery section. No automatic event
capture or strict compaction-checkpoint API v2 is enabled. See
`../../STEP_RECOVERY_STATUS.md` for payload, limits and failure/retry semantics.

The service runs maintenance on its own loop. Default promotion is deterministic;
optional layer classification is configured on the owner side and uses the same
explicit local model. A sorter request can block this basic serial server while
it completes; foreground latency/concurrency scheduling is not certified.

Only bounded tool-result assertions and a separate latest error cross the lifecycle
API, not full transcripts, hidden assistant reasoning, KV or media. Assertions are
still unverified candidates and require owner approval. Legacy compression API v1
is retained: this does not promise fail-closed transcript archival before every reset.

## Owner setup later, in the validation environment

Use a NEW dedicated directory owned by the service account, outside the agent's
mounts. From the source checkout:

```
python -m local_extensions.e4b_memory.service SERVICE_DIRECTORY init
python -m local_extensions.e4b_memory.service SERVICE_DIRECTORY client
python -m local_extensions.e4b_memory.service SERVICE_DIRECTORY serve --port 8767
```

`client` prints `client_id`, `model`, `controller` once. Store the two tokens as
scoped secrets `E4B_MEMORY_MODEL_TOKEN` and `E4B_MEMORY_CONTROLLER_TOKEN`; never
send them to the model or embed them in config/source. Only token hashes are
stored by the service. `revoke CLIENT_ID` invalidates both credentials. Revocation
does not silently clear abandoned activity markers; perform explicit owner recovery
with the agent stopped when necessary.

On the agent side, select `memory.provider: e4b_memory`, disable both built-in
`memory_enabled` and `user_profile_enabled`, and set:

```yaml
memory:
  provider: e4b_memory
  memory_enabled: false
  user_profile_enabled: false
  e4b_memory:
    backend: service
    service_url: http://127.0.0.1:8767/v1/memory
```

Alternatively, the existing restricted profile generator accepts
`--memory-service-url http://127.0.0.1:8767/v1/memory` during `init`.
It still exposes memory only and refuses arbitrary extra tools. Configuring a
service profile does not start the service or copy existing memory to it.

Owner administration stays on the service host:

```
python -m local_extensions.e4b_memory.service SERVICE_DIRECTORY owner status
python -m local_extensions.e4b_memory.service SERVICE_DIRECTORY owner requests
python -m local_extensions.e4b_memory.service SERVICE_DIRECTORY owner approve CLIENT_ID:SESSION_ID EVIDENCE_ID
python -m local_extensions.e4b_memory.service SERVICE_DIRECTORY owner export
```

The other existing owner memory commands remain available under `owner`.
The ordinary profile-local owner CLI refuses service mode instead of creating a
misleading second local store. Review approved summaries as well as their evidence;
approved evidence alone does not prove every sentence of a model's summary.

Stop the service before `configure SETTINGS_FILE`. It atomically replaces settings
under the service lock. The file must have exactly `semantic_sort`, `l0_days`,
`recovery_days`, `base_url`, `model`; sorting defaults to false. No cloud fallback.
The serial service lock prevents two live service instances on the same root.

## Limits and later work

- Loopback HTTP, fixed path, bearer authentication, no proxy/redirect, origin/Host
  checks and bounded bodies/responses are implemented. Live socket/process
  deployment has not been exercised here; handler/client tests used in-memory transports.
- Credentials: 32 issued clients per bank; 128 live surface keys per process.
  Abandoned work fails closed for idle maintenance until owner recovery.
- Do not mount the bank, owner commands or service credentials database into the
  agent. Same-account processes are not an OS boundary. Assign separate Linux users,
  filesystem ACLs/mounts and controlled transport before claiming file-access denial.
- Loopback is namespace-local. Separate container network namespaces require an
  explicitly reviewed relay/transport, not publicly binding this server.
- Existing local memory is not automatically imported or deleted. Migration,
  backup/restore and rollback need an owner-reviewed procedure before changing a
  populated deployment. No user's actual memory was moved by this change.
- Actual model behavior, Linux permission tests, multi-user authorization, browser
  isolation, strict checkpoint API v2 and trained Memento remain unfinished.

## Verification

Security hardening follow-up: **234 selected CPU-only tests pass**, including both
original parser failures unchanged and 34 added strict-parser cases. All 12 static
checks pass. Duplicate security-sensitive headers/JSON keys, incomplete bodies,
nonstandard NaN/Infinity literals and excessive JSON depth are now refused.

Both service packages require the shared `local_extensions/e4b_http.py` alongside
their directories; retain that file when packaging/copying these extensions.
The following count records the earlier implementation increment:

16 new memory-service tests and 11 section-discovery tests, the prior 56 project
tests and 72 existing tool-search tests passed: **155 CPU-only tests total** on
Windows Python 3.14.7. All 12 repository static checks passed.
