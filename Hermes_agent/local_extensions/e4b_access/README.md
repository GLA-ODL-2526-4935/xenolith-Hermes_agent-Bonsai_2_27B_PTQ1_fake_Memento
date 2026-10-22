# E4B restricted access broker

Opt-in source only; not activated or deployed. No browser, agent, model or GPU is
started by importing this extension. Uses only the Python standard library.

Keep shared `local_extensions/e4b_http.py` alongside this package when copying it.
Both custom services use its strict header/JSON parser. Duplicate security headers
or JSON keys, truncated bodies and excessive nesting are refused. The expanded
offline suite passes 234 selected CPU-only tests; Linux containment remains pending.

## Implemented now

- One small `e4b_access` Hermes bridge: sections → brief catalog → selected schema.
- Model-side HTTP exposes **submit/status only**, with a client credential.
- Requests: open page, download, upload, form submission, message, purchase,
  account change, or plain-text export. Every request needs owner review.
- Browser approvals return a **manual handoff**, not executable approval tokens.
  No browser, download, HTTP fetch of the proposed URL or external action occurs.
- Text exports are created by the owner approval operation in one fixed broker
  `exports/` directory, named by request ID with `.txt` extension. No model paths,
  arbitrary reads, edits, deletes, execution or overwrite. Failed/partial writes
  require owner intervention; matching complete writes can recover after a crash.
- Exact-payload SHA-256 approval, one-hour expiry, one decision per request, bounded
  inbox/text/request/response sizes, transactional state and exclusive file creation.
- Credential-free public HTTPS URL screening. This is NOT DNS/subrequest protection.
- Numeric-loopback client, no proxies/redirects, bounded timeout; server rejects
  browser Origin headers, wrong Host, framing ambiguity and unknown API operations.

The bridge is disabled unless its plugin settings explicitly enable it. It never
registers owner decisions, shell, code execution, raw HTTP or arbitrary file tools.
The existing memory-only launcher still refuses this extra tool. No existing
profile or launcher allowlist was changed.

## Owner operation — later, in the validation/deployment environment

From the Hermes source checkout, choose a NEW dedicated owner-side directory:

```
python -m local_extensions.e4b_access OWNER_DIRECTORY init
python -m local_extensions.e4b_access OWNER_DIRECTORY serve --port 8766
```

`init` returns the client token once. Put it in the scoped secret
`E4B_ACCESS_CLIENT_TOKEN`, never in model messages/config/source. The server stores
only its hash. Owner administration uses local filesystem access, not that token:

```
python -m local_extensions.e4b_access OWNER_DIRECTORY list
python -m local_extensions.e4b_access OWNER_DIRECTORY inspect REQUEST_ID
python -m local_extensions.e4b_access OWNER_DIRECTORY approve REQUEST_ID REVIEWED_DIGEST
python -m local_extensions.e4b_access OWNER_DIRECTORY reject REQUEST_ID REVIEWED_DIGEST
```

Review the whole payload as untrusted text before copying its digest. Approval
cannot change its URL/text/details. An approved download merely identifies what
the owner may download themselves. Files are inert exports, not automatically
opened, executed or added to memory.

Future explicit plugin installation uses the normal Hermes user-plugin directory.
Settings belong under `plugins.entries.e4b_access.settings`: `enabled: true` and
`endpoint: http://127.0.0.1:8766/v1/requests`. This is not sufficient to secure or
activate a full browser agent; keep it off until the Linux boundary is validated.
The loopback transport only works within its network namespace; do not expose it
on all interfaces to connect containers. A reviewed isolated transport/relay will
be needed for separate network namespaces.

## Linux/browser work still required

- Separate OS identities/processes/mounts: agent must not access broker/memory
  state, owner commands, browser profile, host secrets or arbitrary host files.
  A separate Python process under the same unrestricted account is NOT isolation.
- Read-only agent runtime, minimal scratch, no shell/code/MCP/subprocess tools,
  no Docker socket/host-control mounts; audited outbound network policy.
- New disposable browser profile, no inherited credentials/extensions/sessions,
  no local files, denied downloads and uploads by default, no browser execution
  APIs exposed to the model. Block service workers/background download channels.
- Browser adapter with enforced navigation/subrequest/redirect policy, public-only
  DNS/IP egress, private-network/metadata blocking and controlled outbound data.
- Browser action approvals tied to exact page/action/content and invalidated by
  navigation or page changes. Form/message/purchase/account effects need approval.
- OS-controlled download quarantine and exact one-shot download grants if automatic
  downloads are later enabled. Never allow a generic "download anything" grant.
- Wire a new explicit browser/file/memory tool allowlist into a reviewed launcher;
  do not weaken the current memory-only launcher to make an ad-hoc browser work.
- Security acceptance tests: malicious webpage, uploads/exfiltration, redirects,
  DNS rebinding, download popups, filesystem links/races and cross-process escapes.

No protection here can make model responses truthful or guarantee immunity to
prompt injection. Browser text and proposed exports remain untrusted. The current
single-user inbox has no multi-user identity/authorization guarantee.

## Verification

17 new CPU-only tests plus 39 existing memory tests passed (56 total), including
real plugin discovery, the HTTP handler without sockets, approval/expiry, restart,
overwrite refusal, URL screening and request routing. All 12 repository static
checks and targeted Ruff passed. No live service/browser/agent was launched.
