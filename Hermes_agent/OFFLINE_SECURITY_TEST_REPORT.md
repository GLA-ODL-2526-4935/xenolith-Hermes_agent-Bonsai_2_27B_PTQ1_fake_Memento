# Offline security test — 20 October 2026

## Result

**234 passed, zero failed** across ten selected test files on Windows Python 3.14.7
after hardening. The original run was 198 passed/two failed; both failing contracts
now pass unchanged. A further 34 request-parser cases cover both memory and access
services. All 12 repository static checks passed, including code health (zero
blocking/advisory findings), and targeted lint passed.

This is a scoped offline check, not a penetration-test certification or deployment
approval. No Hermes application, live service/socket, browser, model, engine or GPU
was launched. Only disposable test banks were used; no actual owner memory, profile
or credentials were accessed or changed. Generated test credentials are disposable.

## Findings — fixed and regression-tested

### 1. Duplicate authorization headers accepted

The original custom memory HTTP handler used the first Authorization header rather than
requiring exactly one. An in-memory request containing a valid model credential
followed by a conflicting invalid credential returned HTTP 200 instead of refusal.

This demonstrates ambiguous parsing, **not unauthenticated access or controller
privilege escalation**. It requires a valid credential in the tested request.
Different components could interpret duplicates differently if transport components
are later introduced. Both custom services now require exactly one Authorization,
Host, Content-Type and Content-Length header, including case-insensitive duplicates.

Reproduction contract: `tests/local_extensions/test_e4b_memory_security.py`,
`test_duplicate_authorization_is_rejected`.

### 2. Duplicate JSON field names accepted

The original handler's ordinary JSON parser accepted repeated object keys, selecting the last
value. A request containing both `session: other` and `session: session` returned
HTTP 200 under the already authorized session instead of rejecting the ambiguity.

This does **not** demonstrate cross-client access: credential-based client prefixes
are still applied. Both custom services now reject duplicate keys in every JSON
object, including nested arguments, before dispatch.

Reproduction contract: `tests/local_extensions/test_e4b_memory_security.py`,
`test_duplicate_json_keys_are_rejected`.

Both findings concern the newly added local memory-service handler, not a verified
upstream Hermes vulnerability. The initial test-only increment left behavior unchanged;
the owner then authorized hardening. The original failing tests now pass, not skipped
or weakened. Shared parsing is in `local_extensions/e4b_http.py`; both custom service
handlers use it. No core agent-loop, tool permissions or memory approval policy changed.

Additional hardening rejects even empty Origin/Transfer-Encoding headers, incomplete
bodies, non-decimal/oversized length fields, NaN/Infinity literals and JSON nesting
beyond 64 containers. UTF-8 Unicode text and quoted brackets remain supported.
Authentication still happens before body parsing. Ordinary requests work after a
refusal, and the existing functional/owner-approval regression suite stays green.

## Controls exercised successfully

- Invalid/revoked credentials and unauthenticated malformed requests are refused.
- Browser-origin, transfer-encoding, duplicate-length, route and host confusion
  probes are refused without modifying the bank.
- Invalid, deeply nested and oversized JSON inputs receive controlled refusals in
  this Python runtime. Session/instance traversal and malformed identifiers fail.
- A different client or instance cannot borrow an established memory surface;
  session recovery does not disclose another client's latest error.
- Expired/cross-session append grants, wrong evidence kinds and exhausted write
  budgets are refused. Model credentials cannot reset retrieval limits through
  lifecycle/controller operations.
- Existing tests retain owner-only administration, no automatic self-approval,
  no local fallback on service failure, approval-bound exports and restricted paths.
- Guessed unauthorized tools, invalid tool arguments and mixed local-tool batches
  do not reach handlers through the real Hermes dispatch path.
- Prior discovery tests retain session-scoped catalogs/schemas and owner hook denial.
- Injection-like catalog text remains untrusted returned data; discovery neither
  executes it nor grants a guessed terminal tool. This does not prove a future model
  will ignore that text.

## Limits and remaining work

- No Linux ACL/mount/user/container/network boundary was exercised. A general file,
  shell, code or plugin capability must remain disabled unless actual isolation
  prevents reaching service storage and owner secrets.
- No claim that owner-approved evidence semantically proves every model-written
  sentence: evidence-kind/ID gating is not automatic truth verification.
- Client/session namespace separation is not multi-user private memory; persistent
  records are still one owner's bank.
- No live HTTP transport, proxy, concurrent-client, timing/load or OS escape test.
- No adversarial model/webpage benchmark, browser automation or download-interception
  acceptance test. A model is unnecessary for permission checks but necessary for
  measuring its prompt-injection susceptibility.
- The two reproduced parser gaps are fixed; the selected suite is green. Live service
  acceptance still requires the deployment and adversarial checks above.

## Re-run

From the Hermes checkout, with the isolated test interpreter selected:

```text
scripts/run_tests.sh tests/local_extensions/ tests/tools/test_tool_search_sections.py tests/tools/test_tool_search_security.py tests/tools/test_tool_search.py tests/tools/test_tool_search_multiquery.py -j 1 --file-timeout 180
```

The canonical runner isolates the profile and clears credentials. The new tests
are `tests/local_extensions/test_e4b_memory_security.py`,
`tests/tools/test_tool_search_security.py` and the hardening follow-up's
`tests/local_extensions/test_e4b_http_security.py`. No dependencies were changed.
Production changes are confined to the two opt-in service handlers and their shared
parser; no live profile was activated or owner memory migrated.
