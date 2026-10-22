# Research-only security preparation — 22 October 2026

Implemented in Hermes; nothing activated. Intended first deployment: one isolated
Brave Origin browser for public research and staged memory tools only. Coding,
terminal, arbitrary file/desktop access and downloads/uploads are outside scope.

## Changes

- New owner-selected research approval scope, stored in the broker database.
  Only public-HTTPS page handoffs survive submission and approval checks; exports,
  downloads/uploads and consequential actions are refused. This is manual handoff,
  not browser execution. Legacy brokers retain their manual scope.
- Matching staged discovery exposes only page handoff and request status in this
  scope. Owner configuration, not model arguments, chooses the bridge scope.
  Configure both bridge and broker as research if using this optional bridge.
- Actual memory-only launcher uses exact configured model/memory endpoint guard;
  other loopback ports, Unix socket connections, binds and datagrams are refused.
  Python auditing is defense in depth, not containment of arbitrary native code.
- Both local HTTP services use a five-second total request-read deadline, limiting
  slow-fragment occupation. Rate limiting/admission and dispatch deadlines remain
  separate pending work; servers remain serial.
- Legacy host-network recipe explicitly marked unsuitable for the new deployment.
  Requirements and remaining Fedora gates are in
  `local_extensions/e4b_memory/deploy/RESEARCH_ONLY.md`; checklist section21.

## Verification

Canonical CPU-only test runner: **382 passed, zero failed**, 17 test files, including
23 new research-specific cases. Includes broker persistence/legacy compatibility,
approval revalidation, scope-limited discovery, exact endpoint rejection and a
fake-clock fragmented-read deadline test. Existing memory/access/security/step
recovery/discovery regressions also passed. Repository `scripts/check --base HEAD`:
12 checks passed; targeted Ruff passed.

These are offline behavior/static checks, not Fedora penetration testing. No
Hermes application, browser, model, engine, GPU, WSL or Linux setup script ran.
The `Linux thing` folder was not changed or executed.

## Still blocked before activation

Automatic search/open/read adapter, dedicated browser profile and verified Origin
policies, private network transports, OS identities/namespaces, immutable code,
Windows-mount/filesystem/desktop denial and public-only browser egress. One installed
browser does not prevent local-file access. Current launcher remains memory-only;
do not enable general Hermes browser, code, shell, file or desktop tools as a shortcut.
