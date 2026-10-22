# Research-only target (not activated)

Current scope: public-web research in one dedicated Brave Origin browser, plus
the existing staged memory tools. No coding, terminal, desktop control, arbitrary
files, uploads, downloads, extensions, account actions or personal browser data.
Web text and retrieved memories are untrusted data, never permission grants.

## What is available now

- The restricted launcher remains memory-only and refuses additional tools.
- Its Python network guard allows only the configured model/memory endpoints,
  not every loopback port. This is defense in depth, not an OS sandbox.
- The owner may initialize a **new**, disabled approval broker with
  `python -m local_extensions.e4b_access DIRECTORY init --scope research`.
  Configure an optional bridge with `scope: research` as well. The broker persists
  this scope independently and rejects everything except public page handoffs.
  These are manual requests, not search, navigation or browser execution.
- Existing manual brokers remain manual; they are not silently converted.
- Request parsing has a five-second total read budget. Service dispatch and
  admission/rate limiting are separate, still pending safeguards.

## Required on Fedora before enabling browser tools

- Dedicated unprivileged agent/browser identities and isolated namespaces;
  read-only code/config/plugins, narrowly writable state separate from code.
- No host networking, host sockets, desktop session bus, Wayland/X11 control,
  wallet, home directory, personal browser profile or Windows mounts exposed.
  Removing other browsers does not prevent access to mounted Windows files.
- Agent egress only to its explicit private model/memory transports. Browser
  egress only to public web destinations; deny loopback, private/link-local ranges,
  IPv6 equivalents and metadata endpoints after DNS resolution and on redirects.
  Do not expose the memory service or its credentials to the browser.
- Design and test private service transport first. The existing loopback clients
  are not a cross-namespace transport; do not switch to host networking or bind
  services publicly to work around that missing piece.
- Keep SELinux enforcing, Chromium sandbox enabled, no root/privileged container,
  no `--no-sandbox`, no container socket and no model-facing remote-debugging port.
- Build a trusted constrained adapter: search, open and read only. No arbitrary
  JavaScript/evaluate, extension installation, file chooser, download or general
  browser debugging API. Keep downloads manual in a separate owner context.
- Validate the installed Brave Origin managed-policy support/path and inspect
  `brave://policy`: block downloads, file pickers, local-file URLs, extensions,
  password saving/sync, geolocation, notifications, camera and microphone.
  Do not assume regular Brave's managed-policy directory applies to Origin.
- Test browser escapes, DNS rebinding/redirects, uploads/downloads, filesystem and
  Windows mount denial, service credentials, restart and memory promotion before
  activation. Policies alone are not a filesystem or network security boundary.

Automatic browsing is deliberately unavailable until these gates pass. Memory
records remain text and owner-editable; the model gets only the approved memory
API, not memory paths. Another-model maintenance and multi-user isolation remain
later work. No Linux deployment, browser, model or engine was run for this change.
