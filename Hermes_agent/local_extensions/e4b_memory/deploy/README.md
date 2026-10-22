# Optional Linux isolation recipe (not built or run here)

**Legacy prototype, not the research deployment.** Its host network and writable
profile do not meet the current research-only isolation requirements. Do not use
this recipe as approval to enable browser/work tools. See `RESEARCH_ONLY.md` for
the pending target requirements; no replacement deployment is activated here.

Use only after the CPU tests pass and a local model server is ready. Build from
the Hermes repository root using this Dockerfile and a reviewed Python 3.14 base
image digest supplied as `MEMORY_BASE_IMAGE`. Do not mount the Docker socket,
your home, source checkout, SSH credentials or other Hermes profiles at runtime.
Build only from a clean source export plus this approved extension, never a
working directory containing user state/secrets. The Dockerfile-specific ignore
file excludes common sensitive artifacts but cannot recognize every secret.

Runtime posture: non-root UID 10001, read-only container root, dropped capabilities,
<!-- no-tmp: ok — deliberate container tmpfs mount, not host scratch storage. -->
`no-new-privileges`, a bounded temporary `/tmp`, and only one dedicated writable
profile directory mounted at `/profile`. On Linux, host networking lets the
loopback-only client reach the existing local server without loading another model.
Host networking is NOT network-namespace isolation; the launcher additionally
blocks non-loopback Python socket activity. Configure OS egress controls separately
if untrusted subprocesses ever become necessary (none are exposed as model tools).

First run the image's `init /profile/new --base-url http://127.0.0.1:PORT/v1
--model YOUR_LOCAL_MODEL_ID` command with `/profile` owned by UID 10001. Then run
`run /profile/new` with the same mount. Both are explicit operator actions, never
automatic on plugin installation. The launcher checks the exact memory-only tool
surface before each turn and refuses unexpected tools. No shell/file/code/MCP
tools are enabled. Keep approved plugin source immutable in the image/profile.

This confines the process from unrelated host files; it does NOT put memory JSON
outside the agent process's OS permissions. A separate memory service outside the
work-tool sandbox is still needed before enabling arbitrary work tools. Do not
describe this recipe as a complete multi-user or hostile-plugin security boundary.
