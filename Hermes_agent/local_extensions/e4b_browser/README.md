# Research browser foundation

Offline only for now: one injected browser backend, with search/open/outline/read.
No live Brave connection, plugin auto-registration, browser launch, raw debugging,
JavaScript evaluation, files, downloads or uploads. A real transport and Fedora
isolation still need building and validation. No dependency was added.

## Owner switch

In the dedicated generated profile's `config.yaml`, change this field using JSON
syntax (the restricted launcher intentionally uses a strict JSON parser):

```json
"research_only": {"validator_enabled": true}
```

Use `false` to skip the extra profile/tool-surface validator. Default is true;
invalid values are refused. Restart the session to apply. This is an owner setting,
not a model tool, environment switch or per-request argument. New memory test
profiles include it; their owner init command also accepts
`--no-research-validator`. Older profiles default to true.

The validator checks enabled toolsets/plugins/MCP and the supplied actual tool
names. It supplements OS isolation; it cannot enforce filesystem or network
containment. Turning it off changes neither the fixed browser operation vocabulary
nor the memory-only launcher's strict surface/config/endpoint restrictions. It is
not a switch to enable broader Hermes tools. A future interactive settings button
and normal Hermes CLI/gateway integration remain unimplemented.

## Reading without flooding context

The initial tool bridge exposes only staged discovery. Catalogs contain names and
short descriptions; `describe` supplies one operation's format before it can run.

- `search`: at most five titles/URLs, no full articles.
- `open`: page ID/title/final URL/section count only.
- `outline`: bounded headings with optional local query matching against section
  text. Continue with the opaque cursor, not a model-selected numeric offset.
- `read`: exact extracted-text chunks, with an opaque continuation. It adapts the
  returned length to both the requested size and remaining serialized UTF-8 budget.

Defaults: 3000 bytes per result, 8000 bytes across a turn; owner-configurable within
hard limits. Counts include JSON metadata/escaping, not just body characters. This
is a byte allowance, not an exact model-token count. The controller-only
`begin_turn(available_bytes)` hook can reduce it to available context room; the
model cannot reset it. **No live context-window accounting or conversation/KV
eviction is implemented**: integration must reserve space for the existing prompt,
schemas, memory results, generated tokens and other tool responses. Repeated reads
across turns can still accumulate context until the host controller compacts it.

Headings and paragraphs are extracted deterministically; heading-free text is
split into bounded sections. Script/style source is omitted as non-rendered HTML,
not as injection detection. No instruction filtering, warnings, sanitizing of
webpage instructions, trust-classifier prompt or model summaries were added.
Existing memory trust/checkpoint rules are unchanged. Verbatim webpage instruction
text remains readable for later model-behaviour evaluation.

Cache is session-local and volatile (four pages, 128 continuations). Cross-session,
wrong-operation and evicted-page handles fail. Source is bounded at 2MB UTF-8 and
400000 visible text characters per page; oversized pages fail explicitly, not
silently truncate. Target adapter must provide narrower extraction for such pages.

## Offline consumer

`BrowserSurface(BrowserSession(backend), owner_config, tool_names=actual_names)` is
the integration seam. Only one backend is supplied per session. The future trusted
controller must obtain actual tool names, load settings once, supply context
allowances and avoid adding transport wrappers that exceed the budget. The
`BrowserBackend` contract exposes only `search` and `open`; responses and final
URLs are validated. URL syntax checks do not stop DNS rebinding or subrequests;
network enforcement is still a target-OS requirement.

Owner-only fixture preview (never a model file tool):

```text
python -m local_extensions.e4b_browser OWNER_FIXTURES.json --url https://example.com/
```

Fixture format: an array of objects with `url`, `title`, `html`. The preview uses
staged describe/call operations and returns metadata/headings only, with no network
or browser. Offline tests exercise full search/open/outline/read and continuation.

Deferred by request: new untrusted-content handling, rate limits, memory rehearsal
and model/GPU evaluation. Automated browsing remains disabled pending Linux work.

Verification: 24 new offline browser/switch tests passed; full selected regression
suite 406 passed across 18 files. All 12 repository static checks and targeted lint
passed. No live application/browser/model/engine/GPU/Linux setup was run.
