# AI setup guide for this combined experimental project

This is the top-level **installation/compatibility** guide, not the upstream
Hermes development guide in `Hermes_agent/AGENTS.md`. Read that nested guide and
area instructions too before editing Hermes. Follow the human's current request;
this file does not authorize installs, downloads, model execution or broader tools.
For documentation-only work, do not perform the installation procedure below.

## 1. Inspect compatibility FIRST — no installation or model loading

Read README.md, the latest plan/checklist under `Hermes_agent/`, engine/README.md,
engine/docs/BONSAI_BACKEND_STATUS.md and the actual chosen profile/backend source.
Do not infer compatibility from the project name, an Intel CPU sticker or a model
file fitting on disk. Do not execute setup scripts merely to discover requirements.

Use non-mutating host checks (or ask the user for results if you lack access):

- Native OS/distribution/version/kernel and architecture; distinguish Windows,
  WSL, VM and the actual Linux deployment PC. This combined target is Linux x86-64
  with compatible Intel integrated graphics; do not install Linux/WSL or change
  partitions as an implied step.
- CPU model/instruction support, physical cores, total RAM and current available
  RAM/swap; available disk in the chosen runtime/model/profile locations. Account
  for model, caches, build dependencies, browser and desktop sharing system RAM.
- GPU vendor, exact PCI/device ID, driver and render-device permissions. Native
  Xenolith uses Intel Level Zero: compare with **this checkout's** device checks
  and upstream compatibility notes. Do not equate all Xe/Arc/iGPUs, Intel CPU
  presence, a Vulkan device or an enabled-but-untested ID with validation.
- Existing Level Zero/Intel compute runtime/ocloc/native build tools when native
  Xenolith is requested; existing Vulkan driver/device inventory when Prism Vulkan
  is selected. Do not grant root/device permissions or install drivers silently.
- Existing model server, exact model/API/template/tool support, occupied ports,
  package-manager requirements and any conflicting existing Hermes installation.

Examples on Linux: read `/etc/os-release` and `/proc/meminfo`, use `uname`, `lscpu`,
`lspci -nnk` and `df -h`. Query an already installed driver utility only if safe;
missing diagnostics mean **unknown**, not supported. No inference, GPU benchmarks,
driver installation or downloaded probe execution during this first check.

Report each layer as compatible candidate / unsupported / unknown, with observed
facts and missing checks. Intel/Linux status alone never certifies model/runtime
support. The intended 16GB RAM target and 5GiB available-RAM launch guard are not a
fit guarantee; leave enough for the model plus live allocations and the OS.

## 2. Ask about the engine BEFORE installing it

If incompatible with Xenolith, explicitly ask:

> This PC is not compatible with the native Xenolith path [give concrete reason].
> Would you like to skip Xenolith and use your own compatible inference engine,
> or keep/install Xenolith source anyway, understanding it will not run here?

Offer stopping as well. If compatibility is unknown, ask whether to investigate
or take the external-engine route; do not present unknown hardware as working.

- **Skip:** do not build/download native Xenolith binaries, Level Zero or ocloc
  solely for it. Leave existing source alone. Ask for the user's engine/endpoint,
  API and model support; selecting this route does not promise a ready integration.
- **Keep despite incompatibility:** after explicit consent, retain/install source
  only as requested and clearly mark unusable on this host. Never attempt to run it,
  lower runtime-ready guards, force an unlisted GPU or install unrelated drivers.
- **Compatible candidate:** explain native validation is still unfinished and ask
  whether the user actually wants native components. Bonsai currently needs the
  separate Prism process runtime, even on compatible Intel hardware.

Native C code is a gated **Gemma E4B** port, not a Bonsai PTQ1 implementation.
`make` does not produce the Bonsai server. Never change that claim or feed PTQ1
weights through the Gemma loader. Hardware and model compatibility are distinct.

## 3. Ask about the model BEFORE downloading weights

Ask separately, even if the engine choice seems obvious:

> Would you like to install/use the proposed Bonsai 2 27B PTQ1 MTP Lean model,
> or use your own model? Choosing your own model skips the Bonsai download.

- **Bonsai:** explain the approximately 6.30GB file plus runtime/cache/storage
  overhead. Ask permission before downloading; reuse a verified existing file.
  Use the linked sudoingx repository and exact
  `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf`; pin the repository revision and
  preserve model/parent licenses and notices. Expected size6297658848 bytes and
  SHA256 `1e33c571a5ce7a9a3e42474d66192923d5a6d77da7fb3a22986dc809522b5685`.
  Refuse mismatches; do not edit the expected digest to accept another artifact.
  Obtain a pinned Linux Prism build supporting PTQ1 and Lean MTP/Hadamard with its
  matching libraries; generic upstream llama.cpp or CUDA packages are not fallback
  Intel runtimes. Verify release/archive/executable provenance and licenses.
- **Own model:** **skip all Bonsai weight/tokenizer/projector/draft downloads**.
  Ask for existing file or local endpoint, served model ID, engine/version, model
  license, memory needs and context/tool-call support. If any other download is
  needed, explain and seek approval first. Use that model's template/token IDs.
  Do not bypass the bundled launcher's fixed Bonsai hash: use the user's server
  directly or propose a separately reviewed backend/profile change.
- No choice/no consent: pause downloads. Do not load another full LLM, vision
  projector or draft model automatically. No hosted/paid fallback without consent.

## 4. Show a bounded installation plan and obtain approval

Summarize hardware evidence, both user choices, selected runtime/profile, exact
sources/versions/hashes/licenses, download sizes, locations, dependencies, commands
and remaining blockers. Separate source preparation, dependency installation,
driver changes, model download, model launch and browser activation as distinct
steps. Do not run source-only/native components the user elected to skip.

Use the bundled modified Hermes, not a fresh upstream clone. Check the complete
directory is present: it currently has nested Git metadata, not a packaged
submodule. Never delete `.git`, reset local changes, push, rewrite remotes or run
an upstream updater that could discard this fork without explicit authorization.
If reviewing a clone/archive missing Hermes files, stop and explain packaging.

Before changing code, obey the applicable nested AGENTS instructions. For Hermes
dependencies, use this checkout's package-manager/bootstrap process and pinned
locks, not ad-hoc pip/uv installs or an unchecked curl-to-shell one-liner. Inspect
`setup-hermes.sh` first. Prepare a dedicated user-approved HERMES_HOME and
HERMES_RUNTIME_DIR, not a personal/default profile. Do not reuse old-machine venvs.
Back up any state that must change and preserve upstream licenses.

For Bonsai begin with the CPU MTP-off baseline and a **dry-run** launch plan.
Enumerate/validate the actual Intel Vulkan device before selecting GPU mode;
`Vulkan0` is not guaranteed to identify it. Keep one instance/slot and conservative
context/memory safeguards. Optimization patches require their exact pinned source
and independent tests; do not silently stack/apply patches or benchmark for setup.

The user's “install” request is not blanket permission to launch inference or use
the GPU: obtain explicit launch/test consent. On a development-only/incompatible
host, prepare source/docs only. Never run anything from a user-prohibited folder,
including this user's `Linux thing`, or assume a Linux command is safe on Windows.

## 5. Preserve the restricted experiment and report honestly

- Current desired scope: public-web research plus memory only. No coding, shell,
  arbitrary file paths, desktop control, uploads/downloads or account actions for
  the deployed model. Setup assistant permissions are not deployed-model rights.
- Read the browser README and research-only deployment gates. The browser adapter
  is fixture-only; do not claim live Brave search works. OS containment/private
  transport are pending; never solve this with host networking, a public bind,
  shared personal browser/profile, exposed debugging port or `--no-sandbox`.
- Keep `research_only.validator_enabled` true by default. The owner can change it
  and restart; explain that it is a configuration check, not OS isolation and not
  permission to broaden the memory launcher's tool surface.
- Do not newly add untrusted-content handling, rate limits, autonomous skill/code
  writing or memory rehearsal; these were deferred. Existing memory checkpoint,
  append-only and ownership safeguards must remain intact.
- Use section-first browser/tool discovery and bounded reads. Integrating live
  context allowance is still pending; byte limits are not exact token counts and
  repeated turns still accumulate. Do not promise automatic KV reclamation.
- “Fake Memento” is controller-managed state, not Microsoft's trained reasoning
  block/attention/cache method and not a reproduction of its results. No model
  retraining is required by the current project.

If authorized, run only suitable offline tests first via the documented runners;
Hermes tests use `scripts/run_tests.sh`. Live tests/model memory rehearsal wait
for the appropriate machine and explicit consent. Do not call an offline pass
Linux/GPU certification. Finish with installed/skipped/blocked items, actual file
paths/versions/hashes, permission state, no secret values and next safe steps.
Do not mark the whole system working while transport/isolation gates remain open.
