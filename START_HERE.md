# Combined start — prepare now, run later on the Linux target

The launcher coordinates **existing Prism engine/model loading → readiness check
→ restricted Hermes**, with Hermes already pointing to that engine's loopback
endpoint. This is source prepared for later testing, not an installer or a claim
that the project currently runs. It never downloads weights, builds a runtime,
installs drivers/packages or connects a chat app.

## Fill these later

Copy `launch.example.json` to `launch.local.json` and edit the latter on the
reviewed Linux x86-64 target. Its machine-specific paths are deliberately blank:

- `python_executable`: absolute Python path in the prepared Hermes environment.
- `prism_executable`: absolute path to the compatible owner-installed Linux
  Prism `llama-server`, not native Xenolith's Gemma executable.
- `prism_sha256`: verified SHA256 of that executable; provenance and matching
  shared libraries must also be reviewed separately.
- `model_path`: absolute path to the pinned
  `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf`. Leave blank until the model is added.
- `hermes_profile`: absolute path to a dedicated new profile, with an existing
  parent directory. Do not use a personal/default Hermes profile or source folder.

The default `engine_profile` is the existing CPU/MTP-off baseline. Its existing
5 GiB available-RAM reserve is **not a target fit guarantee** and still needs
target review. Do not enable an optimization patch or Vulkan profile without the
separate runtime/device/parity checks. No GPU is automatically selected by the
CPU baseline. The project's intended Intel/Linux target and OS containment gates
remain unchanged; native architecture screening alone is not certification.

After checking hardware, runtime/model provenance, available RAM and intended
permissions, set `target_hardware_reviewed`, `runtime_reviewed` and
`execution_enabled` to true. These are owner attestations, not automated
certificates. Leave them false now. `automatic_memory_workflow` defaults false;
true enables generation/conflict checks, **not automatic acceptance**.

## Preview and create the one-click entry

Use the prepared Hermes Python from the project root:

```sh
/absolute/hermes/python start.py --config launch.local.json
```

Without `--start` this only reads configuration and prints a plan. It does not
open model/runtime files, create a profile, contact an endpoint or start children.
Missing paths/unchecked gates are listed as blockers. Even a complete plan is not
a compatibility certificate.

On the target, create a new KDE launcher file using that environment:

```sh
/absolute/hermes/python start.py --config launch.local.json --desktop "Start Project.desktop"
```

Review the generated file and mark it trusted/executable using KDE's normal
controls. Clicking it explicitly requests startup in a terminal. No desktop
entry, executable bit, autostart registration or launch is installed automatically
by this code change. Alternatively use `start.py --config launch.local.json --start`.
The desktop entry contains absolute paths; recreate it if the project is moved.

## Startup / stop behavior

Explicit start refuses unsupported native OS/architecture, WSL/container,
unfilled settings, unreviewed gates, occupied endpoints and runtime hash mismatch.
The existing backend independently checks Linux, executable ELF, pinned model
GGUF/SHA256 and its RAM reserve before/during engine execution. It does not attach
to or kill an existing server, use a hosted fallback or retry requests.

A fresh dedicated profile is generated with the fixed `bonsai-2-27b` served ID
and the engine profile's endpoint (baseline `http://127.0.0.1:18780/v1`). On later
starts its configuration and copied plugin must still match; changed/unrelated
profiles are refused, not overwritten. Only the automatic-memory switch may be
updated from this launch config. Back up and explicitly refresh a copied plugin
after source upgrades; old profiles are not silently migrated.

Hermes starts only after the owned engine returns the expected model ID within
the configured startup timeout. Exiting Hermes (`/quit`), startup failure, loss
of the engine or Ctrl+C stops only the launcher's owned process groups. Owner
Discord/API tokens and inherited hosted-provider settings are not passed to these
children. Readiness does not prove model-answer/tool-call correctness. Cancellation
of a running model request is not newly implemented.

**Scope:** this launches the existing memory-only CLI prototype. Live Brave
research, Discord/Element/X chat, OS isolation, checkpoint API v2 and target
performance are not activated. A standalone owner approval bot remains separately
started with its separate privileges/credentials. No model/GPU/engine was run on
the Windows preparation laptop. Real startup/readiness/shutdown, copied-profile
upgrades, model correctness and Linux permission boundaries still need target tests.

Memory review instructions: [REVIEW_WORKFLOW.md](Hermes_agent/local_extensions/e4b_memory/REVIEW_WORKFLOW.md).
