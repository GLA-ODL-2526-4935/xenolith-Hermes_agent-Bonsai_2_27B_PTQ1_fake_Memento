# Portable Hermes draft-source learning (source candidate, 23 October 2026)

This package adds persistent cost observations to `intel-source-selection-v1`.
It selects warm-off, MTP windows 1..4 or bounded suffix-chain windows 1..4.
It learns from committed tokens / elapsed cycle microseconds, including drafting,
verification and replay costs. This is online controller learning, not model/head
fine-tuning. Target sampling stays greedy and target verification remains in charge.

The source is applied in the separate `prism-cost-aware-source` checkout beside
the installed runtimes, pinned to sudoingX/llama.cpp
`ff414120c343e6e6cb868013c99f1dde52b27e70`. This is the existing Prism-derived
Lean-compatible branch, not a claim to have rebased onto today's PrismML HEAD.
No installed binary was replaced and no runtime/profile was activated.

## Paper attribution and scope

- [BanditSpec](https://arxiv.org/abs/2505.15141): online selection of speculative
  configurations. Our deterministic bounded-probe, rate-based adaptation is not
  the paper's UCBSpec/EXP3Spec algorithm and inherits no stated regret guarantee.
- [SuffixDecoding](https://arxiv.org/abs/2411.04975): exploit repeated agent output.
  We retain the existing bounded prompt-local chain scan, not its persistent
  suffix tree or long/tree drafting. No prompt history is exported.
- [Hermes Agent](https://github.com/NousResearch/hermes-agent): workflow context.
  The new questions are authored synthetic fixtures based on our local controlled
  plan / one-tool / evidence-answer workflow, not an official Hermes benchmark.

Earlier PTQ1/QIGen and twelve-paper credits remain in the engine README and
the prerequisite packages. This change adds no new quantization/kernel algorithm.

## What persists

Nine arms each store integer `samples`, `tokens`, `elapsed_us`, grouped into 64
4096-token context buckets. Completed cycle observations survive new requests.
Snapshot writes happen at request release and final destruction if dirty, never
per generated token. A stopped/cancelled incomplete cycle is censored; previously
completed cycles remain eligible to save. Profiles contain no prompts, actual
token IDs, reasoning, model parameters, hidden states or KV caches.

The target verifier, MTP acceptance/replay hooks, KV semantics and Intel Vulkan
prefill architecture remain the prerequisite paths. Warm-off keeps the MTP head
warm and therefore is not a true no-head/no-MTP baseline. Measure both separately.
The policy requires one slot, MTP-capable model, temperature 0, cap <=4; it replaces
the old adaptive window tuner while enabled. It uses two off observations, a
minimum two samples per candidate, a >5% measured advantage and bounded probes.
Short tasks can remain off and sparse context buckets may learn slowly. There is
no automatic task classifier or accuracy reward. Long-lived counters reset one
bucket if its time exceeds 5e11 us or samples exceed 5e10, before integer limits.

## Portable profile contract

`hermes-source-profile-v2` contains exact `identity` and numeric `buckets`.
Identity binds algorithm, combined GGUF weights/tokenizer/head artifact digests,
engine binary, hardware/OS, relevant settings/environment and workload file.
The three combined-GGUF hashes intentionally repeat the container SHA256; they
are not independent tensor hashes. A changed head requires a new identity.

On the same machine with the identical binary/settings/model/workload, Prism and
Xenolith's external Prism adapter consume exactly the same JSON. Copying counters
does not retrain or merge them. Native Xenolith's Gemma C graph cannot use this
selector; no native Bonsai graph or new graph ABI is claimed.

Windows/NVIDIA costs must not become Intel CPU/iGPU priors. Generate a fresh
identity on the destination, then use `retarget` to clear foreign costs and
calibrate there. This deliberately sacrifices timing reuse across incompatible
hardware. Same schema/algorithm transfers; incompatible throughput numbers do not.

Python checks actual runtime/hardware/settings before an adapter launch; the C++
loader checks the supplied expected identity against the imported profile. For a
manual server launch, generate/check identity locally before setting its env vars.
These are provenance guards, not attestation of a remotely running server. If
shared libraries, drivers, power policy or thermal conditions materially change,
recalibrate even when the recorded fingerprint does not capture that change.

Output is opt-in and protected by an exclusively created `.lock` directory.
Existing output is refused unless it is explicitly also the input being resumed.
Snapshots replace atomically from a temporary file inside the lock. This is not
a power-loss durability guarantee. Existing locks are never recovered/deleted by
another process; an owner must confirm a crashed writer is gone before recovery.
Malformed/duplicate JSON keys, oversized files, negative/float/bool/overflow
counters, inconsistent totals and mismatched identity are rejected. No automatic
merging or fallback to mismatched statistics. Export errors warn and leave dirty
counters in RAM for a later request; inspect logs and the actual file timestamp.

## Setup and transfer (instructions only; not executed here)

Apply source packages in order to the exact pinned checkout:
`prism-intel-decode` (v3), `prism-intel-decode-increment` (v1),
`prism-source-selection` (v1), then this package. Each `apply.py SOURCE` checks;
`--apply` changes source only. This package's `--reverse` removes only its delta;
reverse dependent deltas in reverse order. Never stack the older cumulative
PTQ1/CPU-MTP packages. Follow existing Intel build/validation instructions; retain
CPU decode / Vulkan prefill choices. QIGen remains an independent optional switch.

Xenolith's `engine/backends/prism.py` has default-off optional profile fields:

```json
{
  "source_policy": true,
  "suffix_drafting": true,
  "adaptive_mtp": false,
  "policy_frozen": false,
  "policy_identity": "/owner/identity.json",
  "policy_input": "",
  "policy_output": "/owner/learned.json"
}
```

These extend a complete existing launch profile; they are not a complete config.
Use MTP cap 4 and include `intel-source-profile-v2` in its runtime requirement.
No existing launch profile is enabled/rewritten. The pre-existing 5 GiB target
RAM-reserve restriction remains unresolved and is not a recommendation for the
16 GiB laptop; choose/fix the target reserve separately before live use.

Record a backend dry-run plan (on the actual target) as `plan.json`. It now
contains `selector_settings`, including a digest of performance env controls.
Paths to profile files are excluded from the performance settings digest.
Generate the expected identity without loading a model:

```sh
python source_profiles.py identity --runtime /path/to/llama-server \
  --settings plan.json --workload hermes-workflows.jsonl \
  --weights-sha256 1e33c571a5ce7a9a3e42474d66192923d5a6d77da7fb3a22986dc809522b5685 \
  --out /owner/identity.json
```

Use the adapter or identical manual server controls:
`LLAMA_SPEC_SOURCE_POLICY=1`, `LLAMA_SPEC_SUFFIX=1`,
`LLAMA_SPEC_IDENTITY=/owner/identity.json`,
`LLAMA_SPEC_PROFILE_OUT=/owner/learned.json`.
For resume set `PROFILE_IN` and `PROFILE_OUT` to the same file. For frozen
evaluation set `PROFILE_IN` only and `LLAMA_SPEC_PROFILE_FROZEN=1`; no output.
Frozen mode never explores or changes the persistent prior. Zero evidence stays
off. `LLAMA_SPEC_SOURCE_FORCE=0..8` remains a diagnostic override; don't use it in
adaptive calibration. The wrapper strips inherited selector/force/profile vars
before applying the explicit launch plan.

Same-environment transfer (exclusive new destination):

```sh
python source_profiles.py copy --profile learned.json --identity identity.json --out xeno-prior.json
```

Different hardware/build/settings, same combined model/head artifact:

```sh
python source_profiles.py retarget --profile learned.json --identity new-target-identity.json --out fresh-target.json
```

This writes empty counters under the new identity. `validate` accepts the same
`--profile` / `--identity` arguments without creating output.

## New questions and evaluation protocol

`hermes-workflows.jsonl`: 24 workflows x planning, tool-decision and final-answer
states =72 requests. First18 workflows/54 requests calibrate, remaining6/18 are
held out, split by workflow. Includes denied access, exact repeated failures,
injection in tool output, partial exports, contradictory/corrected facts, stale
memory, missing measurements, checkpoint durability and repetitive/nonrepetitive
controls. No actual shell/browser/export/memory tools are executed. Final stages
use fixed evidence regardless of the earlier generated plan/tool choice; this is
controlled replay, not a closed-loop success claim.

`hermes_replay.py` defaults to dry-run. Requires an explicit recorded server plan,
loopback endpoint and `--run` to send requests; never launches or kills servers.
Use the same complete workload file for identity in all phases. Reconfigure only
your owned server between phases; the runner does not flip server-global policy.

1. Record no-MTP baseline on both splits with `--mode baseline` (true spec-type
   none). Keep all other settings/model fixed.
2. Calibrate with `--mode learn --split calibration` and profile output enabled.
3. Verify exported JSON, then use a frozen-input server plan and
   `--mode frozen --split heldout --baseline baseline-heldout.jsonl`.
4. Also compare fixed MTP and warm-off controls separately before attributing gains.

Example dry-run (append `--run --out NEW.jsonl` only when inference is authorized):

```sh
python hermes_replay.py --plan plan.json --endpoint http://127.0.0.1:18780/v1/chat/completions --mode learn --split calibration
```

Greedy request defaults are reused exactly from the plan; low reasoning and output
budget are recorded. Oracles are excluded from model messages. Scores are format,
exact tool arguments and final-answer required-string screening, not semantic
proof; review contradictions, omissions and capped outputs manually. Preserve
raw answers, finish reason, usage, server timings, wall time and baseline agreement.
Do not use task accuracy as the cost learner's reward. Alternate runs/order and
compare sustained end-to-end latency and decode speed under the same thermals;
HTTP wall time includes prefill/queueing. Per-arm decode cost lives in the profile.

## Verification status

14 new Python contracts and 16 existing backend contracts pass. Full actual
`common/speculative.cpp` and `server-context.cpp` pass C++17 syntax compilation;
native profile tests compile. Exact applied source hashes and reverse applicability
checked. See `verification.json`. No native test executable, full runtime, model
or GPU was run. No learned timing profile or speed improvement is claimed yet.
Required target gates: native JSON/atomic-write contracts, all-accept/reject/replay
token/KV parity, cancellation/restart/export tests, true no-MTP comparisons and
held-out correctness/latency on the Intel laptop before activation.
