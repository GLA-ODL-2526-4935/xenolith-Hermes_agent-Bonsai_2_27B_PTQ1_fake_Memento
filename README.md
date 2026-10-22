# Xenolith + Hermes Agent + Bonsai 2 27B — experimental “fake Memento”

A local research-agent experiment: a modified Hermes Agent, text memory, bounded
working context and an optional Intel-focused inference setup. The aim is to keep
useful task state between steps **without training the model again**.

**Experimental, Linux + compatible Intel integrated-GPU PC target only.** This is
not a finished installer or a validated end-to-end release. The intended machine
has 16GB shared RAM; fitting the model file is not proof that inference, context
and desktop workloads fit. Windows/macOS, other GPUs and arbitrary Intel devices
are not supported deployment targets for this combined setup. Individual upstream
projects and a user-supplied engine may support other hardware.

## What differs from ordinary Hermes?

This includes the upstream [Hermes Agent](https://github.com/NousResearch/hermes-agent)
inside `Hermes_agent/`, plus experimental local extensions:

- Single-user, text-only temporary and persistent L0–L3 memory; staged retrieval
  and controller/owner-gated additions. The model cannot edit existing records.
- Source-linked task/step state and controller-managed request rebuilding, rather
  than requiring a successful Memento fine-tune.
- Section-first tool discovery: names/descriptions first, one selected format next.
- Research-only validator switch and an offline browser foundation that reads
  bounded page sections rather than dumping entire websites into context.
- Separate Linux CPU/Vulkan runtime profiles and guarded launch planning.

Current intended permissions are **web research + memory, not coding, shell,
arbitrary files or desktop control**. Ordinary Hermes has broader capabilities;
copying this repository does not automatically restrict its normal launch mode.
The dedicated launcher is still memory-only. Live Brave Origin browsing,
filesystem/network isolation and full CLI/gateway integration are unfinished.
Do not enable generic Hermes browser/work tools as a shortcut.

### Engine and model: an important distinction

The native [Xenolith engine](https://github.com/simoneiacomino/xenolith) targets
Intel Xe integrated GPUs using Linux Level Zero. This fork preserves a gated,
unvalidated Gemma E4B native port; it **does not natively run Bonsai PTQ1**.
Bonsai currently uses the separate [PrismML llama.cpp runtime](https://github.com/PrismML-Eng/llama.cpp)
through `engine/launch.py`. Do not feed Bonsai to the native Gemma loader.

The proposed model is [Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
specifically the community [PTQ1_0 MTP Lean variant](https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF).
It is optional; you can supply your own model and compatible local engine instead,
skipping the Bonsai download. This requires separate compatibility checks; the
bundled Bonsai launcher deliberately checks one exact model hash.

“Fake Memento” means **inspired by**, not an implementation of Microsoft's
[MEMENTO: Teaching LLMs to Manage Their Own Context](https://arxiv.org/abs/2604.09852)
([research code](https://github.com/microsoft/memento)). That paper trains
block/summary behaviour and changes inference attention/cache handling. Here the
controller preserves explicit task state and rebuilds requests without retraining
the base model. We do not reproduce the paper's implicit KV information channel,
physical KV eviction or reported performance gains.

## Install / prepare on the target Linux PC

There is no one-command installation of the complete system yet. Read
[Agent.md](Agent.md) first; check hardware, then choose engine and model **before
any downloads**. Keep the modified `Hermes_agent/` folder: a fresh upstream Hermes
installation will not include these changes.

1. Obtain this complete project directory with both `engine/` and `Hermes_agent/`.
   The local Hermes checkout has its own Git metadata; it is not yet a packaged
   submodule/release. Check that a clone/archive actually includes the modified
   Hermes source and extensions before proceeding.
2. On Linux, check OS/architecture, CPU, Intel GPU PCI ID, drivers, RAM and disk.
   Native Xenolith needs a supported device and Level Zero/Intel build tools;
   Prism Vulkan needs a working Vulkan device. CPU mode is a separate fallback,
   not proof that native Xenolith works. Follow [backend setup](engine/docs/BONSAI_BACKEND_STATUS.md).
3. With approval, prepare the bundled Hermes environment using its pinned package
   manager, in a dedicated owner-selected `HERMES_HOME` and `HERMES_RUNTIME_DIR`:

   ```sh
   cd Hermes_agent
   bash ./setup-hermes.sh --runtime-only
   ```

   This installs dependencies and may download managed tools. Review its effects
   first; do not run a remote installer, enable all toolsets, start ordinary
   `hermes`, or overwrite an existing personal profile. See the nested
   [development instructions](Hermes_agent/AGENTS.md).
4. If selecting Bonsai, separately obtain a pinned compatible **Linux Prism**
   runtime and `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` from the sources above.
   Keep their licenses and verify hashes. No CUDA/Windows binary or stock llama.cpp
   substitution. Preview from `engine/` without loading a model:

   ```sh
   python3 launch.py --profile profiles/bonsai27-linux-cpu.json \
     --engine /ABS/PATH/TO/llama-server \
     --model /ABS/PATH/TO/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf
   ```

   Only after target validation and explicit approval does adding `--start` load
   the model. The native `make` target builds Gemma code, **not** this Bonsai runtime.
   If choosing your own engine/model, skip this download/launcher route and verify
   the local API/template/tool-call contract independently.
5. For the first isolated **memory-only** test, create a new profile from
   `Hermes_agent/`, substituting your actual local endpoint and served model ID:

   ```sh
   bash scripts/run-in-hermes-env python -m local_extensions.e4b_memory.restricted \
     init /ABS/NEW/MEMORY_PROFILE --base-url http://127.0.0.1:18780/v1 --model YOUR_MODEL_ID
   ```

   See [memory setup](Hermes_agent/local_extensions/e4b_memory/README.md) for the
   explicit `run` step and owner approval workflow. Keep the research validator on.
   Live research browsing must wait for the
   [Linux isolation gates](Hermes_agent/local_extensions/e4b_memory/deploy/RESEARCH_ONLY.md)
   and [browser connection/context-budget integration](Hermes_agent/local_extensions/e4b_browser/README.md).

Reference Bonsai artifact: 6,297,658,848 bytes; SHA256
`1e33c571a5ce7a9a3e42474d66192923d5a6d77da7fb3a22986dc809522b5685`.
The launcher requires a 5GiB available-RAM reserve before and during execution;
that is a stop guard, **not** a total memory requirement or fit guarantee.

## Ask an AI to help install it

Open this complete folder in a coding assistant such as OpenAI Codex and ask:

> Read the top-level Agent.md. Check this PC's compatibility without installing
> or loading anything first. Explain blockers, then ask me whether to skip or
> retain Xenolith and whether to use Bonsai 2 27B PTQ1 or my own model. Show the
> installation plan and download sizes before proceeding. Do not start inference
> or enable browser/system access without my approval.

The top-level `AGENTS.md` points compatible assistants to `Agent.md`;
[Codex documents its AGENTS.md discovery here](https://developers.openai.com/codex/guides/agents-md/).
Other assistants may need you to attach/read `Agent.md` explicitly. A chat without
computer access can explain steps but cannot install software for you. Never send
it passwords, memory files or API secrets.

## Status, credits and licensing

Use the [functional checklist](Hermes_agent/PROJECT_FUNCTION_CHECKLIST.md) and
[active v5 plan](Hermes_agent/Local_Bonsai27_Hermes_Xenolith_Memory_Plan_v5.md)
for current memory/security/browser status; engine-specific notes are in
[`engine/README.md`](engine/README.md). Offline tests are not Intel/Linux runtime
certification. No production-readiness or speed guarantee is made.

Developed with assistance from **OpenAI Codex (ChatGPT)** for implementation,
offline checks and documentation. This is an independent experiment, not an
official OpenAI, Nous Research, PrismML, Microsoft or Xenolith release/endorsement;
AI assistance does not replace human review.

New project-specific code and documentation are **MIT** under [LICENSE](LICENSE).
Upstream code retains its original licenses/copyright notices: Xenolith and
Hermes are MIT; Bonsai model weights are Apache-2.0, **not relicensed as MIT**.
Runtime, dependencies and optional model licenses remain separate. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing a bundle.


## Intel decode research plan — 22 October 2026

See the [engine structure roadmap](engine/docs/ENGINE_DECODE_STRUCTURE_ROADMAP.md),
[engine checklist section 4A](PROJECT_FUNCTION_CHECKLIST.md#4a-bonsai-prism-paper-derived-decode-structures)
and [twelve paper references with attribution](engine/README.md#decode-research-roadmap-and-paper-references).
Priority is BITCOS feasibility, bitnet.cpp exact lookup and Sandwich phase scheduling;
these are design gates, not measured speed gains or newly activated runtime code.


The three priority papers now have an [inactive source package](engine/patches/prism-intel-decode/README.md)
and implementation status in checklist 4B. BITCOS remains a conditional codec/kernel
experiment; full target runtime/parity/performance validation is pending for all paths.


Additional engine source work is in the [optional v3 increment](engine/patches/prism-intel-decode-increment/README.md)
and [additional paper review](engine/docs/ADDITIONAL_INFERENCE_REVIEW.md): exact
LUT local-work reductions and owned MTP hidden-row copy reuse, both inactive and
reversible. Native target parity/DRAM/speed validation is still pending; this does
not change live profiles or the Intel Vulkan architecture.
