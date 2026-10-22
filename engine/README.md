# Xenolith engine components

Two explicit routes:

| Route | Implementation | Current status |
|---|---|---|
| Native Gemma4 E4B | Existing C/Level Zero code, Gemma-specific profile and Q4_0 graph | Unvalidated; default gate retained |
| Bonsai 2 27B PTQ1 Lean | `launch.py` → owner-installed Linux PrismML llama.cpp → local `/v1` endpoint | Launcher/profiles implemented; target runtime untested |

Bonsai is not passed through the Gemma loader/template, and native snapshot/wire capabilities are not advertised for the process backend. Its PTQ1 graph, transformations and MTP are provided by Prism; there is no second full draft model in the launch command. See [backend status/runbook](docs/BONSAI_BACKEND_STATUS.md).

`python3 launch.py --profile profiles/bonsai27-linux-cpu.json --engine /opt/prism/bin/llama-server --model /models/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` prints a JSON plan only. Launch requires an explicit `--start` on the compatible Linux target; Windows refuses before opening engine/model files. No installer or startup hook runs here.

`make check-backends` uses Python-only no-model contract tests. `make` still builds only the native E4B engine and requires compatible Linux GCC, Level Zero and Intel ocloc. No C build, engine/model test or GPU workload was run in this Windows revision.

The [active plan](docs/Local_Bonsai27_Hermes_Xenolith_Memory_Plan_v5.md), [checklist](docs/PROJECT_FUNCTION_CHECKLIST.md) and [E4B source status](docs/E4B_PORT_STATUS.md) distinguish present source, historical reference checks and pending target validation. Vision, native Bonsai kernels, full Hermes integration and physical KV eviction remain unfinished. Original licensing remains in [LICENSE.md](LICENSE.md).

## Intel CPU optimization increment — 22 October 2026

This increment supersedes the earlier blanket no-prefix-reuse configuration **only for the opt-in CPU candidate**. PTQ1 decoder/packing/Hadamard/matrix kernels remain deferred at the user's request. The original CPU MTP-off/prefix-off profile and live Hermes settings remain available and inactive here.

Implemented source: exact-prefix-friendly controlled-state serialization with full source/evidence/correction preservation, local HTTP keep-alive without request replay, separate prompt/decode/draft CPU threads, optional affinity masks and polling, complete-verification-microbatch guards, a CPU-only candidate profile and a dry-run24-plan tuning renderer. The cumulative cpu-mtp-v1 external-runtime patch adds buffer reuse, all-accepted hidden-copy elision, early enforcement of per-sequence draft limits and optional per-request timing-based MTP window selection (1..min(4,cap), four samples/window, a probe every32 valid cycles). It includes greedy-mtp; apply to the clean pinned source, never stack both patches.

Candidate: Linux CPU, device none and target/draft GPU layers0, one slot,8192 context, batch256/ubatch128, six prompt/decode threads, polling0, p_min0, adaptive cap3. Prefix cache is requested with128MiB prompt-cache budget, two additional context checkpoints and idle-slot archiving off. Live KV/checkpoints are outside that budget; >=5GiB MemAvailable safeguard remains. CPU backend sampling is independently selectable, not GPU placement. No CUDA switch is emitted. No affinity topology or fastest thread/batch choice is presumed.

Hermes owner CLI options are `--reuse-prefix --keep-alive`; both are opt-in. They do not activate ordinary Hermes or store/reinject prior reasoning. Prefix eligibility is exact rendered-token identity; hybrid state restoration/recomputation, changed user instructions and removed-reasoning tails need target parity acceptance. No persistent KV cache or automatic inference/tool retry is added. SSD pending-action writes/readback remain mandatory. Existing controlled-run reasoning-off/output1024 defaults are retained; the launcher's low/4096 request defaults are a separate production-provider configuration decision.

The actual C++ policy compiled and passed standalone tests. Python launcher/harness contracts, patch application/reverse/hash checks and dry-run tuning rendering were checked without a model. The full runtime remains uncompiled and unbenchmarked. Adaptive warmup can cost time on short answers and does not compare MTP-off internally. Compare whole-task latency, token/action parity, cache hits, recurrent replay, RAM/swap and sustained thermals before adoption. Do not remove synchronization or recurrent checkpoints on an assumption that CPU execution makes them unnecessary.

Implementation and target procedure: `engine/patches/prism-cpu-mtp/README.md` in Xeno; cumulative patch pinned to sudoingX ff414120c343e6e6cb868013c99f1dde52b27e70. Target compilation/validation and runtime activation remain pending. No model, engine service, GPU or WSL was loaded here; only the standalone policy test binary ran.

