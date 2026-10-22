# Bonsai 27B backend status

22 October 2026. Active project model candidate: Bonsai 2 27B PTQ1 Lean with the existing controlled-step harness. This is an additional **Prism process backend**, not a native PTQ1/Qwen graph inside Xenolith. The E4B C graph, template fingerprint and default runtime gate are unchanged.

## Implemented here

- `launch.py` and `backends/prism.py`: explicit Linux-only foreground launcher, dry-run by default, model SHA256 check on explicit start, loopback-only serving, one slot, no hosted fallback, no implicit endpoint attachment, no external command shell.
- Separate CPU and Intel Vulkan profiles. CPU starts with MTP off; Vulkan is an unvalidated one-token MTP candidate. Both start at 8192 context, q4_0 K/V, six threads, batch256/ubatch128, reasoning effort low and4096 output limit. These are conservative candidate settings, not target benchmarks.
- Five GiB available RAM reserve before and during explicit launch. Stops only its own child. Signal handlers retain foreground lifecycle. Failure does not delete model, memory or cache files. A checkpoint of an interrupted answer is not claimed.
- Dry-run exposes endpoint, exact argv, environment, client request defaults, and capability limitations. `--start` refuses on Windows before model/engine file access. Model files are never read by dry-run.
- External runtime owns GGUF templates, PTQ1 decoding, Hadamard/sign transforms, hybrid recurrent/attention graph and optional MTP. It opens one combined Lean file, not a second full draft model. Runtime allocation/shared-weight behavior still requires target RSS measurements.

## Target install and use (do not run on this Windows development host)

Install a pinned **Linux PrismML llama.cpp** CPU or Vulkan package appropriate to the device, with its matching shared libraries; record release, archive SHA256, executable SHA256, driver and license inventory. The reference requirement is `prism-b10787-73bfe6c` or a pinned compatible newer build. Plain upstream llama.cpp is not a PTQ1 fallback. Do not copy Windows CUDA DLLs or use the NVIDIA-only optimized binary on Intel.

On the target, first inspect the runtime's version/device inventory without opening a model; install/verify Mesa Vulkan support if needed. Select the actual enumerated Intel device, which is not guaranteed to be `Vulkan0`. The CPU profile does not require a Vulkan GPU. Device names, native flags and packaged dependencies require acceptance on that exact build; no automatic runtime download/install occurs here.

Preview (only renders JSON):

```sh
python3 launch.py --profile profiles/bonsai27-linux-cpu.json \
  --engine /opt/prism/bin/llama-server \
  --model /models/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf
```

After all target preparation/validation gates, `--start` explicitly starts that process on Linux. It verifies the model digest, which reads the file but does not create an inference graph. On successful startup use `http://127.0.0.1:18780/v1`; copy **request_defaults** from the dry-run into the owning controller's local provider configuration. Server defaults alone cannot override a client still sending `enable_thinking:false`; current Hermes `controlled-run` adapter defaults to reasoning off, and needs an explicit separately reviewed adapter option before low-reasoning production use.

Profiles are not imported or activated automatically. The existing C `make` target still builds the Gemma-only native engine. `make check-backends` runs no-model Python contract tests. Neither command converts Gemma weights to Bonsai or activates Hermes.

## Optimization and acceptance order

1. Pin/validate exact Linux package, actual CPU/Vulkan device and model digest. Begin CPU, MTP off,8192 context, then repeat on Vulkan. Keep >=5GiB MemAvailable; record peak RSS/shared allocations/swap. Shared Intel RAM is not dedicated NVIDIA VRAM.
2. Reproduce logits/tokens and JSON actions against the same file in Prism. Verify low/off reasoning modes, no cap loops, cancellation and memory-stop behavior. No native Gemma token IDs/template/snapshots may be reused.
3. Benchmark CPU4/6/8 threads, batch128/256/512, ubatch64/128/256 and available cache types without changing sampling. Measure prefill/decode, end-to-end task latency and thermal steady state. Keep one model instance and one slot.
4. Compare MTP off and n-max1/2/3/4 with p-min0/0.3/0.5/0.7. Confirm the winner with repeats and output/evidence comparison; measure accepted/drafted, verified tokens per cycle, and time. CPU or iGPU may be faster with MTP disabled. A CUDA batch-invariance switch is not an Intel/Vulkan identity guarantee.
5. Sweep context8K/16K/32K only within measured memory limits; do not reserve262K merely because the model supports it. Keep vision unloaded until independently accepted. Add caching only after trace/retention and actual reuse/eviction validation; current harness deliberately rebuilds requests and disables prompt reuse.
6. Run historical six questions, direct workflows, real isolated edit/check/export/correction tasks, then longer unseen adaptive workflows and restart/uncertain-side-effect tests. Full Hermes/native tools and security isolation remain independent gates.

## Separate Windows evidence (not Intel/Linux certification)

The model SHA256 is `1e33c571a5ce7a9a3e42474d66192923d5a6d77da7fb3a22986dc809522b5685`,6297658848bytes. Windows RTX5080 optimized branch: eight reasoning-off scenarios manually correct; n-max3/p-min0 gave64.0decode tok/s versus50.5off (+26.6%), exact24/24 final-request matches; these are final-request replays, not full-loop speed measurements. Low-reasoning actual file workflows passed3/3,17requests, no cap/loop errors,61.9decode tok/s including reasoning. Reasoning text was discarded before the next step. Neither result proves Intel speed, Vulkan parity, physical KV reclamation, long-session reliability, or complete Hermes integration.

## Not implemented / not tested here

Native Bonsai PTQ1/Qwen35 C or Level Zero kernels; native MTP; native wire-protocol translation; native session/KV snapshots; physical block eviction; vision; CPU/iGPU runtime validation; complete live Hermes CLI/gateway deployment. Do not route Bonsai through `profile.c` or lower `runtime_ready` guards to make it load.

No model, engine binary, C build, GPU workload, WSL, Linux application or inference service was started by this repository change. Only Python parsing/contract tests and dry-run JSON rendering are allowed here.

Sources: https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md ; https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10787-73bfe6c ; https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF . Preserve original engine and external runtime licenses separately.

## Temperature-0 MTP source increment — 22 October 2026

The opt-in launcher now specifies matching greedy server/client defaults (temperature0, top-k1, top-p1, min-p0, repeat penalty1, seed3407, samplers top_k/temperature). CPU MTP-off and one-token Vulkan candidate defaults are retained. Client requests can override server defaults.

`engine/patches/prism-greedy-mtp/` contains a source-only patch pinned to sudoingX/llama.cpp ff414120c343e6e6cb868013c99f1dde52b27e70: direct argmax drafts at p_min0, no probability/sort/RNG work in that draft path, and conditional sampler cloning only when checkpoint/replay is possible. No recurrent/KV rollback or cache ownership rules are changed. Existing batched target verification already provides weight-reuse opportunities. The target top-k1 path already avoids softmax; vocabulary projection remains necessary.

Patch applicability, resulting source hashes, offline launcher contracts and the clone-bound arithmetic were checked. The patch has not been compiled, applied to an installed runtime, or benchmarked. It is not automatically compatible with the separately recommended Prism73bfe6c release. Apply/check only against the exact base; target CPU/Vulkan build, token parity, rejection/replay/cache-state and speed/memory acceptance remain required. No model or inference process was loaded. See `engine/patches/prism-greedy-mtp/README.md` in the Xeno repository for provenance and target acceptance.

## Intel CPU optimization increment — 22 October 2026

This increment supersedes the earlier blanket no-prefix-reuse configuration **only for the opt-in CPU candidate**. PTQ1 decoder/packing/Hadamard/matrix kernels remain deferred at the user's request. The original CPU MTP-off/prefix-off profile and live Hermes settings remain available and inactive here.

Implemented source: exact-prefix-friendly controlled-state serialization with full source/evidence/correction preservation, local HTTP keep-alive without request replay, separate prompt/decode/draft CPU threads, optional affinity masks and polling, complete-verification-microbatch guards, a CPU-only candidate profile and a dry-run24-plan tuning renderer. The cumulative cpu-mtp-v1 external-runtime patch adds buffer reuse, all-accepted hidden-copy elision, early enforcement of per-sequence draft limits and optional per-request timing-based MTP window selection (1..min(4,cap), four samples/window, a probe every32 valid cycles). It includes greedy-mtp; apply to the clean pinned source, never stack both patches.

Candidate: Linux CPU, device none and target/draft GPU layers0, one slot,8192 context, batch256/ubatch128, six prompt/decode threads, polling0, p_min0, adaptive cap3. Prefix cache is requested with128MiB prompt-cache budget, two additional context checkpoints and idle-slot archiving off. Live KV/checkpoints are outside that budget; >=5GiB MemAvailable safeguard remains. CPU backend sampling is independently selectable, not GPU placement. No CUDA switch is emitted. No affinity topology or fastest thread/batch choice is presumed.

Hermes owner CLI options are `--reuse-prefix --keep-alive`; both are opt-in. They do not activate ordinary Hermes or store/reinject prior reasoning. Prefix eligibility is exact rendered-token identity; hybrid state restoration/recomputation, changed user instructions and removed-reasoning tails need target parity acceptance. No persistent KV cache or automatic inference/tool retry is added. SSD pending-action writes/readback remain mandatory. Existing controlled-run reasoning-off/output1024 defaults are retained; the launcher's low/4096 request defaults are a separate production-provider configuration decision.

The actual C++ policy compiled and passed standalone tests. Python launcher/harness contracts, patch application/reverse/hash checks and dry-run tuning rendering were checked without a model. The full runtime remains uncompiled and unbenchmarked. Adaptive warmup can cost time on short answers and does not compare MTP-off internally. Compare whole-task latency, token/action parity, cache hits, recurrent replay, RAM/swap and sustained thermals before adoption. Do not remove synchronization or recurrent checkpoints on an assumption that CPU execution makes them unnecessary.

Implementation and target procedure: `engine/patches/prism-cpu-mtp/README.md` in Xeno; cumulative patch pinned to sudoingX ff414120c343e6e6cb868013c99f1dde52b27e70. Target compilation/validation and runtime activation remain pending. No model, engine service, GPU or WSL was loaded here; only the standalone policy test binary ran.

