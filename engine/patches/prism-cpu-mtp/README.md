# CPU MTP candidate — cpu-mtp-v1

22 October 2026. Intel CPU-only/Linux target; no dGPU assumption. No PTQ1 decoder, Hadamard, packing, matrix multiplication or quantization kernels are changed. Full runtime compilation and model benchmarks remain pending. The small policy test below was compiled and executed without an engine/model.

## Source and application

This **cumulative** patch includes the earlier greedy-mtp changes. Apply to a separate clean checkout of sudoingX/llama.cpp commit `ff414120c343e6e6cb868013c99f1dde52b27e70`, not on top of the earlier patch or to Prism73bfe6c. The manifest records normalized source hashes. Upstream licensing remains applicable.

```sh
python3 apply.py /path/to/pinned/source
python3 apply.py /path/to/pinned/source --apply
```

The first command checks only. Neither command builds, fetches, starts inference or reads model weights. Unexpected commit/file contents or an existing new header cause refusal. A compatible CPU executable with matching libraries must subsequently be built on the intended toolchain. Record complete source diff, dependencies and executable hash. Merely putting `cpu-mtp-v1` in a profile is a compatibility requirement, not binary attestation.

## Implemented source candidates

- Keep the greedy draft path at p_min0 and conditional sampler cloning from greedy-mtp.
- Reuse the per-sequence drafting bitmap instead of allocating one each draft call. Reserve the small verification-row buffers once; larger prompts can still grow them.
- Avoid copying the final hidden row again on all-accepted verification: process() already retained it. Partial acceptance still copies its selected row.
- Respect the per-sequence/context/output draft cap *during* generation. Previously the generic wrapper could trim already-generated excess drafts afterward. Zero-cap sequences do not decode a discarded draft.
- Add `--spec-draft-adaptive` / `--no-spec-draft-adaptive` (default off). Per-request/per-sequence policy explores windows1..min(4,n-max), samples each four times, then chooses lower exponentially smoothed **cycle microseconds per accepted-drafts-plus-one target token**. A different window is probed every32 valid cycles. Context/confidence-shortened cycles and invalid samples are excluded. Policy resets per request; no learned tuning state is persisted. With n-min>1 the fixed-window path remains in use.

Adaptive timing begins before drafting and ends at target acceptance, including verification and associated runtime work. It excludes HTTP/tool overhead and can be affected by system scheduling. It is not a claim that adaptation always beats a fixed window. It does not test MTP-off internally; compare that baseline separately. Warmup can lose time on short answers. Acceptance and cache rollback still use the original target verification rules.

No engine synchronization barrier, KV ownership rule or recurrent snapshot depth is removed. Batched verification already offers weight reuse; kernel work remains deferred.

## CPU launcher and harness

`profiles/bonsai27-linux-cpu-candidate.json` is opt-in and unvalidated: device none, target/draft GPU layers0, one slot,8192 context, low reasoning, output4096, p_min0, adaptive cap3, batch256/ubatch128, six decode/prompt threads, polling0, exact-prefix caching,128MiB prompt-cache budget and two context checkpoints. Idle-slot archiving is disabled. The latter checkpoints/live context are **additional memory**, not included in the128MiB budget. The >=5GiB available-RAM stop remains. Baseline CPU profile still has MTP off and prefix reuse off.

Optional profile fields: `threads_batch`, nonzero plain-hex `cpu_mask`/`cpu_mask_batch`, polling0/1 for target/draft, `prefix_cache`, `cache_ram_mib` (0..256), `adaptive_mtp`, `draft_backend_sampling`. Affinity is empty until target topology is inspected; no guessed performance-core mapping. At least n-max+1 tokens must fit a microbatch. CPU backend sampling is selectable: it is CPU execution, not GPU placement; compare on/off rather than assume one is faster. The CPU candidate chooses off; ordinary configurations preserve the runtime's on default. No CUDA environment switch is emitted for these CPU/Vulkan profiles.

The opt-in Hermes `controlled-run` flags `--reuse-prefix --keep-alive` reorder the same JSON information with original user sources/tool definitions/plan first, growing evidence next and phase/current step last. They request exact prompt reuse on dedicated slot0 and reuse one local HTTP connection. Every request still contains only system/user messages and excludes prior assistant reasoning. User corrections invalidate the changed prefix naturally; old context is never treated as a replacement for supplied input. The runtime must restore/recompute hybrid state correctly at the exact reusable token boundary; this remains a target acceptance gate. No persistent KV cache file is introduced.

Transport retries, redirects, proxies and hosted fallback remain forbidden. Connections close after the foreground controller finishes or fails. Durable pending-action checkpoints and evidence readback remain intact. These flags are not enabled in live profiles; the ordinary Hermes loop is unchanged. The existing CLI still defaults to reasoning off/output1024; adopting the launcher's low/4096 defaults in a production provider remains a separate configuration decision, not silently changed here.

## Offline tests and tuning plans

```sh
c++ -std=c++11 -Wall -Wextra -Werror test-policy.cpp -o /tmp/test-mtp-policy
/tmp/test-mtp-policy
python3 ../../cpu_sweep.py --engine /path/to/llama-server --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf
```

The second Python tool only renders24 plans: threads2/4/6/8, batch128/ubatch64 or256/128, MTP-off/fixed3/adaptive3. It never opens weights or starts a process. Extend separate prompt-thread counts, affinity and polling only after the first target measurements. Use the same prompt/output budgets and warmup protocol; compare repeated runs and sustained thermals, not only one short completion. The standalone C++ test exercises actual policy code, cost-based selection, adaptation, excluded samples, exploration, bounds/reset and the existing sampler-clone bound. It is not a full engine/cache test.

Required before activation: full CPU runtime build and upstream tests; token-ID/action parity against original and MTP-off; reject-first/middle/last/all, EOS/cancel, short/long contexts, full-checkpoint replay; prefix reuse on/off equivalence after corrections/plan changes/removal of reasoning; HTTP error and cancellation lifecycle; measured prompt/decode/whole-task time, cache hits, cycles, peak RAM/swap and steady-state thermals. Keep a pinned original-runtime fallback. PTQ1 kernel efficiency is explicitly deferred at the user's request.
