# Runtime lifecycle increment v1

23 October 2026. Implements the feasible subset of the six-repository review for
the **external Prism Bonsai backend**. The native Xenolith Gemma/Level Zero graph,
Intel Vulkan placement, model weights and installed runtime binaries are unchanged.

## Composition and rollback

This is an additive delta on the **current composed local source snapshot** at
sudoingX/llama.cpp `ff414120c343e6e6cb868013c99f1dde52b27e70`: Intel-decode v3,
compact increment, source selection/QIGen and persistent/default source profiles.
The exact nine changed/new input files and ten prerequisite hashes in
`manifest.json` are authoritative. It is not directly applicable to clean upstream,
clean ff41412, v3 alone or an arbitrary installed Prism package. Earlier alternative
cumulative patches must not be stacked. No older package is replaced.

```sh
python3 apply.py /path/to/composed-source             # check only
python3 apply.py /path/to/composed-source --apply     # source only
python3 apply.py /path/to/composed-source --reverse   # undo this delta first
```

HEAD, all prerequisite/input hashes and patch digest are checked before applying.
Reverse checks the exact modified files and prerequisites. On drift, stop and
review; do not reset user changes or bypass hashes. This tool does not build,
download, load a model or start a server. A new matching full runtime build is
required before any inference can use these changes. Existing binaries retain
their previous behavior. Reverse this delta before reversing older packages.

## 1. MTP row identity and rejection contracts

Qwen35 and Qwen35MoE MTP graphs now gather `inp_out_ids` **before** storing
`t_h_nextn` for masked capture. Unmasked capture still retains every row. This
matches the context extractor's existing masked `n_outputs` contract; no sampling,
target acceptance, model/head weights, snapshot precision or rollback shortcut is
introduced. The fix is active when the source delta is applied/built.

The model-free regression executes the actual before/after graph statements with
distinct row IDs. It covers all masks for lengths1..9, non-prefix/reversed selections
and unmasked capture. The old statements produce1021 masked-row mismatches; the
new statements pass. This proves the row-selection regression in that fixture,
not its frequency or quality impact in previous model runs.

The existing actual `common_sampler_sample_and_accept_n` function is executed with
synthetic target responses for every rejection position at draft lengths0..16,
including nonconsecutive logit indices, full acceptance, bonus and sampler history.
Its implementation and the actual recurrent rollback code remain unchanged. This
does not execute a real GDN/conv rollback or prove batched/unbatched model parity.

## 2. Optional cache-line CPU partitioning

Compile the CPU C target with `GGML_CPU_CACHELINE_SCHEDULE` to enable; absent by
default. Append this define to existing build flags rather than replacing them.
`GGML_CPU_CACHELINE_MIN_BYTES` can override the experimental65536-byte minimum
weight work per worker. This is a tuning parameter, not a measured Intel optimum.

- Only ordinary complete one-column matvec, one-row dot traits, supported F32
  output alignment and non-reference requests enter the policy.
- Divide physical64-byte output lines, accounting for a view's partial first line.
  Restrict effective workers for small rows; retain unsnapped parallelism when
  there are few long rows and line ownership would unnecessarily serialize them.
- Run after activation conversion and its existing barrier, before generic chunk
  dispatch. The existing PTQ1/QIGen/LUT implementation still computes each slice;
  all reduction/scale/Hadamard operations remain unchanged. No second threadpool,
  new packing or scratch buffer is introduced. Idle workers still participate in
  the surrounding graph/threadpool lifecycle.
- Higher dimensions, reference mode, unsupported alignment and other dot traits
  keep the original dispatch. Multi-column MTP verification remains on its existing
  schedule; this increment does not claim a new verification reuse optimization.

## 3. Measurements and bounded compute-buffer reservations

`GGML_RUNTIME_TELEMETRY=1` enables fixed-size process aggregates and graph-allocator
summaries. Default is off. No prompt, token IDs, user data, per-token arrays or
unbounded trace is retained. C++ counters are mutex-protected and saturate on
overflow; periodic logs are at powers of two, with summaries at slot release,
backend free and graph-allocator free. The C graph allocator retains its existing
single-owner execution assumption.

Counters include allocation API calls/requested bytes/failures, logical tensor-copy
payload bytes, backend synchronize calls, verification attempts, drafted/matched
tokens, rejected suffix lengths and checkpoint-replay decisions. Graph summaries
include reservation count, allocation attempts and conservative capacity high-water
bounds. Aggregate verification attempts can include replay and all draft sources;
they are not a new per-source acceptance-rate definition. The existing selector's
per-arm accounting and normal server acceptance metrics remain authoritative.

**Measurement limits:** allocation requests can include nested calls and repeated
capacity reuse; requested bytes are not live bytes or RSS. Copy bytes count calls
through the instrumented full-tensor copy routes, including the scheduler's direct
async success. Backend-internal kernels, raw memcpy, partial writes, page migration
and hardware cache traffic are not fully counted. These are API payload counters,
not measured DRAM traffic, and must not be added to memory-controller traffic.
Sync counts cover the backend synchronize API, not every event wait.

`GGML_GRAPH_BUFFER_LIMIT_MIB=N` optionally caps a **single GGML graph allocator's
planned compute-buffer capacity bound**; unset or0 means no configured cap.
Positive values use exact integer MiB. Invalid/negative/overflowing input refuses
physical reservation instead of silently disabling the limit.

Preflight counts each shared allocator once and includes retained high-water
capacity. It rejects over-budget/overflow plans before any physical buffer is
freed or allocated. A failed reservation blocks graph allocation until an explicit
successful reserve, so changed planning metadata cannot be used with old buffers.
It does not evict active state, requantize checkpoints, shorten required recurrent
tails or silently change the MTP window. Existing allocation failure handling must
stop/retry explicitly.

This is **not a total-process RAM cap**. Model mappings, persistent KV/recurrent
states, target versus draft allocators, host vectors, driver allocations and
backend-private alignment/reservations remain separate. Multiple contexts each
have their own limit. Do not pick N from16GB total RAM or assume the preparation
laptop's5GiB reserve applies to the deployment machine. A target RSS/available-RAM
guard remains necessary. No new limit is enabled in shipped launch profiles.

## 4. Conservative transfer-overhead reduction; larger work deferred

`GGML_SCHED_SYNC_INPUTS_ONCE=1` optionally coalesces destination waits across a
**consecutive run of synchronous user-input copies** in one graph split. Default
is off. The first input still waits for the old destination use; each copy remains
synchronous, preserving caller-buffer lifetime. A non-user path resets the flag
because it may enqueue async work. Nothing crosses a split/event generation.
No host staging ring, cross-request pointer cache or asynchronous lifetime change
is introduced. Existing events and non-user synchronization stay intact.

Direct GDN snapshot writes and a persistent alternative packed-weight layout are
**not added**: destination alias/lifetime proofs and target copy/cache measurements
are still missing, and the earlier compact hidden-row capture and bounded PTQ1
LUT/batch experiments already address related copies/layouts. No duplicate arena,
selector, checkpoint pool, second model representation, lossy checkpoint format,
OpenVINO/NPU path or CUDA dependency was added.

## Verification and remaining acceptance

`verification.json` records model-free tests executed on the Windows preparation
host and full source translation-unit compile checks. The fixture executes actual
graph/verification/preflight/input-wait statements with small mocked tensor/backend
interfaces; it is not a full linked GGML runtime or recurrent-kernel execution.
Whole allocator/backend/Qwen35/Qwen35MoE/server compile checks pass. CPU compile
checks pass with the scheduling switch off/on and with AVX2 QIGen enabled.
MinGW lacks Windows thread-power-throttling SDK declarations, so CPU compile-only
checks select the existing older-Windows priority branch; production source is
unchanged there. No compiler warnings were treated as proof of runtime correctness.

Portable reproduction after applying to the exact source, with GCC/G++ available:

```sh
python3 test_runtime_lifecycle.py --source /path/to/composed-source \
  --build-dir /path/to/writable-test-output
```

- [x] Model-free graph row identity, all verifier rejection positions, physical
  output coverage, parser/overflow, bounded preflight/alias and concurrent counters.
- [x] Actual input-wait hook across every8-input user/non-user pattern, events
  present/absent, enabled/disabled; copy count preserved.
- [x] Full edited translation-unit compile checks and source-patch round-trip.
- [ ] Full linked Linux CPU/Vulkan build and actual recurrent/conv rollback parity.
- [ ] Real-model greedy token/logit and MTP row parity at microbatch boundaries,
  cancellation, replay, source switches and memory-reservation failures.
- [ ] Target Intel speed, RSS, DRAM traffic and thermal measurements, each switch
  independently, using the same binary/profile identity for learned costs.
- [ ] Direct snapshot writes/alternative layouts only if profiling justifies them.

No speedup, extra acceptance, lower DRAM traffic or Intel certification is claimed.
Changing the binary invalidates the existing cost-profile runtime identity; do not
reuse NVIDIA-derived timing priors for a newly built Intel runtime.

## Sources and notices

- Arcint [0010 masked capture](https://github.com/marfrit/arcint/blob/0b0a95c2eff5fa5664437fcceeaca04453fd59d4/contrib/llama.cpp/patches/0010-mtp-masked-nextn-rows-of-the-outputs.patch),
  [0047 input copies](https://github.com/marfrit/arcint/blob/0b0a95c2eff5fa5664437fcceeaca04453fd59d4/contrib/llama.cpp/patches/0047-input-copies.patch),
  [0050 snapshots](https://github.com/marfrit/arcint/blob/0b0a95c2eff5fa5664437fcceeaca04453fd59d4/contrib/llama.cpp/patches/0050-gdn-snapshots-in-cache.patch).
  Graph fix adapted from the masked-capture pattern; narrower wait-coalescing
  implementation. Apache-2.0 license included as `LICENSE.arcint`.
- llama.cpp-ng [CPU scheduling](https://github.com/mikechambers84/llama.cpp-ng/blob/4fe72e9af820c7f3b6848fb6d46ce1f6f237969b/ggml/src/ggml-cpu/ggml-cpu.c):
  destination-line ownership and minimum work per worker. Our partition helper
  additionally handles physical misalignment and is independently implemented.
- MNN [allocator](https://github.com/alibaba/MNN/blob/5c22b9f082275dce1cdf82c8a1a0b2a4a1c2976d/source/core/BufferAllocator.cpp)
  and SGLang [state commit](https://github.com/sgl-project/sglang/blob/f9129915a36a2d0bac708110c617ca1861b5a100/python/sglang/srt/speculative/spec_utils.py):
  lifecycle/design references, not copied backends or compressed state.
- Existing llama.cpp/Prism source remains subject to `LICENSE.llama-cpp` and file
  notices. New independent helpers use MIT SPDX identifiers. No LLM.xpu code copied.
  Previous paper references and optimization packages remain in engine/README.md.
