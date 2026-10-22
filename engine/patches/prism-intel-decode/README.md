# Intel decode v3: three paper-derived source candidates

22 October 2026. **Source preparation only, inactive.** This package does not
download a model, build/install a runtime, start a process, change live profiles,
touch Hermes or run GPU/WSL. No native code/model was executed this revision.
Compiler and Python results are recorded in `verification.json`; there is no
verified speedup, compiled-kernel parity or target RAM/thermal acceptance yet.

## Pin and composition

Exact base: [sudoingX/llama.cpp ff414120c343e6e6cb868013c99f1dde52b27e70](https://github.com/sudoingX/llama.cpp/tree/ff414120c343e6e6cb868013c99f1dde52b27e70).
`intel-decode.patch` is cumulative: it includes the existing PTQ1 AVX2/reuse v2
and CPU-MTP v1 changes plus the optional lookup path and phase hooks. **Do not
stack it with `prism-ptq1-avx2`, `prism-ptq1-reuse`, `prism-cpu-mtp` or the older
greedy-MTP patch.** Keep those original packages intact for comparisons/recovery.

`python3 apply.py /path/to/clean/pinned/source` checks commit, normalized input
hashes, patch hash and applicability without changing source. `--apply` explicitly
applies source only. New-file conflicts or changed pinned inputs cause refusal;
do not reset a modified checkout to make the check pass. A clean pinned checkout
and independent combined full-runtime tests remain necessary before deployment.
`manifest.json` pins every changed source; `LICENSE.llama-cpp` retains its license.
This is the external Bonsai Prism backend, not a native Xenolith Gemma port.

## 1. bitnet.cpp: exact row-tiled AVX2 lookup

`ptq1-pair-lut.h` implements an independent TL1_1-style exact pair lookup. Nine
ternary pair sums are stored in separate low/high byte planes and reconstructed
as signed int16 values with AVX2 shuffles. No table requantization is performed.
One 128-weight block per row is decoded into bounded local indices; tables are
reused across up to 16 output rows, and indices across 1–5 activation columns.
The four original Q8 scale groups and per-output block/FP32 reduction order remain.
There is no persistent weight expansion, rewritten GGUF or extra model.

The patch wires the tile into the **actual GGML matmul chunk** only when
`GGML_PTQ1_ENABLE_PAIR_LUT` is defined consistently in CPU C translation units.
Ordinary 2-D PTQ1/Q8, contiguous K, FP32 output and 1–5 total/current columns are
required. Broadcasts, reference requests, larger prefill, other types/strides and
unsupported CPUs retain the original path. With both opt-ins defined, LUT takes
priority over tiny-batch reuse; benchmark separate builds to choose between them.
`GGML_PTQ1_DISABLE_AVX2` overrides both; neither new opt-in is defined by default.
The existing cumulative single-column AVX2 path remains as in v2 when applicable.

Lookup construction, index/scatter work, row stride/cache behavior and register
pressure can outweigh saved multiply/add work. This is a candidate, not a claim
that a shuffle kernel is faster. Use strict floating-point compilation: no
fast-math/reassociation, `-ffp-contract=off`, with target parity required.

## 2. BITCOS: exact conditional codec and comparison kernel

`ptq1-bitcos.h` provides caller-buffer block/row pack, bounded view validation,
portable bit deposit, optional BMI2 PDEP and an AVX2 dot kernel. A block holds its
original 2-byte scale, a 16-byte nonzero bitmap and compact positive-sign bits.
The dot preserves all four Q8 scales; `-128` activations do not require negation.
The codec interprets every byte with pinned base-3 semantics, including otherwise
noncanonical encodings. Scale bits and decoded weights remain unchanged.

Row pack preflights capacities before writes, includes uint32 block offsets in
its optional size gate and leaves raw PTQ1 selected when conversion is not smaller.
Codec input, output and offset buffers must be separate/non-overlapping; this is
not an in-place conversion API.
Bounded sign loads do not assume padding. Feature compilation with `-mbmi2` needs
target BMI2 support; `GGML_PTQ1_BITCOS_NO_BMI2` retains the portable comparison.
Validated row views borrow immutable caller-owned data/offsets: validation happens
once before repeated decode, not as another full metadata pass for every token.
The checked row-dot entry remains available; changing buffers invalidates a view.
No AVX-VNNI-INT8 or AVX-512 is assumed. Repacking is explicit caller work.

`bitcos_cost.py --model MODEL.gguf` samples up to 4096 blocks per tensor, labels
estimates and **cannot select sampled tensors**. `--full-scan` reads every PTQ1
block; it may take substantial time. The bounded reader recognizes pinned type143,
checks metadata/shape/offset bounds, counts exact byte padding and row indices,
and never writes to the model. Its packing floor excludes descriptors/alignment,
page cache and driver allocations. It does not certify the file SHA256.

**Deliberate integration boundary:** BITCOS is a codec/kernel and feasibility
candidate, not a registered GGUF type, resident whole-model conversion or automatic
GGML dispatch. PTQ1 remains the live/default representation. A backend layout hook
is still gated on this exact model's measured density, full numerical tests and a
single resident representation compatible with Intel Vulkan prefill. Registering
an unconditional second weight cache would contradict that gate. A smaller format
is not automatically faster; include unpack/indexing and DRAM measurements.

## 3. Sandwich: topology plans and real server phase hooks

`sandwich_plans.py` reads saved topology or, with explicit `--inspect-target`, the
Linux target's sysfs and process affinity. It excludes offline/disallowed CPUs,
groups SMT siblings into physical cores, records cache sharing and preserves
unknown core types. It does not infer P/E type from clock frequency or presume
symmetric/power-of-two cores. Candidate masks include physical/logical and reported
core-type sets; bounded plans are printed as JSON, never executed.
Partial groups also follow actual shared L2/L3 membership where reported, with
round-robin candidate coverage so the budget is not spent only on one core class.

The cumulative patch extends existing `common_threadpools` ownership rather than
creating another model process. `LLAMA_SANDWICH_VERIFY_THREADS=1..64` explicitly
enables one additional paused verification pool on the **server target context**;
an optional plain-hex `LLAMA_SANDWICH_VERIFY_MASK` chooses its permitted cores.
Decode and prefill continue using existing `-t`/`-tb` and mask/strict/poll settings.
Draft/MTP contexts retain their existing pools and do not allocate a third pool.
Absent the variable, phase selection does nothing and introduces no extra sync.
Malformed masks/counts and GPU/multislot/embedding activation are rejected.

Before `llama_decode`, the server classifies its actual token provenance: prompt
tokens select prefill even for one token; generated single tokens select decode;
short generated batches select verification; mixed/multislot input selects the
original two-pool behavior. On a phase change, synchronize outstanding graphs,
pause old pools, attach the selected existing pool to both roles and set matching
thread counts. This avoids the runtime's `n_tokens>1` heuristic assigning short
verification to the prefill plan. Repeated same-phase batches do not switch again.
Pool destruction/sleep follows the original owner lifetime; no KV/recurrent state,
acceptance rule, cancellation, queue ownership or sampler is changed.

All extra pool stacks/metadata and phase synchronization costs must be measured.
Use the pinned backend's native thread-pool implementation (`GGML_OPENMP=OFF`)
for this candidate: OpenMP builds ignore its pool pause/affinity semantics. A
compile pass against mocks does not establish real pool wakeup or affinity.
This candidate is CPU-only; the existing separate Vulkan prefill route is preserved,
not newly migrated between devices. No compiler replacement or layer-specific
kernel autotuner is claimed. The prior MTP window policy is inherited unchanged.

Use `--topology topology.json --cost-key key.json` to render at most 16 plans
(`--limit 1..32`). The key requires model/head hashes, runtime build, kernel flags,
KV type, context bucket, full sampler/MTP policy and workload hash. Saved topology
and phase masks also enter each plan digest. With `--results measured.json
--baseline PLAN_ID`, the tool selects but does not apply a plan. Require at least
three finite whole-task timing samples, correctness/memory passes and no observed
timing overlap with the baseline. A fixed MTP policy in the cost key prevents
confounding with another online tuner. There is no arbitrary minimum speed gain.
This is offline plan selection, not a new jointly adaptive MTP controller.

## Checks and future target acceptance

Completed here: cumulative patch applicability/output-hash/reverse round-trip;
actual patched x86 dispatch and full matmul chunk compile in five opt-in/fallback
variants; C++ kernel/benchmark objects compile with AVX2, BMI2 and forced portable
signs; actual scheduler methods and provenance policy compile against mock APIs;
seven Python analyzer/topology/cost-key tests pass. These are **not a linked full
runtime build, executed SIMD test or numerical/model certification**.

Future authorized target tests (not run here):

```sh
python3 -B -m unittest -v test_tools
c++ -std=c++17 -O2 -mavx2 -ffp-contract=off -fno-fast-math test-kernels.cpp -o test-kernels
./test-kernels
c++ -std=c++17 -O2 test-sandwich.cpp -o test-sandwich
./test-sandwich
c++ -std=c++17 -O2 -mavx2 -ffp-contract=off -fno-fast-math benchmark-kernels.cpp -o benchmark-kernels
./benchmark-kernels 32768 8192 5 5 -1
```

Keep assertions enabled. Repeat with BMI2 only on supported target hardware, plus
ASan/UBSan/guard pages. Kernel tests cover noncanonical bytes, activation extremes,
signed finite scales, 0/1/3/17 blocks, partial/16-row tiles, 1–5 columns, padded
strides, row-cost gates, insufficient buffers, invalid offsets and sentinels. The
mock scheduler test covers disabled behavior, provenance, sync-before-switch,
matching counts/pools, repeated-phase no-op, invalid inputs and pool lifetimes.
No real thread-pool race/performance claim follows from mocks.

Benchmark labels: 0 pinned baseline, 1 prior AVX2, 2 new pair LUT, 3 BITCOS,
4 prior batch reuse (single-column fallback). Table setup is inside timed passes;
BITCOS conversion time and offset/payload bytes are reported separately. Default
mixed bytes and optional encodings 0..255 test differing densities; encoding128
is an intentionally all-zero extreme, never a model-representative claim.
The benchmark holds small synthetic alternatives for comparison, not duplicate
production model weights. It reports microseconds, not model tokens/second.

Then build separate clean pinned full runtimes, validate dispatch/strides/reference
fallback, logits/greedy tokens/agent actions, all MTP accept/reject/replay cases,
prefix restoration and cancellation. Compare same questions/budgets, MTP off/fixed/
adaptive separately, actual DRAM counters, CPU decode, iGPU prefill, whole-workflow
latency, RSS/swap, pool memory and sustained temperatures. Retain fallback on any
parity failure, memory overflow or timing indistinguishable from noise. The
preparation PC's 5 GiB reserve is not a newly selected 16 GB target reserve.

## Attribution

Independent algorithm adaptations, not complete paper ports or replicated results:

- [BITCOS](https://arxiv.org/abs/2609.16338): bitmap/compact-sign codec and unpack comparison.
- [bitnet.cpp](https://arxiv.org/abs/2502.11880): exact element-wise lookup with two int16 byte planes.
- [Sandwich](https://arxiv.org/abs/2507.18454): topology-aware phase plans and switching.
- [T-MAC](https://arxiv.org/abs/2407.00088): prior v2 locality/unpack-reuse and exact-LUT comparison, inherited here without approximate arithmetic.

The [engine README](../../README.md#decode-research-roadmap-and-paper-references)
cites the remaining papers, including FlashAttention/Flash-Decoding as audited
upstream mechanisms, FlashInfer as pending layout work, and LLM in a Flash's
offloading as rejected for this resident dense workload. This increment adds no
attention kernel, CUDA dependency, speculative sparsity or offloading.
