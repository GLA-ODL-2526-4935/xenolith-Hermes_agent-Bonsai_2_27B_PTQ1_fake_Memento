# Intel decode increment v1 — optional delta on v3

22 October 2026. Source-only, inactive. No model/engine/native test executable,
GPU or WSL was run. This is a reversible addition to `prism-intel-decode` v3,
not a replacement runtime or a native Xenolith Bonsai port. Original packages,
profiles, weights and the Intel Vulkan architecture are unchanged.

## Composition and rollback

Start with clean sudoingX/llama.cpp commit
`ff414120c343e6e6cb868013c99f1dde52b27e70`, then apply the v3 cumulative package
using its documented checks. **Only then** check this delta:

```sh
python3 apply.py /path/to/pinned-v3-source
python3 apply.py /path/to/pinned-v3-source --apply
# Revert just this increment; refuses drift in any prerequisite source:
python3 apply.py /path/to/pinned-v3-source --reverse
```

Default is check-only. Commit, every v3 source hash, new-file conflicts and patch
digest are checked before mutation; output hashes are checked afterward. Reverse
requires the exact modified state. Never reset user changes to force applicability.
Do not stack v3 with the earlier AVX2/reuse/CPU-MTP cumulative packages. This delta
changes just the pair-LUT header and MTP capture site, and adds one helper header.
Neither flag below is enabled by default. Disable flags for a baseline build, or
reverse the delta to restore the v3 source exactly.

## A. Smaller exact lookup work

Define `GGML_PTQ1_PAIR_LUT_COMPACT` consistently in CPU C translation units to
choose this variant. Runtime LUT dispatch still additionally requires existing
`GGML_PTQ1_ENABLE_PAIR_LUT`, AVX2 and all v3 shape/reference guards. The existing
`GGML_PTQ1_DISABLE_AVX2` override and non-AVX2 fallback remain intact.

- Accumulate the 16 pair lookups for a 32-activation scale group in signed int16;
  widen to int32 once per group. Each lookup is within [-256,256], so every prefix
  sum is within [-4096,4096]. There is no saturation, overflow or approximate sum.
  Source widening operations fall from 32 to 2 per group; machine cost requires
  measurement. Four Q8 scales and FP32 block/reduction order remain unchanged.
- Build exact low/high table planes algebraically in SIMD values, avoiding the
  scalar table loop and explicit 32-byte table staging. Compilers may still spill.
- Decode the final eight packed trits in SSE2 and write exactly eight bytes,
  replacing nested scalar multiply/shift loops. Two-byte input loads remain bounded.
- Full 16-row tiles overwrite every index byte and skip redundant scratch clearing.
  Partial tiles still initialize all lanes read by the vector lookup.

Existing table reuse across rows and unpack reuse across MTP columns are inherited,
not counted as new work. No persistent expanded index table/weight cache, activation
requantization, modified numerical weights or global packed layout is introduced.
These reduce source operations/local scratch traffic; **actual DRAM bytes/token,
instruction counts and speed have not been measured**.

## B. One copy per captured MTP hidden row

Define `LLAMA_MTP_COMPACT_VERIFY` for the common C++ target to select the helper.
For N captured rows, keep rows 0..N-2 in owned `verify_h`, and copy final row N-1
directly into owned `pending_h`. Previously it went through `verify_h` first.
The existing acceptance branch already reads `verify_h` only for earlier rows;
full acceptance uses `pending_h` and performs no additional copy.

This removes exactly one hidden-row copy (`4*n_embd` payload bytes, one read plus
one write) per sequence per nonempty capture, and one row of vector live size.
Allocator capacity is reused and not shrunk; no RSS reduction is promised.
`verify_h_rows` retains the original N and rejection indices remain unchanged.
N=1 needs no earlier-row storage. Per-row getters and owned storage are retained;
no target-output pointer survives another decode. There is no new synchronization,
KV/recurrent mutation, rollback shortcut or sampler/acceptance-policy change.

The process hook is shared by prefill and verification, so the saving applies to
both. Earlier rows are retained even during prefill; dropping them requires
additional lifecycle evidence. No prefill/verification provenance guess is added.

## Checks completed versus target acceptance

`verification.json` records patch apply/hash/reverse; shipped apply-tool fixture
checks (only HEAD mocked); four Python mathematical/byte-contract tests; C/C++
compile checks for actual dispatch and fallback shapes, both original/compact
kernel/benchmark objects, actual MTP capture and acceptance-copy hooks. The native
before/after test covers all 65,536 activation pairs, all 65,536 packed tail words,
rows1..16, columns1..5, output guards, NaN payload copying, buffer grow/shrink reuse
and all acceptance positions. **Those C++ tests were compiled, not executed.**
No full runtime was linked, no native numerical parity established, no model run,
no target DRAM/latency/thermal evidence. Python oracles are not executed AVX2 parity.

Future authorized target checks (strict FP, assertions on):

```sh
python3 -B -m unittest -v test_oracles
c++ -std=c++17 -O2 -mavx2 -ffp-contract=off -fno-fast-math test-increment.cpp -o test-increment
./test-increment
# Existing randomized baseline-vs-LUT/BITCOS coverage, new variant selected:
c++ -std=c++17 -O2 -mavx2 -DGGML_PTQ1_PAIR_LUT_COMPACT -ffp-contract=off -fno-fast-math test-kernels.cpp -o test-kernels
./test-kernels
```

Repeat with sanitizers/guard pages and exact pinned full-runtime builds. Compare
neither flag, each flag separately and both using the same v3 base, compiler and
cost key. Existing synthetic `benchmark-kernels.cpp` compiles with either flag;
compare separate builds, including table setup. Do not interpret it as model tok/s.
Use whole-task repeated timing, actual DRAM counters, RSS/swap and sustained thermals.

Full model acceptance must cover every accepted prefix 0..4, all-rejected and
all-accepted batches, sequence reset/reuse, prompt boundaries, cancellation,
checkpoint restore/replay and bounded recurrent rollback at/over its capacity.
Check hidden carryover and logits/greedy tokens, not just final answer agreement.
Keep target and draft state, sampler/RNG and positions consistent. Preserve all
existing checkpoint/state copies until these tests establish an exact alternative.
Repeat CPU decode and the existing Intel iGPU path separately; no device migration
or new Vulkan layout is enabled by this increment. Keep fallback on any failure
or timing indistinguishable from noise. Select the 16GB target's RAM reserve
separately; its existing launch guard remains a known configuration issue.

## Research and attribution

The exact LUT is an independent continuation of the previously cited
[bitnet.cpp](https://arxiv.org/abs/2502.11880) and
[T-MAC](https://arxiv.org/abs/2407.00088) candidates; the local integer bounds and
copy elision are engineering changes, not a claim to implement another full paper.
New investigations and compatibility decisions, including
[FLUTE](https://arxiv.org/abs/2407.10960),
[LUT-GEMM](https://arxiv.org/abs/2206.09557),
[Ghidorah](https://arxiv.org/abs/2505.23219) and
[ReplaySSM](https://arxiv.org/abs/2610.13778), are in
[the additional review](../../docs/ADDITIONAL_INFERENCE_REVIEW.md).
None is claimed as a new complete implementation. Original llama.cpp MIT license
is retained in `LICENSE.llama-cpp`.
