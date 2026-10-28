# CPU hot-path increment - 23 October 2026

This two-header delta follows the composed `prism-suffix-workspace` and
`prism-intel-decode-increment` sources at Prism commit
`ff414120c343e6e6cb868013c99f1dde52b27e70`. It does not replace older packages.
Check exact source hashes with `python apply.py SOURCE`; add `--apply` to apply,
or `--reverse` to restore both previous headers. Reverse this layer before older
layers. Never force application over source drift. Payloads and patch are hashed.

## Suffix search and voting

The existing opt-in `LLAMA_SPEC_SUFFIX_WORKSPACE=1` path scans the retained prompt
once, computes each candidate's longest matching suffix up to width 16, and keeps
matches at the largest width. It replaces sorted votes with two-counter candidate
elimination and exact recounting. Every qualifying vote is at least half of all
matches, hence above one third of available votes and retained by elimination.
The recount preserves smallest-token tie breaking and includes exhausted matches
in the acceptance denominator. History remains bounded to 8191 prompt tokens plus
the current token; caps, alias fallback, request-owned scratch, release hooks and
zero warm allocations are unchanged. No new persistent cache or index is added.

## SIMD scale candidate: disabled by default

Build with `GGML_PTQ1_PAIR_LUT_VECTOR_SCALE` only for comparisons, with AVX2 and
strict FP (`-ffp-contract=off`, no fast-math). It converts sums directly into two
FP32 vectors, applies the four Q8 scales in their original order and accumulates
blocks with separate multiply/add operations. Tail lanes use initialized scales;
only valid output rows are stored. The old scalar scaling code remains available
with this flag absent. Both existing compact and noncompact variants are retained.

Mixed synthetic results on the preparation PC do NOT justify enabling this flag:
the compact variant often became slower, especially on tail tiles. Explicit SIMD
is not automatically better than compiler-generated code. No Intel laptop or
model throughput improvement is claimed. No installed binary was rebuilt.

## Verification and scope

`verification.json` and logs contain model-free checks run with GCC 13.2, AVX2,
strict FP on this Windows preparation PC: 836772 suffix output comparisons against
the legacy oracle, alias/cap/history boundaries, four independent workspaces and
zero warm allocations; 40960 bitwise tile comparisons per compact setting (81920
total), every row count 1..16, columns 1..5, zero/multiple blocks, finite signed
FP16 scales, signed Q8 extremes, random bytes, padded input/output strides and
output canaries. Synthetic timing includes unpacking/LUT setup. This is not Intel
deployment certification, model inference or GPU validation.

The suffix benchmark labels `legacy_us` / `reuse_us` compare the PREVIOUS scratch
implementation against this delta (not the original allocating implementation).
Short repetitive inputs can be slightly slower. See logs rather than assuming
every pattern improves. The scalar/vector benchmark reports median times over
nine interleaved trials. Build optimizations must preserve contraction settings.

## Can MTP and suffix work together?

Both are available to the current selector, but only one drafts each cycle. Every
chosen chain goes through the target verifier. The current code does not merge
their chains and does not automatically fall back to MTP on a suffix miss.

A future hybrid arm can run bounded suffix lookup first, verify its chain on a
hit, and call MTP on a miss. Its measured reward must include lookup, draft,
synchronization, verification and rollback costs so the selector can disable it
when unprofitable. Preserve profile-version compatibility and implementation
ownership on hit/miss, including draft catch-up after suffix acceptance.

Do not concatenate chains without first advancing the MTP hidden/KV/recurrent
state through the suffix prefix. Two alternative branches require additional
state checkpoints and verification scheduling. Neither hybrid extension is
implemented in this delta; recurrent-state parity tests and target timing are
required before enabling one. No second model-weight copy is needed for the
suffix-first fallback design. Existing Intel iGPU prefill remains unchanged.

This delta optimizes existing local code; it adds no new paper-derived algorithm
claim. Earlier paper attribution remains in the preceding packages and engine
README. Model weights, scales, quantization and sampling rules are unchanged.

Full `common/speculative.cpp` and x86 `quants.c` compile checks also pass;
quantization was checked with default flags and compact/vector enabled. The full
Prism quantization unit requires FMA/F16C compiler target flags for unrelated
existing kernels; FP contraction remained disabled. Package apply/reverse and
source-drift rejection pass in an isolated fixture (commit query mocked there).
