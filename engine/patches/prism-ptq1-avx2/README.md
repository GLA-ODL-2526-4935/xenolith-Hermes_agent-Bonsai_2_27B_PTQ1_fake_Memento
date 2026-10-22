# PTQ1 AVX2 source candidate — ptq1-avx2-v1

Prepared 22 October 2026 for the Bonsai Prism process backend. **Source prepared and compile-checked only. The kernel, test program, full engine and model have not been executed. No speedup or numerical parity is claimed.**

## Scope

This adds an AVX2 specialization to `ggml_vec_dot_ptq1_0_q8_0` in the external runtime that Xenolith's Bonsai launcher uses. It does not make the native Gemma/Level Zero engine accept PTQ1, change the on-disk quantization, alter the model or activate a live profile.

The existing 128-bit implementation separately decodes the first 16 payload bytes and the final 8 payload bytes for five trit stages. This candidate pairs those streams in AVX2's two lanes, performs the threshold extraction and offset-corrected integer dot arithmetic together, and routes their partial sums to the original four Q8 activation groups. The final eight `qh` trits retain the existing SSE computation.

Loads remain bounded to 16 + 8 payload bytes, 2 `qh` bytes, and the corresponding 16/8 activation bytes. There is no 32-byte overread of the 24-byte payload, persistent weight expansion or per-block heap allocation. The code retains signed-activation `-128` handling and separate scales. Float scale/reduction order follows the original source; compiler flags and real output parity still require testing.

The existing SSE2/SSSE3 and generic fallback bodies are retained. The AVX2 branch is selected only in an AVX2-enabled CPU build variant. Use the runtime's normal CPU feature selection; do not enable AVX2 for unsupported machines. `-DGGML_PTQ1_DISABLE_AVX2` forces the retained baseline branch even in an AVX2 build for comparison/rollback.

VNNI, multi-token verification microkernels, prefill repacking and Hadamard fusion remain separate future changes. Wider arithmetic can lose to lane-assembly costs or bandwidth limits; this is a candidate to measure, not a guaranteed win.

## Pinned source and composition

Repository: https://github.com/sudoingX/llama.cpp

Commit: `ff414120c343e6e6cb868013c99f1dde52b27e70`

The relevant upstream CPU files match Prism `73bfe6cc485554822f80c473a3c0c4f25e80f41b`, but this package's installer intentionally requires the pinned sudoingX commit. File hashes are in `manifest.json`.

This is an **independent, additive** patch, unlike the cumulative CPU-MTP patch. It touches only `ggml/src/ggml-cpu/arch/x86/quants.c` and adds `ptq1-avx2.h` beside it. It can compose with `prism-cpu-mtp` because their files are disjoint. Do not apply it twice, or treat this package as a replacement for CPU-MTP.

Later, on a reviewed source checkout:

```sh
python3 apply.py /path/to/pinned/llama.cpp
python3 apply.py /path/to/pinned/llama.cpp --apply
```

The first command checks only. The second changes source only. Neither downloads, compiles, runs inference, opens a model or changes profiles. The installer verifies the Git commit, patch hash and input files, refuses an existing new header, and checks patch applicability before applying.

## Checks completed without executing native code

- Generated patch against the locally retained pinned source; checked application, reverse application and expected output hashes on a disposable source fixture.
- Compiled the prepared comparison test to an object with AVX2 enabled, strict warnings and `-ffp-contract=off`. No executable was linked or run.
- Compile-checked the actual patched dispatch fragment as C in AVX2, forced SSE fallback and SSE-only configurations. This uses standalone type/conversion scaffolding, not a full runtime build.

`test-kernel.cpp` contains the exact upstream SSE helper/body as the comparison baseline, plus the candidate header. It is supplied for later execution on compatible hardware. Its fixtures sweep byte-code patterns, all signed activation values, four distinct activation scales, one through three weight blocks and zero-length behavior. Its simplified type/scaling scaffolding is not a substitute for upstream integration tests.

## Required later before adoption

1. Build the complete pinned runtime, with the patch and separately with `GGML_PTQ1_DISABLE_AVX2`. Run applicable upstream quantization/backend tests.
2. Compile and execute the supplied comparison test on AVX2 hardware; add guarded-tail/allocation-boundary checks and sanitizers. Require exact integer group sums and check actual floating outputs; investigate any bit-level differences.
3. Verify actual CPU dispatch/disassembly and the target feature set; preserve a non-AVX2 fallback build.
4. Compare no-MTP and fixed/adaptive MTP on the same prompts, context and generation budget; verify token/action correctness, acceptance, replay, prefix reuse and memory behavior.
5. Benchmark kernel and complete workflows with repeated interleaved runs and sustained laptop thermals. Keep this candidate inactive if it regresses or has unresolved correctness failures.

No new target RAM reserve is selected. The user clarified that the old 5 GiB guard belongs to the preparation laptop, not the 16 GB target. Existing target-profile reserve changes are separate work; this kernel does not change them.

Licensing: upstream llama.cpp MIT terms apply to the source-derived patch/helpers; preserve the source checkout's LICENSE. This package contains no model weights or runtime binaries.
