# Bounded suffix workspace increment

23 October 2026. Source-only additive experiment for the external Prism Bonsai backend.
The installed server, native Xenolith Gemma graph, PTQ1 weights/kernels, exact
MTP verifier/rollback and Intel CPU-decode/iGPU-prefill placement are unchanged.

## What changed

The legacy suffix function allocates history, match, vote and result vectors on
each lookup. This increment adds an optional path that:

- Reuses match/vote scratch across suffix cycles within one request.
- Reads the last 8,191 committed prompt tokens plus the last token through a
  synchronous borrowed view, eliminating the up-to-32-KiB history copy per lookup.
- Fills the caller's existing result vector, at most four draft tokens.
- Releases scratch at request end/cancellation via the existing end hook and
  defensively at the next begin. Destructor ownership remains the final fallback.

For 32-bit llama_token and 64-bit size_t, requested scratch capacity is 96 KiB
(8,192 positions plus 8,192 votes), allocated lazily on first useful lookup.
The tiny output buffer is caller-owned; allocator metadata/capacity rounding is
additional. No global pool, retained cross-request token history, borrowed pointer
cache or second model representation was added. Release returns storage to the
C++ allocator; it does not promise OS RSS reduction or secure memory erasure.

Longest-match order, minimum width 2, maximum width 16, majority threshold,
exhausted-occurrence denominator, token tie-breaking and maximum draft length 4
are unchanged. Every candidate still goes through the existing target verifier.
The compatibility alias path calls the original helper. Prompt borrowing is valid
only during this synchronous lookup; it must not be reused for async backends.

## Enable and composition

Default is off. After building this exact source, opt in with:

```sh
LLAMA_SPEC_SUFFIX_WORKSPACE=1 /path/to/llama-server <existing arguments>
```

Unset it or use 0 to retain the legacy suffix path. This does not change source
selection choices or their default policy. The recent three-arm selector test is
an isolated benchmark runtime, not silently merged into this source package.

The existing Xenolith source-profile environment digest includes this switch;
create/recalibrate a matching identity when changing it. Standalone wrappers must
also bind it into their settings hash. Never reuse NVIDIA costs on Intel.

This delta requires the current composed source at
`ff414120c343e6e6cb868013c99f1dde52b27e70` **with runtime-lifecycle applied**.
The exact prerequisites in manifest.json are authoritative. It cannot be stacked
on arbitrary clean upstream or older cumulative patches.

```sh
python3 apply.py /path/to/composed-source            # check only
python3 apply.py /path/to/composed-source --apply    # source only
python3 apply.py /path/to/composed-source --reverse  # undo this first
```

Undo this increment before undoing runtime-lifecycle or earlier packages. HEAD,
all input/dependency hashes and patch digest are guarded. Drift must be reviewed,
not forced or reset. The tool neither builds nor launches inference.

## Validation and acceptance checklist

```sh
python3 test.py --source /path/to/applied-source --build-dir /writable/test-output
```

- [x] 836,772 exact draft-vector parity cases against the actual legacy helper:
  exhaustive small alphabets, random histories through 17K tokens, retained-window
  boundaries, repeated histories, invalid/short caps and input/output alias cases.
- [x] Four independent workspaces exercise concurrent ownership without a shared pool.
- [x] After warm-up, zero temporary allocations over 1,000 lookups per fixture,
  compared with 4,000 in the legacy fixture; scratch release/reuse tested.
- [x] Full edited speculative.cpp compiles. Actual begin/end hook fixtures test
  request release, sequence isolation and profile-save exception cleanup.
- [x] Check-only, wrong-HEAD/drift refusal and real patch apply/reverse fixture.
- [ ] Full linked Linux CPU/Vulkan build and real-model target/logit/action parity.
- [ ] Target Intel cold/warm lookup cost, allocator/RSS behavior, full task latency,
  tokens/s and selector source costs. Short requests pay initial scratch allocation.

The bundled microbenchmark is a component test on the preparation host, not an
Intel deployment benchmark. It times lookup only; it does not measure actual DRAM
controller traffic or establish any end-to-end gain. Allocation counts exclude
model/backend allocations. Numerical model parity and GPU/OS compatibility are
not certified by these tests. No inference was started for this increment.

## Source references and overlap decisions

Independently implemented; no source from these engines was copied into the patch.
Their current pinned code was inspected for ownership/lifetime structures:

- [MLC-LLM draft workspace manager](https://github.com/mlc-ai/mlc-llm/blob/8978ea9aec626d64096aaa92fbc89bda2e3ea802/cpp/serve/draft_token_workspace_manager.cc):
  bounded reusable speculative workspace and explicit slot ownership. Adopted the
  ownership principle, not full-vocabulary probability arrays, trees or TVM.
- [OpenVINO CPU memory control](https://github.com/openvinotoolkit/openvino/blob/695d69efe543b9e6c75361bc5d3df9c866d9c6ed/src/plugins/intel_cpu/src/memory_control.cpp):
  reusable memory blocks with explicit release. Adapted locally; graph allocation
  reuse is already present in GGML, so no competing allocator was introduced.
- [ONNX Runtime execution frame](https://github.com/microsoft/onnxruntime/blob/84829e76d4fbb4a7258202f777fa4ff0876a3564/onnxruntime/core/framework/execution_frame.cc):
  known-lifetime buffer planning/fallback. Adopted bounded caller-owned scratch;
  no process-wide arena or allocator dependency.
- [XNNPACK memory planner](https://github.com/google/XNNPACK/blob/26186f85939753b9a509734b193aad2bcb17518f/src/memory-planner.c):
  lifecycle-aware reuse overlaps GGML. Kept as an audit reference, not duplicated.
- [KTransformers AVX2 MXFP4 kernel](https://github.com/kvcache-ai/ktransformers/blob/a694e718e8b3ec2e221277d674e369f3be38d2a4/kt-kernel/operators/avx2/mxfp4-moe.hpp):
  native-lane activation ordering and shape-dependent staging are future profiling
  candidates. Existing PTQ1 batch/LUT conversion already overlaps; MXFP4/MoE code
  is not compatible with PTQ1 scales/Hadamard and was not ported.
- [mistral.rs recurrent prefix cache](https://github.com/EricLBuehler/mistral.rs/blob/012b9bec941e62eecf6acc0f5a8a3e696f7bf906/mistralrs-core/src/prefix_cacher.rs):
  separate KV/recurrent ownership and capacity accounting. Existing checkpoint and
  cache gates overlap. Full paged-prefix migration needs backend parity and is deferred.

Existing upstream MIT notices and license remain intact. This independent helper
uses MIT. reviewed-sources.json records source revisions and fetched files; the
full review is engine/docs/ADDITIONAL_ENGINE_REVIEW.md. Prior paper references
remain in engine/README.md and older source packages.
