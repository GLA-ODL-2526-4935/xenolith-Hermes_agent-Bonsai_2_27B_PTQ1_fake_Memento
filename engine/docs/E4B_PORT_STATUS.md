# Gemma 4 E4B port status

> Historical native-port scope. The active project candidate is now Bonsai 2 27B via the additional Prism backend; see v5 plan (local development notes; excluded from source distribution) and [backend status](BONSAI_BACKEND_STATUS.md). Statements below about pending training reflect 20 October and are not the current deployment prerequisite. Native E4B validation remains unfinished.

Updated: 20 October 2026. Engine changes apply only to this combined fork.
Baseline upstream commit: `88e8661d190985c50c638049dd2fec195a0fec63`.

## Implemented, not model-validated

- E4B-only text backbone: 42 layers, hidden 2,560, dense FFN 10,240,
  eight query heads, two KV heads, local head width 256 / global width 512.
  The old model selector, MoE routing, expert weights and expert GPU kernels
  have been removed from the active engine path.
- Strict GGUF name, type, shape and required-tensor validation. Q4_0 matrices
  and F32 norms only for this baseline. Reject unsupported formats rather than
  silently reinterpreting QAT tensors. The exact training/export artifact still
  needs a pinned conversion manifest.
- PLE tensors and complete computation: scaled token lookup, model projection,
  per-layer normalization, mixing, gated GELU projection and residual injection.
  The approximately 1.48 GiB Q4_0 token PLE table stays in the file mapping;
  only the selected row is decoded. Other matrices use shared Q4_0 repacking.
- Dense GELU-gated FFN and post-FFN normalization; optional output scaling;
  tied output embeddings or an explicit output matrix; logit softcap.
- Independent K/V projections, unweighted V normalization, GQA groups of four,
  local causal 512-token attention and global attention without query scaling.
- NeoX half-pair RoPE: local base 10,000; global base 1,000,000 with the
  converter's proportional/partial rotation factors. The GGUF rope factor tensor
  is required and must match the forthcoming export.
- Shared-KV tail: layers 0–23 own KV; layers 24–41 read donor 22 (local) or
  23 (global). Only 20 local + four global donor caches are allocated, not
  35 + seven independent caches. Sizes follow the requested context capacity.
- Donor-aware COW cache reads/writes, local cache mirrors, anchors and snapshot
  payload dimensions. E4B layout fingerprints reject old-model snapshots.
- E4B CPU projection shapes and a new generic Level Zero Q4_0/Q8 kernel.
  Prefill is deliberately token-serial across layers with synchronized GPU
  projections and CPU attention/normalization. It is not optimized batched GEMM.
- Diagnostic final-logit and last-token layer-state dumps through the existing
  oracle command. F32-activation mode bypasses activation quantization but
  shares this graph and half-precision KV; it is NOT an independent oracle.
- Descriptor-only test source covers physical donors, shared tail and widths.
  It was not executed. The Makefile no longer advertises obsolete private
  MoE/prefill tests or benchmarks as E4B validation.

## Still unfinished

- Compile/link verification on compatible Linux development hardware.
- Independent deterministic comparisons with the same model in HF/llama.cpp:
  forward states, logits, generation, CPU versus GPU projections, prefill versus
  decode, shared donors, ring rollover, long positions, COW promotion, rewind,
  snapshot restore, cancellation and multi-turn sessions.
- Actual RAM/latency measurements on the target 16 GB Intel laptop. The model's
  maximum 131,072 context is not a memory/performance guarantee.
- Optimized batch prefill, attention kernels and scheduling. The baseline
  shadow-start call completes its work synchronously; asynchronous progress
  and cancellation latency require a later scheduling pass.
- The final fine-tuned checkpoint, QAT-to-GGUF layout, exact tokenizer/merge
  manifest and chat-template fingerprint. The serving profile still pins its
  inherited template and must not be treated as verified for the new model.
- External model-specific tests belong in the user's separate testing folder.
  Historical upstream test/bench sources remain for reference, not as
  passing E4B tests.
- OpenAI-compatible bridge, Hermes integration, staged tool discovery,
  Memento delimiter/context policy and physical KV eviction.
- Vision/audio towers: this implementation is text-only.

## Safety and verification

`runtime_ready = 0` remains the default validation gate. The execution code is
implemented, but full opening stops before GPU initialization unless the separate
validation environment explicitly sets `XE_ALLOW_UNVALIDATED_E4B=1`. That opt-in
does not certify correctness. Do not enable this on the current laptop.

Only static source checks were performed here: C preprocessing/parsing, internal
call/member references and targeted source review. No compilation, linking,
engine/model execution, numerical tests or GPU access occurred.

## Architecture references

- [Google E4B config](https://huggingface.co/google/gemma-4-E4B/blob/main/config.json)
- [Transformers Gemma 4](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
- [llama.cpp Gemma 4 graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4.cpp)
- [llama.cpp Gemma conversion](https://github.com/ggml-org/llama.cpp/blob/master/conversion/gemma.py)

These are architecture references, not a substitute for pinning the actual
fine-tuned checkpoint and converter revisions.
