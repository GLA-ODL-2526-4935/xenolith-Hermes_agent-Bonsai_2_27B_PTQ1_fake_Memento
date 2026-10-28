# Bonsai Hermes MTP step1800 — experimental trained head

This is a locally adapted MTP head for the exact community Bonsai 2 27B PTQ1 Lean
backbone identified in `bonsai-model.json`, not a separately trained full model.
The original public base already includes its original MTP head; these artifacts
replace that head. The base backbone remains byte-identical.

## Files and usage

- `bonsai-hermes-mtp-step1800-bf16.safetensors`: 15 BF16 `mtp.*` training tensors,
  with no backbone, embedding or output matrix. This is a training checkpoint,
  not a directly loadable llama.cpp model or a complete optimizer resume state.
- `bonsai-hermes-mtp-step1800-mixed-q6k-q8_0.zip`: quantized head payloads plus the
  tested GGUF header/layout. It contains no backbone tensor payloads and is not
  directly loadable. Q6_K matrices, Q8_0 K/V matrices and F32 norms match the tested
  export. The ZIP is a distribution artifact; do not pass it as a draft model.
- `trained-mtp.json` and `SHA256SUMS`: artifact identity, training metrics,
  tensor types, limitations and exact reconstructed-model digest.

Download/transfer the exact base and release ZIP separately. From the repository root:

```sh
python3 models/mtp_artifact.py apply \
  --base models/Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf \
  --package models/artifacts/bonsai-hermes-mtp-step1800-mixed-q6k-q8_0.zip \
  --output models/Bonsai-Hermes-MTP-step1800-mixed.gguf
```

Python 3.11+ standard library only. Keep about 6.3 GB additional disk space for
the reconstructed combined GGUF. The utility refuses to overwrite existing files,
verifies the base and reconstructed SHA256, and does not start inference. The
result is one combined Lean model with shared embeddings/output weights rather
than a second full draft model. A full reconstruction was verified byte-for-byte.

The existing engine launcher intentionally pins the original base digest. It will
reject the trained result. Do not overwrite its expected hash or claim the trained
head is integrated: a separately reviewed trained-artifact configuration and
target validation are still required. The CPU MTP-off baseline remains unchanged.

## Training and evaluation

Step1800 continued from step720 for 1080 updates, three epochs, using 100 new
synthetic Hermes workflow tasks plus 20 replay tasks. Frozen packed PTQ1 backbone
features and model-generated traces were used; only the MTP head was updated.
Validation was checked every50 updates with patience3. Best validation KL was
0.1413530630990863; top1 agreement was0.9109375. These are head-distillation metrics,
not task correctness or broad model capability scores. See the manifest for details.

Windows RTX5080 testing found60/60 full-token agreement at window3 on development
snapshots and12/12 on fresh related synthetic snapshots. Window2 passed30/30 on
the development suite; fresh window2 testing remains pending. Window4 agreed on
only1/30 full sequences and is unsuitable for a strict greedy-parity default.
More training did not demonstrate a stable speed improvement over step720.
These results do not establish CPU speed or Intel iGPU correctness. Real browser
workflows, sustained thermals and end-to-end Mint deployment remain unvalidated.

## Provenance and terms

Community source: https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF
Backbone: https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
MTP donor identified by that community source: https://huggingface.co/Qwen/Qwen3.8-27B

The source model cards label these weights Apache-2.0. Applicable upstream LICENSE
and NOTICE files are preserved in `licenses/`; they are separate from the source
project's MIT license. Modification: MTP-only local distillation and mixed head
quantization; no backbone retraining. Packaging-date source revisions are recorded
without claiming they were the original download revisions. This is an independent
experimental derivative, not an official PrismML, Qwen or sudoingx release.

The quantized ZIP in `models/artifacts/` is explicitly tracked through Git LFS.
The BF16 training checkpoint remains local and ignored.
No push has been performed yet. Clone with Git LFS and run `git lfs pull` to obtain
the quantized ZIP after publication. The BF16 checkpoint is optional for inference.
Keep this card, the manifest, checksums and license notices with the head.
