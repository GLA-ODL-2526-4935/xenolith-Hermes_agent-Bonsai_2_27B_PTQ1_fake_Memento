# Source distribution and separate model artifacts

## Included in the source tree

- The modified Hermes source, local memory/review/controlled-workflow extensions and tests; upstream package metadata, lockfiles and licenses remain present.
- Combined startup source and blank `launch.example.json`; no personal profile or credentials.
- Native Xenolith source, external Prism process launcher, profiles, pinned patch packages and their build/application instructions. Native Xenolith is not a Bonsai runtime.
- Model identity metadata, trained-head model card, checksums, license notices, exact reconstruction utility and synthetic artifact tests in `models/`.

The project vendors Hermes source as ordinary tracked files, not an empty Git submodule. Source files currently uncommitted must be included in the intended development commit before pushing. A push does not transmit working-tree-only edits.

## Kept local, excluded from future commits

Base/reconstructed GGUFs, ignored contents of `models/artifacts/` (except the explicitly LFS-tracked quantized ZIP), existing planning/checklist files, engine roadmap/design candidate notes, Windows RTX5080 calibration records, generated microbenchmark results, caches, environments, logs, training checkpoints, output/reports directories and machine-specific launch settings. Existing local files are preserved. Removing a tracked local report from the index stages its deletion from the next source revision; it does not erase older Git history.

Retain upstream optional components, test fixtures and dependency locks. They are source dependencies or upstream project material, rather than personal run outputs; stripping them merely to reduce file count risks breaking supported paths.

## Separate release assets

`models/artifacts/bonsai-hermes-mtp-step1800-bf16.safetensors` contains the fifteen BF16 MTP-only training tensors. `models/artifacts/bonsai-hermes-mtp-step1800-mixed-q6k-q8_0.zip` contains the quantized replacement head plus reconstruction metadata and license notices. The quantized ZIP is included through Git LFS; the BF16 checkpoint remains local. Neither belongs in ordinary Git blobs. Distribute the ZIP with `models/MODEL_CARD.md`, `models/trained-mtp.json`, `models/SHA256SUMS` and `models/licenses/`.

The original 6.30 GB base remains available from its upstream source. Its hash is pinned in `models/bonsai-model.json`. The quantized ZIP has an exact Git LFS tracking rule; after source publication, clone with Git LFS and run `git lfs pull`. The BF16 checkpoint is not required for inference. No push has been performed yet.

## External requirements and current gaps

This is a source distribution, not an offline installation bundle. Mint needs the Hermes dependencies from its documented bootstrap, a compatible separately built/installed Linux Prism runtime with matching libraries, and the model weights. The relevant runtime source revisions and patch composition constraints are in the patch READMEs. Windows binaries/environments must not be copied as Linux runtimes.

Begin with the original CPU MTP-off baseline after checking the target. The trained-head replacement has been reconstructed byte-for-byte, but the existing launcher still intentionally rejects its different digest. A reviewed trained-artifact configuration is required before enabling it through that launcher. Do not edit the original expected checksum to bypass this boundary.

The 5 GiB RAM reserve in current profiles/validation is still the preparation-machine setting; the user has not selected a replacement reserve for the 16 GB Mint target. Live Brave research integration, containment, complete end-to-end workflow acceptance, Intel CPU/iGPU parity and speed remain unfinished. Window4 trained-head greedy divergence remains unresolved. Source/package checks do not certify these features.

No installation, model execution, remote transfer, Git commit, push or release publication is performed by preparing this distribution.
