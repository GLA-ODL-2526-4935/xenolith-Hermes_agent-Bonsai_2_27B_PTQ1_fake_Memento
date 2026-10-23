# Default-on persistent source selector

This delta requires exact `intel-source-profile-v2` and is already applied in the
dedicated `prism-cost-aware-source` checkout. The older installed binaries are
not overwritten. Eligible single-slot draft-MTP runs, cap1..4 and minimum<=1, now
enable the suffix/MTP/warm-off selector by default. `LLAMA_SPEC_SOURCE_POLICY=0`
disables it; `LLAMA_SPEC_SUFFIX=0` removes suffix arms. Unsupported configurations
retain their original path unless selection is explicitly forced on.

All enabled requests share a process-local context-bucket cost bank by default.
Without explicit profile paths, this bank ends when the process exits. Setting
identity and profile input/output paths enables persistent disk export/resume;
the authorized benchmark launcher supplies them automatically. No implicit file
writes are introduced into arbitrary llama.cpp applications. Frozen mode remains
input-only, no learning/probes. Existing Xenolith launch configurations remain
explicit choices; this delta changes the dedicated Prism engine defaults.

The Python identity helper now hashes the engine plus adjacent DLL/shared-library
bundle, rather than only the small server loader executable. Old file-only
identities deliberately mismatch and must be regenerated. Hardware/settings/model
compatibility checks and the v2 JSON schema remain. Windows NVIDIA timing costs
must still be reset before calibrating Intel CPU/iGPU.

See [the profile contract](../prism-source-profile/README.md) for algorithms,
controls and fixtures. Inspired by [BanditSpec](https://arxiv.org/abs/2505.15141)
and [SuffixDecoding](https://arxiv.org/abs/2411.04975); still a bounded adaptation,
not a full implementation or a guarantee from either paper.

`apply.py SOURCE` checks; `--apply` installs this exact delta and `--reverse`
removes it. No build/model operation is performed by that helper.
