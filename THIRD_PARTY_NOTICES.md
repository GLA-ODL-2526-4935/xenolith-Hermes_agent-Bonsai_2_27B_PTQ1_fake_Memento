# Licensing and upstream attribution

This is a combined source experiment, not a blanket relicensing of its contents.
The root [LICENSE](LICENSE) licenses original project contributions under MIT.
Upstream notices remain applicable to inherited code and modifications. Preserve
those notices in copies and distributions; a new MIT notice does not remove them.

| Component | Source / terms | Treatment here |
| --- | --- | --- |
| Xenolith engine | [simoneiacomino/xenolith](https://github.com/simoneiacomino/xenolith), MIT; Copyright2026 Simone Iacomino | Original text retained in [engine/LICENSE.md](engine/LICENSE.md). |
| Hermes Agent | [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent), MIT; Copyright2025 Nous Research | Original text retained in [Hermes_agent/LICENSE](Hermes_agent/LICENSE). |
| New integration, local extensions, tests and documentation | Copyright2026 GLA-ODL-2526-4935 and respective contributors, MIT | Root LICENSE applies to original additions; inherited portions keep upstream notices. |
| PrismML llama.cpp runtime | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp/blob/master/LICENSE), MIT; ggml authors and other applicable upstream notices | Separately installed; retain the exact selected release/source license and third-party notices. Optional sudoingX branch patches do not replace its upstream license. |
| Bonsai 2 27B weights | [PrismML model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), Apache-2.0 | Not supplied/relicensed by this project's MIT license. Download only on user choice. |
| Community PTQ1 MTP Lean weights | [sudoingx model card](https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF), Apache-2.0 label; parent/donor provenance in that repository | Preserve exact model/parent/donor LICENSE/NOTICE files and verify obligations for the pinned artifact before redistribution. |
| MEMENTO research | [paper](https://arxiv.org/abs/2604.09852), [Microsoft research implementation](https://github.com/microsoft/memento) | Research reference, not vendored/claimed as an implemented trained method. Any future imported code/data needs its own license review. |

Model weights, datasets, browser binaries, drivers, libraries and managed Hermes
dependencies are separate works. Their licenses may differ from MIT and must be
checked for the exact installed versions. A model card label alone is not a full
redistribution inventory. For Apache-2.0 artifacts, retain applicable license,
NOTICE and attribution files and document changes as required by those terms.
If the user supplies another engine/model, record its actual terms instead.

No weights or external runtime binaries are added by this documentation change.
Before publishing a binary/model bundle, produce a pinned component/license
manifest; this notice is not a claim that all dependencies have been audited.
Project names and links acknowledge provenance, not endorsement or trademark rights.
