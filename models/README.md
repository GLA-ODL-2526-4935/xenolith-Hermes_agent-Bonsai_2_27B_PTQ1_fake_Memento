# Local Bonsai model

Place `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf` in this directory. The local Windows preparation checkout contains the verified file; Git clones do not include it. See `bonsai-model.json` for the exact size and SHA256. The original community Lean file includes its bundled MTP components; this is not a separately trained replacement head.

Source: https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF
Parent: https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
Consult the source and parent model cards for applicable licenses and notices before redistributing weights.

GGUF files are ignored by Git. Transfer the weights separately to the Linux laptop, or obtain the exact matching artifact from the source. A source-code push does not transfer the model. Keep hardware/model setup gates in `../Agent.md`; this move does not install or start inference.

For initial correctness checks use this base with MTP disabled. The file format requires a compatible Prism runtime; ordinary upstream llama.cpp is not interchangeable. Set the model path explicitly in your local launch configuration after Linux hardware checks. No launch configuration is changed by placing the file here.

## Locally trained replacement MTP head

See [model card](MODEL_CARD.md) and [artifact manifest](trained-mtp.json). The quantized head ZIP is tracked through Git LFS; the BF16 checkpoint stays
local. Install Git LFS and run `git lfs pull` after cloning to fetch the ZIP. The ordinary pinned base and
MTP-off launcher are unchanged.
