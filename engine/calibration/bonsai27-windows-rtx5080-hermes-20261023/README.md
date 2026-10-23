# Calibration archive - Windows RTX 5080 Laptop GPU

Real numeric online-selector observations, not model/head training weights.
See RESULTS.md for the bounded-reasoning protocol, speed and correctness limits.
This archive is deliberately not referenced by any active Intel launch profile.
The v2 profile can be consumed by Xenolith's external Prism backend only with
matching runtime bundle, model/head, hardware and settings identity.
For Fedora Intel CPU/iGPU, create a fresh local identity and use the profile
helper's retarget command: it clears the incompatible NVIDIA timing counters.
Native Xenolith's Gemma C graph does not consume these counters.
