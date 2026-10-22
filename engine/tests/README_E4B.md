# Test scope after the E4B replacement

Only `test_model_profiles.c` is currently exposed by `make check-layout`.
It checks descriptor and donor layout without a model or GPU; it has not
been run on this machine.

Other inherited tests and benchmarks use former private internals, dimensions,
kernel names or old-model fixtures. They are historical source, not an E4B
passing suite, and may no longer compile.

Model-specific correctness tests will live in the user's separate testing
folder. The required validation matrix is in `docs/E4B_PORT_STATUS.md`.
