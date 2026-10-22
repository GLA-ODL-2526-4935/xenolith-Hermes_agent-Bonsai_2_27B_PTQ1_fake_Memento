# E4B-only build. No model execution is part of the default target.
CC=gcc
BASE_CFLAGS=-O3 -march=native -std=c11 -Wall -Wextra -pthread
BUILD_COMMIT=$(shell git rev-parse --short=9 HEAD 2>/dev/null || printf unknown)
NUMERIC_SOURCE_HASH=$(shell sha256sum xenolith.c xenolith.cl | sha256sum | cut -d' ' -f1)
NUMERIC_CFLAGS_HASH=$(shell printf '%s' '$(BASE_CFLAGS)' | sha256sum | cut -d' ' -f1)
CFLAGS=$(BASE_CFLAGS) -DXE_BUILD_COMMIT='"$(BUILD_COMMIT)"' \
 -DXE_NUMERIC_SOURCE_HASH='"$(NUMERIC_SOURCE_HASH)"' \
 -DXE_NUMERIC_CFLAGS_HASH='"$(NUMERIC_CFLAGS_HASH)"'
LDLIBS=-lm -lze_loader
GPU_SPV=xenolith_gpu.spv
GPU_OBJ=xenolith_gpu_spv.o
OBJECTS=main.o xenolith.o format.o json.o profile.o kvstore.o conversation.o runtime.o serve.o

xenolith: $(OBJECTS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJECTS) $(GPU_OBJ) $(LDLIBS)

$(OBJECTS): xenolith.h
xenolith.o: xenolith_internal.h models/gemma4_arch.h format.h
main.o: profile.h serve.h conversation.h kvstore.h
format.o: format.h
json.o: json.h
profile.o: profile.h format.h json.h
kvstore.o: kvstore.h xenolith_internal.h format.h
conversation.o: conversation.h kvstore.h format.h
runtime.o: runtime.h xenolith_internal.h kvstore.h conversation.h profile.h format.h json.h
serve.o: serve.h runtime.h kvstore.h conversation.h profile.h json.h

$(GPU_SPV): xenolith.cl
	ocloc compile -file $< -device xe-lp -spv_only -output xenolith_gpu \
	 -output_no_suffix -out_dir . -options '-cl-std=CL3.0' -q
$(GPU_OBJ): $(GPU_SPV)
	ld -r -b binary -o $@ $<
	objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents $@

# This target is optional, CPU-only, and does not load model weights.
tests/test_model_profiles: tests/test_model_profiles.c models/gemma4_arch.h
	$(CC) -O2 -std=c11 -Wall -Wextra -I. -o $@ $<
check-layout: tests/test_model_profiles
	./tests/test_model_profiles

# Legacy tests/benchmarks are retained as historical source only. They depend
# on removed private MoE/prefill internals and are not E4B validation targets.
# Numerical/model validation belongs in the user's separate model folder.
clean:
	rm -f $(OBJECTS) $(GPU_OBJ) $(GPU_SPV) xenolith tests/test_model_profiles
.PHONY: clean check-layout

# Python-only backend contract checks; never loads models or invokes inference.
PYTHON ?= python3
check-backends:
	$(PYTHON) -B -m unittest discover -s tests -p test_prism_backend.py -v
.PHONY: check-backends
