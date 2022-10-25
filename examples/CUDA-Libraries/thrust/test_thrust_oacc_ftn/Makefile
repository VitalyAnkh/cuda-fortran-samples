################################################################################
#
# Copyright (c) 2017, NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
#
################################################################################

TEST 	 = tsort3
FC       = nvfortran
CXX      = nvcc
EXE	 = exe

CUFILE = keys.cu
# nvcc requires specifying compute capability to generate code for. The following logic
# attempts to figure this out automatically by looking through GPUs on current systems.
# However, it if fails to do so it may need done manually like the example.
# Example for Ampere: --generate-code arch=compute_80,code=sm_80
SYSCOMPCAP := $(shell nvaccelinfo | grep "Default Target:" | awk -F "cc" '{ print $$2 }' | sort | uniq)
NVCCOPTIONS := $(foreach CCAP,$(SYSCOMPCAP), --generate-code arch=compute_$(CCAP),code=sm_$(CCAP))

# These are compatible flags between NVHPC and nvcc
CXXFLAGS  = -c $(NVCCOPTIONS)
ifeq ($(TEST),tsort3)
FCFLAGS   = -acc=gpu -gpu=nordc -pgc++libs
else
FCFLAGS   = -acc=gpu -gpu=nordc,managed -pgc++libs
endif

# We use the cudafor curand module, so need all these
LIBR     = -cuda -cudalib=curand
all: build run verify

build: $(TEST).f90
	$(CXX) $(CXXFLAGS) $(CUFILE) -o keys.o
	$(FC) $(FCFLAGS) keys.o -o $(TEST).$(EXE) $< $(LIBR)

run: $(TEST).$(EXE)
	$(RUN) ./$(TEST).$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof *.o
