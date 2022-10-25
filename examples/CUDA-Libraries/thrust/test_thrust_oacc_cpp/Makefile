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

TEST 	 = tsort6
NVXX     = nvcc
CXX      = nvc++
EXE	 = exe

CUFILE = keys.cu
# nvcc requires specifying compute capability to generate code for. The following logic
# attempts to figure this out automatically by looking through GPUs on current systems.
# However, it if fails to do so it may need done manually like the example.
# Example for Ampere: --generate-code arch=compute_80,code=sm_80
SYSCOMPCAP := $(shell nvaccelinfo | grep "Default Target:" | awk -F "cc" '{ print $$2 }' | sort | uniq)
NVCCOPTIONS := $(foreach CCAP,$(SYSCOMPCAP), --generate-code arch=compute_$(CCAP),code=sm_$(CCAP))

UNAME := $(shell uname -a)
ifeq ($(findstring ppc64, $(UNAME)), ppc64)
NVXXFLAGS  = -c $(NVCCOPTIONS)
else
NVXXFLAGS  = -ccbin nvc++ -c $(NVCCOPTIONS)
endif

# These are compatible flags between NVHPC and nvcc
ifeq ($(TEST),tsort6)
CXXFLAGS   = -acc=gpu -gpu=nordc -cudalib=curand
else
CXXFLAGS   = -acc=gpu -gpu=nordc,managed -cudalib=curand
endif

all: build run verify

build: $(TEST).cpp
	$(NVXX) $(NVXXFLAGS) $(CUFILE) -o keys.o
	$(CXX) $(CXXFLAGS) keys.o -o $(TEST).$(EXE) $<

run: $(TEST).$(EXE)
	$(RUN) ./$(TEST).$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof *.o
