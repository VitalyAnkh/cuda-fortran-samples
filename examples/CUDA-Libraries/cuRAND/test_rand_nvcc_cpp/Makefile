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

TEST 	 = trandnv1
CXX      = nvcc
EXE	 = exe

# curand device code currently only supported with nollvm
ifeq ($(TEST),trandnv1)
CXXFLAGS  = -ccbin nvc++ -cudalib=curand
else
CXXFLAGS  = -ccbin nvc++
endif

all: build run verify

build: $(TEST).cu
	$(CXX) $(CXXFLAGS) -o $(TEST).$(EXE) $<

run: $(TEST).$(EXE)
	$(RUN) ./$(TEST).$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof
