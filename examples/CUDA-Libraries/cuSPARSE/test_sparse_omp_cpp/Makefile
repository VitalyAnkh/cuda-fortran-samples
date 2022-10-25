################################################################################
#
# Copyright (c) 2020, NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
#
################################################################################

TEST     = tcusparseompc1
CXX      = nvc++
EXE      = exe

CXXFLAGS  = -fast -mp=gpu -cudalib=cusparse,cublas
ifeq ($(TEST),tcusparseompc2)
CXXFLAGS  = -fast -mp=gpu -gpu=managed -cudalib=cusparse,cublas
endif
# curand device code currently only supported with nollvm
ifeq ($(TEST),tcusparseompc3)
CXXFLAGS  = -fast -mp=gpu -cudalib=cusparse,cublas
endif

all: build run verify

build: $(TEST).cpp
	$(CXX) $(CXXFLAGS) -o $(TEST).$(EXE) $<

run: $(TEST).$(EXE)
	$(RUN) ./$(TEST).$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof
