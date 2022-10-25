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

TEST 	 = tcufft2dc3
CXX      = nvc++
EXE	 = exe

ifeq ($(TEST),tcufft2dc3)
CXXFLAGS  = -fast -acc=gpu -cudalib=cufft
else
CXXFLAGS  = -fast -acc=gpu -gpu=managed -cudalib=cufft
endif

all: build run verify

build: $(TEST).cpp
	$(CXX) $(CXXFLAGS) -o $(TEST).$(EXE) $< $(LIBR)

run: $(TEST).$(EXE)
	$(RUN) ./$(TEST).$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof
