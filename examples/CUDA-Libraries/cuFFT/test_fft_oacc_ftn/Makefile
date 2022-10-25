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

TEST 	 = tcufft2df1
FC       = nvfortran
EXE	 = exe

FCFLAGS  = -fast -acc=gpu -cudalib=cufft
ifeq ($(TEST),tcufft2df2)
FCFLAGS  = -fast -acc=gpu -gpu=managed -cudalib=cufft
endif
ifeq ($(TEST),tcufft2df3)
FCFLAGS  = -fast -acc=gpu -gpu=managed -cuda -cudalib=cufft
endif

all: build run verify

build: $(TEST).f90
	$(FC) $(FCFLAGS) -o $(TEST).$(EXE) $<

run: $(TEST).$(EXE)
	$(RUN) ./$(TEST).$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof
