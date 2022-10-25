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

TEST 	 = tcusparse3
FC       = nvfortran
EXE	 = exe

ifeq ($(TEST),tcusparse3)
FCFLAGS  = -Mpreprocess -fast -acc=gpu -cudalib=cusparse
endif
ifeq ($(TEST),tcusparse4)
FCFLAGS  = -Mpreprocess -fast -acc=gpu -gpu=managed -cudalib=cusparse
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
