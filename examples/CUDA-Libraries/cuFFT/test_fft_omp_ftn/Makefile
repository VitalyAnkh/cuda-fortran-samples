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

TEST     = tcufft2dompf1
FC       = nvfortran
EXE      = exe

ifeq ($(TEST),tcufft2dompf1)
FCFLAGS  = -fast -mp=gpu -cudalib=cufft
else ifeq ($(TEST),tcufft2dompf2)
FCFLAGS  = -fast -mp=gpu -gpu=managed -cudalib=cufft
else
FCFLAGS  = -fast -mp=gpu -gpu=managed -cudalib=cufft -cuda
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
