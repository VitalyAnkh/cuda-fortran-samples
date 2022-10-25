################################################################################
#
# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
#
################################################################################

TEST 	 = lu_fallback
FC		 = nvfortran
EXE		 = exe

FCFLAGS  = -cuda -gpu=nvlamath -cudalib=nvlamath -O2

all: build run verify

build: $(TEST).F90
	$(FC) $(FCFLAGS) -o $(TEST).$(EXE) $<

run: $(TEST).$(EXE)
	export NV_LAMATH_FALLBACK=1; \
	$(RUN) ./$(TEST).$(EXE)
	@echo 'This is a negative example. It is supposed to print error messsages and terminate the application.'
	$(RUN) ./$(TEST).$(EXE)

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof
