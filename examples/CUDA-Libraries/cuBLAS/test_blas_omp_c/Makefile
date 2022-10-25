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

TEST     = tblasxtomp
CC       = nvc
EXE      = exe

ifeq ($(TEST),tblasxtomp)
CCFLAGS  = -fast
else
CCFLAGS  = -fast -mp=gpu
endif

LIBR     = -cudalib=cublas

all: build run verify

build: $(TEST).c
	$(CC) $(CCFLAGS) -o $(TEST).$(EXE) $< $(LIBR)

run: $(TEST).$(EXE)
	$(RUN) ./$(TEST).$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof
