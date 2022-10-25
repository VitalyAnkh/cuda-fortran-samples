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

NPROCS  = 2
UNAME := $(shell uname -a)
MPICC   = mpicc
RUN	= mpirun
OBJ	= o
EXE	= out
MPIOPT  =
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
MPICC	= nvc
RUN	= mpiexec
OBJ	= obj
EXE	= exe
MPIOPT  = -Mmpi=msmpi
endif
CCFLAGS = -fast -DHAVE_STDLIB_H

all: build run verify

build: buflimit.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -o buflimit.$(EXE) buflimit.c

run: buflimit.$(EXE)
	$(RUN) -np 2 ./buflimit.$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
