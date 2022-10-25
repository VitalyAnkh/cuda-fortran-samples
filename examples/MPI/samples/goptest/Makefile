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
RUN     = mpirun
OBJ	= o
EXE	= out
MPIOPT  =
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
MPICC   = nvc
MPIOPT  = -Mmpi=msmpi
RUN     = mpiexec
OBJ	= obj
EXE	= exe
endif
CCFLAGS = -fast

all: build run verify

build: goptest.c getopts.c gopf.c grph.c rate.c util.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c gopf.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c grph.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c rate.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c util.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c getopts.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -o gop.$(EXE) goptest.c getopts.$(OBJ) gopf.$(OBJ) grph.$(OBJ) rate.$(OBJ) util.$(OBJ)

run: gop.$(EXE)
	$(RUN) -np $(NPROCS) ./gop.$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
