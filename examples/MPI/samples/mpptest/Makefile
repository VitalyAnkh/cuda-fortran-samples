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
OBJ     = o
EXE	= out
MPIOPT  =
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
MPICC   = nvc
RUN	= mpiexec
OBJ	= obj
EXE	= exe
MPIOPT  = -Mmpi=msmpi
endif
CCFLAGS = -fast

all: build run verify

build: mpptest.c gopf.c grph.c ovlap.c pair.c pattern.c util.c getopts.c tstauto.c rate.c mpe_seq.c copy.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c gopf.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c grph.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c ovlap.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c pair.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c pattern.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c util.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c getopts.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c tstauto.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c rate.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c mpe_seq.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -c copy.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -o mpptest.$(EXE) mpptest.c gopf.$(OBJ) grph.$(OBJ) ovlap.$(OBJ) pair.$(OBJ) pattern.$(OBJ) util.$(OBJ) getopts.$(OBJ) tstauto.$(OBJ) rate.$(OBJ) mpe_seq.$(OBJ) copy.$(OBJ)

run: mpptest.$(EXE)
	$(RUN) -np $(NPROCS) ./mpptest.$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
