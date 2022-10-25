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
BUFSIZE = 100000000
UNAME := $(shell uname -a)
MPICC   = mpicc
RUN     = mpirun
OBJ	= o
EXE	= out
CCFLAGS = -fast -DHAVE_STDLIB_H
MPIOPT  =
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
      MPICC   = nvc
      RUN     = mpiexec
      OBJ	= obj
      EXE	= exe
      CCFLAGS	+= -stack:10000000
      MPIOPT  = -Mmpi=msmpi
endif

all: build run verify

build: copytest.c copytest2.c duff.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -o copytest.$(EXE) copytest.c
	$(MPICC) $(MPIOPT) $(CCFLAGS) -o copytest2.$(EXE) copytest2.c

run: copytest.$(EXE) copytest2.$(EXE)
	@echo "------------------- Running copytest.$(EXE) ---------------------------"
	$(RUN) -np $(NPROCS) ./copytest.$(EXE) $(BUFSIZE)
	@echo "------------------- Running copytest2.$(EXE) ---------------------------"
	$(RUN) -np $(NPROCS) ./copytest2.$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
