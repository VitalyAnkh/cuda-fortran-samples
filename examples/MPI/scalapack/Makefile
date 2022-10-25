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
MPIFC   = mpif90
RUN     = mpirun
OBJ     = o
EXE	= out
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
MPICC      = nvc -Mmpi=msmpi
MPIFC      = nvfortran -Mmpi=msmpi
RUN     = mpiexec
OBJ     = obj
EXE	= out
endif

FCFLAGS = -fast

all: build run verify

build: pdgetrrv.f pdlafchk.f pdlaschk.f pdludriver.f pdluinfo.f pdmatgen.f pmatgeninc.f
	$(MPIFC) $(FCFLAGS) -c pdgetrrv.f
	$(MPIFC) $(FCFLAGS) -c pdlafchk.f
	$(MPIFC) $(FCFLAGS) -c pdlaschk.f
	$(MPIFC) $(FCFLAGS) -c pdluinfo.f
	$(MPIFC) $(FCFLAGS) -c pdmatgen.f
	$(MPIFC) $(FCFLAGS) -c pmatgeninc.f
	$(MPIFC) $(FCFLAGS) -o xdlu.$(EXE) pdludriver.f pdgetrrv.$(OBJ) pdlafchk.$(OBJ) pdlaschk.$(OBJ) pdluinfo.$(OBJ) pdmatgen.$(OBJ) pmatgeninc.$(OBJ) -Mscalapack

run: xdlu.$(EXE)
	$(RUN) -np $(NPROCS) ./xdlu.$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
