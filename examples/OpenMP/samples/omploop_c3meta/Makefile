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

CC       = nvc
OBJ     = o
EXE	= out
RUN     =
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
OBJ     = obj
EXE	= exe
endif

CCFLAGS  = -fast
OMPFLAGS = -Minfo=mp -mp=gpu,multicore $(OPT)

all: build run verify

build: omploop_c3m.c
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o omploop_c3m.$(EXE) $<

run: omploop_c3m.$(EXE)
	./omploop_c3m.$(EXE) 1024 1024 1000 GPU
	./omploop_c3m.$(EXE) 1024 1024 1000 HOST

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
