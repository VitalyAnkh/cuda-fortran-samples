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
OBJ	= o
EXE	= out
RUN     =
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
OBJ	= obj
EXE	= exe
endif
CCFLAGS  = -fast
OMPFLAGS = -Minfo=mp -mp=gpu $(OPT)

all: build run verify

build: omploop_c2.c
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o omploop_c2.$(EXE) $<

run: omploop_c2.$(EXE)
	$(RUN) ./omploop_c2.$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.dwf *.pdb prof
