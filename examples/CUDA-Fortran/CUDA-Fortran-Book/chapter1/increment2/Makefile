
#
#     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.
# 

OPT=-O2
FC=nvfortran
F90FLAGS = $(OPT)
OBJ=o
EXE=out
RUN=
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
   OBJ=obj
   EXE=exe
endif

# Uncomment the CUDAFLAGS line if you have hardware requirements that require
# a specific compute capability
#CUDAFLAGS = -gpu=cc60

all: build run verify

build: increment2.cuf
	$(FC) $(CUDAFLAGS) $(F90FLAGS) -o increment2.$(EXE) $<

run: increment2.$(EXE)
	$(RUN) ./increment2.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf increment2.$(EXE) *.$(OBJ) *.mod *.dwf *.pdb prof

