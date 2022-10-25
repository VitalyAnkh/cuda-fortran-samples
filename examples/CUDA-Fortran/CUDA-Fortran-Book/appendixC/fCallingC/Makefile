
#
#     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.
# 

OPT=-O2
FC=nvfortran
F90FLAGS = -m64 $(OPT)
OBJ=o
EXE=out
RUN=
UNAME := $(shell uname -a)
NVXX     = nvcc
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
   OBJ=obj
   EXE=exe
endif

# nvcc requires specifying compute capability to generate code for. The following logic
# attempts to figure this out automatically by looking through GPUs on current systems.
# However, it if fails to do so it may need done manually like the example.
# Example for Ampere: --generate-code arch=compute_80,code=sm_80
SYSCOMPCAP := $(shell nvaccelinfo | grep "Default Target:" | awk -F "cc" '{ print $$2 }' | sort | uniq)
NVCCOPTIONS := $(foreach CCAP,$(SYSCOMPCAP), --generate-code arch=compute_$(CCAP),code=sm_$(CCAP))

# Uncomment the CUDAFLAGS line if you have hardware requirements that require
# a specific compute capability
#CUDAFLAGS = -gpu=cc60

all: build run verify

build: fCallingC.cuf zero.cu
	$(NVXX) -O3 \
	$(NVCCOPTIONS) \
	-c zero.cu
	$(FC) $(CUDAFLAGS) $(F90FLAGS) -gpu=nordc -o fCallingC.$(EXE) zero.o fCallingC.cuf


run: fCallingC.$(EXE)
	$(RUN) ./fCallingC.$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf fCallingC.$(EXE) *.$(OBJ) *.mod *.dwf *.pdb prof
