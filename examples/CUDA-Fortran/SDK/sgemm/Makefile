#
#     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

F90FLAGS = -fast
OBJ     = o
EXE     = out
RUN     =
CUBLASLIB    = -cudalib=cublas
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
OBJ     = obj
EXE     = exe
endif

# Uncomment the CUDAFLAGS line if you have hardware requirements that require
# a specific compute capability
#CUDAFLAGS = -cuda

all: build run verify

build: sgemm.cuf
	nvfortran $(CUDAFLAGS) $(F90FLAGS) $(OPT) -o sgemm.$(EXE) $< $(CUBLASLIB)

run: sgemm.$(EXE)
	$(RUN) ./sgemm.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf sgemm.$(EXE) *.$(OBJ) *.mod *.dwf *.pdb prof
