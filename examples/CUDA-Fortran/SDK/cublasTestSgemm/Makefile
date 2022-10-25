
#
#     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

OBJ	= o
EXE	= out
RUN     =
F90FLAGS = -fast
CUDAFLAGS = -cuda
CUBLASLIB= -cudalib=cublas
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
OBJ	= obj
EXE     = exe
endif

# If you have hardware requirements that require a specific compute
# capability, update the CUDAFLAGS line

all: build run verify

build: cublasTestSgemm.F90
	nvfortran $(CUDAFLAGS) $(F90FLAGS) $(OPT) -o sgemm_gpu.$(EXE) $< $(CUBLASLIB)
	nvfortran $(F90FLAGS) -o sgemm_host.$(EXE) $< -lblas

run: sgemm_gpu.$(EXE) sgemm_host.$(EXE)
	@echo '----------------------------- GPU run ----------------'
	$(RUN) ./sgemm_gpu.$(EXE)
	@echo '----------------------------- CPU run ----------------'
	$(RUN) ./sgemm_host.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf sgemm_gpu.$(EXE) cublasTestSgemm.$(OBJ) sgemm_host.$(EXE) *.dwf *.pdb prof

