
#
#     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

F90FLAGS = -fast -Mpreprocess
OBJ=o
EXE=out
RUN=
UNAME := $(shell uname -a)
CUFFTLIB=-cudalib=cufft
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
   OBJ=obj
   EXE=exe
   CUFFTLIB=-defaultlib:cufft
endif

# Uncomment the CUDAFLAGS line if you have hardware requirements that require
# a specific compute capability
#CUDAFLAGS = -gpu=cc60

all: build run verify

build: cufftTest.cuf cufft_m.cuf precision_m.cuf
	nvfortran $(CUDAFLAGS) $(F90FLAGS) -c precision_m.cuf
	nvfortran $(CUDAFLAGS) $(F90FLAGS) -c cufft_m.cuf
	nvfortran $(CUDAFLAGS) $(F90FLAGS) -o cufftTest.$(EXE) cufftTest.cuf cufft_m.$(OBJ) precision_m.$(OBJ) $(CUFFTLIB)

run: cufftTest.$(EXE)
	$(RUN) ./cufftTest.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(OBJ) *.mod cufftTest.$(EXE) *.dwf *.pdb prof

