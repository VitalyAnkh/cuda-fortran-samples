
#
#     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

NTHREADS=2
OBJ=o
EXE=out
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
   OBJ=obj
   EXE=exe
endif
all: build run verify

build: fftpde_4.F
	@echo ------------------------------------ building fftpde.$(EXE)
	nvfortran -fast -mp -Minfo -o fftpde.$(EXE) fftpde_4.F

run: fftpde.$(EXE)
	@echo '------------- executing fftpde.$(EXE) with $(NTHREADS) threads'
	export OMP_NUM_THREADS=$(NTHREADS); export NO_STOP_MESSAGE=1; ./fftpde.$(EXE)

verify: ../check.F
	@nvfortran -o check.$(EXE) ../check.F
	./check.$(EXE)

clean:
	@echo 'Cleaning up...'
	@rm -rf fftpde.$(EXE) check.$(EXE) fft.out *.dwf *.pdb prof
