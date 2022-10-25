
#
#     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
OBJ=o
EXE=out
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
   OBJ=obj
   EXE=exe
endif

all: build run verify

build: adj1.f90
	@echo ------------------------------------ building adj1.$(EXE)
	nvfortran -Mallocatable=03 -Minfo -o adj1.$(EXE) adj1.f90

run: adj1.$(EXE)
	@echo '------------- executing adj1.$(EXE) '
	./adj1.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.mod adj1.$(EXE) *.dwf *.pdb prof
