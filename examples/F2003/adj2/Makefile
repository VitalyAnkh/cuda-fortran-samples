
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

build: adj2.f90
	@echo ------------------------------------ building adj2.$(EXE)
	nvfortran -Minfo -o adj2.$(EXE) adj2.f90

run: adj2.$(EXE)
	@echo '------------- executing adj2.$(EXE) '
	./adj2.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.mod adj2.$(EXE) *.dwf *.pdb prof
