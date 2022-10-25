
#
#     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

#
# just typing 'make' prints out this help message
#

help:
	@echo "To build and run a specific example, do the following:"
	@echo "    "
	@echo "make [NTHREADS=n] linpack_test    : Linpack Test"
	@echo "make [NTHREADS=n] matmul_test     : Matrix Multiply Test"
	@echo "    "
	@echo "make [NTHREADS=n] all             : all of the above tests"
	@echo "    "
	@echo "where NTHREADS is set to the number of threads to test, default is 1."



NTHREADS = 1
OBJ=o
EXE=out
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
   OBJ=obj
   EXE=exe
endif


linpack_test:
	cd linpack; make OBJ=$(OBJ) EXE=$(EXE) build; make NTHREADS=$(NTHREADS) run; make OBJ=$(OBJ) EXE=$(EXE) clean
matmul_test:
	cd matmul; make OBJ=$(OBJ) EXE=$(EXE) build; make NTHREADS=$(NTHREADS) run; make OBJ=$(OBJ) EXE=$(EXE) clean
all: linpack_test matmul_test
