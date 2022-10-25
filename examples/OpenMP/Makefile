
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
	@echo "make fftpde_test                 : FFTPDE Test(serial)"
	@echo "make [NTHREADS=n] fftpde_test1   : FFTPDE Test(parallel, step 1)"
	@echo "make [NTHREADS=n] fftpde_test2   : FFTPDE Test(parallel, step 2)"
	@echo "make [NTHREADS=n] fftpde_test3   : FFTPDE Test(parallel, step 3)"
	@echo "make [NTHREADS=n] fftpde_test4   : FFTPDE Test(parallel, step 4)"
	@echo "make [NTHREADS=n] fftpde_test5   : FFTPDE Test(parallel, step 5)"
	@echo "make [NTHREADS=n] matmul_test    : Matrix Multiply Test"
	@echo "    "
	@echo "make [NTHREADS=n] all            : all of the above tests"
	@echo "    "
	@echo "where NTHREADS is set to the number of threads to test, default is 1."

NTHREADS=1

fftpde_test:
	cd fftpde/step0; make build; make run; make verify; make clean

fftpde_test1:
	cd fftpde/step1; make build; make NTHREADS=$(NTHREADS) run; make verify; make clean

fftpde_test2:
	cd fftpde/step2; make build; make NTHREADS=$(NTHREADS) run; make verify; make clean

fftpde_test3:
	cd fftpde/step3; make build; make NTHREADS=$(NTHREADS) run; make verify; make clean

fftpde_test4:
	cd fftpde/step4; make build; make NTHREADS=$(NTHREADS) run; make verify; make clean

fftpde_test5:
	cd fftpde/step5; make build; make NTHREADS=$(NTHREADS) run; make verify; make clean

matmul_test:
	cd matmul; make build; make NTHREADS=$(NTHREADS) run; make clean

all: fftpde_test fftpde_test1 fftpde_test2 fftpde_test3 fftpde_test4 \
     fftpde_test5 matmul_test
