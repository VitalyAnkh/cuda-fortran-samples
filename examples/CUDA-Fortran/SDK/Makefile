

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

UNAME := $(shell uname -a)
help:
	@echo "To build and run a specific example, do the following:"
	@echo "    "
	@echo "make bandwidth_test           : Bandwidth Test"
	@echo "make cublas_test              : cuBLAS Test"
	@echo "make cufft_test               : cuFFT Test"
	@echo "make cufinfo_test             : cuf Info Test"
	@echo "make cufkernel_test           : cuf Kernel Test"
	@echo "make devicequery_test         : Device Query Test"
	@echo "make finitedifference_test    : Finite Difference Test"
	@echo "make transpose_test           : Transpose Test"
	@echo "make sgemm_test               : Matrix Multiply Test(single precision)"
	@echo "     "
	@echo "make all                      : all of the above tests"

bandwidth_test:
	cd bandwidthTest; make build; make run; make clean

cublas_test:
	cd cublasTestSgemm; make build; make run; make clean

cufft_test:
	cd cufftTest; make build; make run; make clean

cufinfo_test:
	cd cufinfo; make build; make run; make clean
cufkernel_test:
	cd cufkernel; make build; make run; make clean
devicequery_test:
	cd deviceQuery; make build; make run; make clean
finitedifference_test:
	cd finiteDifference; make build; make run; make clean

transpose_test:
	cd transpose; make build; make run; make clean

sgemm_test:
	cd sgemm; make build; make run; make clean
all: bandwidth_test cublas_test cufft_test cufinfo_test cufkernel_test \
     devicequery_test finitedifference_test transpose_test sgemm_test
