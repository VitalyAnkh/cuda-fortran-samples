
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
	@echo "make ftn_test3       : OpenACC FTN with cublasXt host calls"
	@echo "make ftn_test4       : OpenACC FTN calls within data regions"
	@echo "make c_testxt        : OpenACC C with cublasXt host calls"
	@echo "make c_test1         : OpenACC C calls within data regions"
	@echo "make cpp_testxt      : OpenACC C++ with host calls"
	@echo "make cpp_test2       : OpenACC C++ calls within data regions"
	@echo "make ftn_test3_omp   : OpenMP FTN calls within data regions"
	@echo "make ftn_testxt_omp  : OpenMP FTN calls with cublasXt host calls"
	@echo "make c_testxt_omp    : OpenMP C with cublasXt host calls"
	@echo "make c_test1_omp     : OpenMP C calls within data regions"
	@echo "make cpp_testxt_omp  : OpenMP C++ with host calls"
	@echo "make cpp_test2_omp   : OpenMP C++ calls within data regions"
	@echo "    "
	@echo "make all             : all of the above tests"

ftn_test3:
	cd test_blas_oacc_ftn; make TEST=tblas3; make clean

ftn_test4:
	cd test_blas_oacc_ftn; make TEST=tblas4; make clean

c_testxt:
	cd test_blas_oacc_c; make TEST=tblasxt; make clean

c_test1:
	cd test_blas_oacc_c; make TEST=tblas1; make clean

cpp_testxt:
	cd test_blas_oacc_cpp; make TEST=tblasxt; make clean

cpp_test2:
	cd test_blas_oacc_cpp; make TEST=tblas2; make clean

ftn_test3_omp:
	cd test_blas_omp_ftn; make TEST=tblasomp3; make clean

ftn_testxt_omp:
	cd test_blas_omp_ftn; make TEST=tblasxtomp; make clean

c_testxt_omp:
	cd test_blas_omp_c; make TEST=tblasxtomp; make clean

c_test1_omp:
	cd test_blas_omp_c; make TEST=tblasomp1; make clean

cpp_testxt_omp:
	cd test_blas_omp_cpp; make TEST=tblasxtomp; make clean

cpp_test2_omp:
	cd test_blas_omp_cpp; make TEST=tblasomp2; make clean

all: ftn_test3 ftn_test4 c_testxt c_test1 cpp_testxt cpp_test2 \
     ftn_test3_omp ftn_testxt_omp c_testxt_omp c_test1_omp cpp_testxt_omp cpp_test2_omp
