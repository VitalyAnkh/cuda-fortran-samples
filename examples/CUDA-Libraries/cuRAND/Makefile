
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
	@echo "make cuf_test1       : CUF with host calls"
	@echo "make cuf_test2       : CUF with device calls"
	@echo "make ftn_test3       : OpenACC FTN with host calls"
	@echo "make ftn_test4       : OpenACC FTN calls within data regions"
	@echo "make ftn_test5       : OpenACC FTN calls in compute regions"
	@echo "make c_test6         : OpenACC C with host calls"
	@echo "make c_test7         : OpenACC C calls within data regions"
	@echo "make c_test8         : OpenACC C calls in compute regions"
	@echo "make cpp_test6       : OpenACC C++ with host calls"
	@echo "make cpp_test7       : OpenACC C++ calls within data regions"
	@echo "make cpp_test8       : OpenACC C++ calls in compute regions"
	@echo "make ftn_test3_omp   : OpenMP FTN with host calls"
	@echo "make ftn_test4_omp   : OpenMP FTN calls within data regions"
	@echo "make c_test6_omp     : OpenMP C with host calls"
	@echo "make c_test7_omp     : OpenMP C calls within data regions"
	@echo "make cpp_test6_omp   : OpenMP C++ with host calls"
	@echo "make cpp_test7_omp   : OpenMP C++ calls within data regions"
	@echo "    "
	@echo "make nvcc_test1      : NVIDIA nvcc using pgc++, host calls"
	@echo "make nvcc_test2      : NVIDIA nvcc using pgc++, device calls"
	@echo "    "
	@echo "make all             : all of the above tests"

cuf_test1:
	cd test_rand_cuf; make TEST=trand1; make clean

cuf_test2:
	cd test_rand_cuf; make TEST=trand2; make clean

ftn_test3:
	cd test_rand_oacc_ftn; make TEST=trand3; make clean

ftn_test4:
	cd test_rand_oacc_ftn; make TEST=trand4; make clean

ftn_test5:
	cd test_rand_oacc_ftn; make TEST=trand5; make clean

c_test6:
	cd test_rand_oacc_c; make TEST=trand6; make clean

c_test7:
	cd test_rand_oacc_c; make TEST=trand7; make clean

c_test8:
	cd test_rand_oacc_c; make TEST=trand8; make clean

cpp_test6:
	cd test_rand_oacc_cpp; make TEST=trand6; make clean

cpp_test7:
	cd test_rand_oacc_cpp; make TEST=trand7; make clean

cpp_test8:
	cd test_rand_oacc_cpp; make TEST=trand8; make clean

ftn_test3_omp:
	cd test_rand_omp_ftn; make TEST=trandomp3; make clean

ftn_test4_omp:
	cd test_rand_omp_ftn; make TEST=trandomp4; make clean

c_test6_omp:
	cd test_rand_omp_c; make TEST=trandomp6; make clean

c_test7_omp:
	cd test_rand_omp_c; make TEST=trandomp7; make clean

cpp_test6_omp:
	cd test_rand_omp_cpp; make TEST=trandomp6; make clean

cpp_test7_omp:
	cd test_rand_omp_cpp; make TEST=trandomp7; make clean

# Not certain users will have nvcc installed.  If so,
#
nvcc_test1:
	cd test_rand_nvcc_cpp; make TEST=trandnv1; make clean

nvcc_test2:
	cd test_rand_nvcc_cpp; make TEST=trandnv2; make clean

all: cuf_test1 cuf_test2 ftn_test3 ftn_test4 ftn_test5    \
     c_test6 c_test7 c_test8 cpp_test6 cpp_test7 cpp_test8 \
     ftn_test3_omp ftn_test4_omp c_test6_omp c_test7_omp \
     cpp_test6_omp cpp_test7_omp
