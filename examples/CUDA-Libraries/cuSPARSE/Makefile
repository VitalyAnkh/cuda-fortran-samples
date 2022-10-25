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
	@echo "make cuf_test1       : CUF with simple calls"
	@echo "make cuf_test2       : CUF with managed data"
	@echo "make ftn_test3       : OpenACC FTN with cusparse calls"
	@echo "make ftn_test4       : OpenACC FTN calls with managed data"
	@echo "make cpp_test1       : OpenACC C++ with cusparse calls"
	@echo "make cpp_test2       : OpenACC C++ calls with managed data"
	@echo "make ftn_test3_omp   : OpenMP FTN with cusparse calls"
	@echo "make ftn_test4_omp   : OpenMP FTN calls with managed data"
	@echo "make cpp_test1_omp   : OpenMP C++ with cusparse calls"
	@echo "make cpp_test2_omp   : OpenMP C++ calls with managed data"
	@echo "    "
	@echo "make all             : all of the above tests"

cuf_test1:
	cd test_sparse_cuf; make TEST=tcusparsemv1; make clean

cuf_test2:
	cd test_sparse_cuf; make TEST=tcusparsemv2; make clean

ftn_test3:
	cd test_sparse_oacc_ftn; make TEST=tcusparse3; make clean

ftn_test4:
	cd test_sparse_oacc_ftn; make TEST=tcusparse4; make clean

cpp_test1:
	cd test_sparse_oacc_cpp; make TEST=tcusparsec1; make clean

cpp_test2:
	cd test_sparse_oacc_cpp; make TEST=tcusparsec2; make clean

ftn_test3_omp:
	cd test_sparse_omp_ftn; make TEST=tcusparseomp3; make clean

ftn_test4_omp:
	cd test_sparse_omp_ftn; make TEST=tcusparseomp4; make clean

cpp_test1_omp:
	cd test_sparse_omp_cpp; make TEST=tcusparseompc1; make clean

cpp_test2_omp:
	cd test_sparse_omp_cpp; make TEST=tcusparseompc2; make clean

all: cuf_test1 cuf_test2 ftn_test3 ftn_test4 cpp_test1 cpp_test2 \
     ftn_test3_omp ftn_test4_omp cpp_test1_omp cpp_test2_omp
