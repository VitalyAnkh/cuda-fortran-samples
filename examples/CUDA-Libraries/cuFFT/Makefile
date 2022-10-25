
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
	@echo "make cuf_tfft1       : CUF with normal device data"
	@echo "make cuf_tfft2       : CUF with managed data"
	@echo "make ftn_tfftf1      : OpenACC FTN with normal calls"
	@echo "make ftn_tfftf2      : OpenACC FTN calls with managed data"
	@echo "make ftn_tfftf3      : OpenACC FTN calls alternate managed data"
	@echo "make c_tfftc1        : OpenACC C with normal calls"
	@echo "make c_tfftc2        : OpenACC C calls with managed data"
	@echo "make cpp_tfftc1      : OpenACC C++ with normal calls"
	@echo "make cpp_tfftc2      : OpenACC C++ calls with managed data"
	@echo "make ftn_tfftf1_omp  : OpenMP FTN with normal calls"
	@echo "make ftn_tfftf2_omp  : OpenMP FTN calls with managed data"
	@echo "make ftn_tfftf3_omp  : OpenMP FTN calls alternate managed data"
	@echo "make c_tfftc1_omp    : OpenMP C with normal calls"
	@echo "make c_tfftc2_omp    : OpenMP C calls with managed data"
	@echo "make cpp_tfftc1_omp  : OpenMP C++ with normal calls"
	@echo "make cpp_tfftc2_omp  : OpenMP C++ calls with managed data"
	@echo "make cpp_tfftc3_omp  : OpenMP C++ calls and target depend nowait, with managed data"
	@echo "    "
	@echo "make all             : all of the above tests"

cuf_tfft1:
	cd test_fft_cuf; make TEST=tfft1; make clean

cuf_tfft2:
	cd test_fft_cuf; make TEST=tfft2; make clean

ftn_tfftf1:
	cd test_fft_oacc_ftn; make TEST=tcufft2df1; make clean

ftn_tfftf2:
	cd test_fft_oacc_ftn; make TEST=tcufft2df2; make clean

ftn_tfftf3:
	cd test_fft_oacc_ftn; make TEST=tcufft2df3; make clean

c_tfftc1:
	cd test_fft_oacc_c; make TEST=tcufft2dc1; make clean

c_tfftc2:
	cd test_fft_oacc_c; make TEST=tcufft2dc2; make clean

cpp_tfftc1:
	cd test_fft_oacc_cpp; make TEST=tcufft2dc3; make clean

cpp_tfftc2:
	cd test_fft_oacc_cpp; make TEST=tcufft2dc4; make clean

ftn_tfftf1_omp:
	cd test_fft_omp_ftn; make TEST=tcufft2dompf1; make clean

ftn_tfftf2_omp:
	cd test_fft_omp_ftn; make TEST=tcufft2dompf2; make clean

ftn_tfftf3_omp:
	cd test_fft_omp_ftn; make TEST=tcufft2dompf3; make clean

c_tfftc1_omp:
	cd test_fft_omp_c; make TEST=tcufft2dompc1; make clean

c_tfftc2_omp:
	cd test_fft_omp_c; make TEST=tcufft2dompc2; make clean

cpp_tfftc1_omp:
	cd test_fft_omp_cpp; make TEST=tcufft2dompc3; make clean

cpp_tfftc2_omp:
	cd test_fft_omp_cpp; make TEST=tcufft2dompc4; make clean

cpp_tfftc3_omp:
	cd test_fft_omp_cpp; make TEST=tcufft2dompc5; make clean

all: cuf_tfft1 cuf_tfft2 ftn_tfftf1 ftn_tfftf2 ftn_tfftf3 \
     c_tfftc1 c_tfftc2 cpp_tfftc1 cpp_tfftc2 \
     ftn_tfftf1_omp ftn_tfftf2_omp ftn_tfftf3_omp \
     c_tfftc1_omp c_tfftc2_omp \
     cpp_tfftc1_omp cpp_tfftc2_omp cpp_tfftc3_omp
