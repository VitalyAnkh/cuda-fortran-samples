
#
# Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
	@echo "make lu_managed   : LU calls with managed data"
	@echo "make eigen_managed: Eigen calls with managed data"
	@echo "make lu_oacc      : LU calls with OpenACC directives"
	@echo "make eigen_oacc   : Eigen calls with OpenACC directives"
	@echo "make lu_omp       : LU calls with OpenMP directives"
	@echo "make eigen_omp    : Eigen calls with OpenMP directives"
	@echo "make lu_cuf       : LU calls with CUDA Fortran device arrays"
	@echo "make eigen_cuf    : Eigen calls with CUDA Fortran device arrays"
	@echo "make lu_fallback  : LU calls with fallback enabled"
	@echo "    "
	@echo "make all          : all of the above tests"

lu_managed:
	cd test_managed; make TEST=lu_managed; make clean

eigen_managed:
	cd test_managed; make TEST=eigen_managed; make clean

lu_oacc:
	cd test_oacc; make TEST=lu_oacc; make clean

eigen_oacc:
	cd test_oacc; make TEST=eigen_oacc; make clean

lu_omp:
	cd test_omp; make TEST=lu_omp; make clean

eigen_omp:
	cd test_omp; make TEST=eigen_omp; make clean

lu_cuf:
	cd test_cuf; make TEST=lu_cuf; make clean

eigen_cuf:
	cd test_cuf; make TEST=eigen_cuf; make clean

lu_fallback:
	cd test_fallback; make TEST=lu_fallback; make clean

all: lu_managed eigen_managed lu_oacc eigen_oacc lu_omp eigen_omp lu_cuf eigen_cuf lu_fallback
