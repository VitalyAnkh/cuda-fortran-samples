################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

#
# just typing 'make' prints out this help message
#

help:
	@echo "To build and run a specific example, do the following:"
	@echo "    "
	@echo "make matrix_product_gpu                : parallel matrix_product with the cuBLAS backend"
	@echo "make matrix_product_multicore          : parallel matrix_product with the BLAS backend"
	@echo "make matrix_product_seq                : sequential matrix_product"
	@echo "make mixed_precision                   : mixed-precision matrix_product"
	@echo "make unsupported_compile_time_error    : compile-time error for unsupported case"
	@echo "make unsupported_compile_time_fallback : sequential-fallback for unsupported case instead of compile-"
	@echo "                                         time error, enabled by compiling with a macro"
	@echo "make unsupported_run_time_error        : run-time error for unsupported case"
	@echo "make unsupported_run_time_fallback     : sequential-fallback for unsupported case instead of run-time"
	@echo "                                         error, enabled by creating an environment variable"
	@echo "make cholesky_gpu                      : Cholesky factorization and solution with stdBLAS cuBLAS backend"
	@echo "make cholesky_multicore                : Cholesky factorization and solution with stdBLAS BLAS backend"
	@echo "make no_sync                           : sync vs. no_sync parallel matrix_product with the cuBLAS backend"
	@echo "    "
	@echo "make all                               : all of the above examples"

matrix_product_gpu:
	cd matrix_product; make TEST=matrix_product_gpu; make clean

matrix_product_multicore:
	cd matrix_product; make TEST=matrix_product_multicore; make clean

matrix_product_seq:
	cd matrix_product; make TEST=matrix_product_seq; make clean

mixed_precision:
	cd mixed_precision; make TEST=mixed_precision; make clean

unsupported_compile_time_error:
	cd unsupported_cases; make TEST=compile_time_error; make clean

unsupported_compile_time_fallback:
	cd unsupported_cases; make TEST=compile_time_fallback; make clean

unsupported_run_time_error:
	cd unsupported_cases; make TEST=run_time_error; make clean

unsupported_run_time_fallback:
	cd unsupported_cases; make TEST=run_time_fallback; make clean

cholesky_gpu:
	cd cholesky; make TEST=cholesky_gpu; make clean

cholesky_multicore:
	cd cholesky; make TEST=cholesky_multicore; make clean

no_sync:
	cd no_sync; make TEST=no_sync; make clean

all: matrix_product_gpu matrix_product_multicore matrix_product_seq \
     mixed_precision unsupported_compile_time_error unsupported_compile_time_fallback \
     unsupported_run_time_error unsupported_run_time_fallback \
     cholesky_gpu cholesky_multicore no_sync
