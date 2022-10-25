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

TEST    = matrix_product_gpu
TESTSRC = matrix_product
EXT     = cpp
CXX     = nvc++
ifeq ($(TEST),matrix_product_gpu)
    CXXFLAGS = --c++17 -stdpar -cudalib=cublas
else
ifeq ($(TEST),matrix_product_multicore)
    CXXFLAGS = --c++17 -stdpar=multicore -lblas
else
    CXXFLAGS = --c++17
endif
endif
VERBOSE = -DSTDBLAS_VERBOSE
EXE     = exe

all: build run verify

build: $(TESTSRC).cpp
	$(CXX) $(CXXFLAGS) $(VERBOSE) -o $(TEST).$(EXE) $<

run:
	$(RUN) ./$(TEST).$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE)
