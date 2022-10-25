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

TEST     = mixed_precision
EXT      = cpp
CXX      = nvc++
CXXFLAGS = --c++17 -stdpar -cudalib=cublas
VERBOSE  = -DSTDBLAS_VERBOSE
EXE      = exe

all: build run verify

build: $(TEST).cpp
	$(CXX) $(CXXFLAGS) $(VERBOSE) -o $(TEST).$(EXE) $<

run:
	$(RUN) ./$(TEST).$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE)
