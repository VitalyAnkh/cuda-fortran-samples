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

TEST = cholesky_gpu
CXX  = nvc++
EXE  = exe

ifeq ($(TEST), cholesky_gpu)
  LIB = -stdpar -cudalib=cublas
else
  LIB = -stdpar=multicore -lblas
endif

all: build run verify

build: cholesky.cpp
	$(CXX) -o $(TEST).$(EXE) --c++17 $(LIB) $<

run: $(TEST).$(EXE)
	$(RUN) ./$(TEST).$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE)
