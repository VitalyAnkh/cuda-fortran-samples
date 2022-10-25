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

TEST    = no_sync
TESTSRC = $(TEST)
EXT     = cpp
CXX     = nvc++
CXXFLAGS = --c++17 -stdpar -cudalib=cublas
EXE     = exe

all: build run verify

build: $(TESTSRC).cpp
	$(CXX) $(CXXFLAGS) -o par.$(EXE) $<
	$(CXX) $(CXXFLAGS) -DNO_SYNC -o par_no_sync.$(EXE) $<

run:
	$(RUN) ./par.$(EXE)
	$(RUN) ./par_no_sync.$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE)
