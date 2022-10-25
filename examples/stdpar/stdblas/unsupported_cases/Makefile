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

TEST = compile_time_error
EXAMPLE = unsupported_$(TEST)

ifneq ($(filter compile_time_error compile_time_fallback,$(TEST)),)
    TESTSRC = unsupported_compile_time
    ifeq ($(TEST),compile_time_fallback)
        FALLBACKFLAG   = -DSTDBLAS_FALLBACK_UNSUPPORTED_CASES -DSTDBLAS_VERBOSE
    endif
else
    TESTSRC = unsupported_run_time
    ifeq ($(TEST),run_time_fallback)
        FALLBACKFLAG   = -DSTDBLAS_VERBOSE
        FALLBACKENVVAR = export NV_STDBLAS_RUNTIME_FALLBACK=1;
    endif
endif

CXX = nvc++
EXE = exe

CXXFLAGS = --c++17 -stdpar -cudalib=cublas
EXT      = cpp

all: build run verify

build: $(TESTSRC).cpp
ifeq ($(TEST),compile_time_error)
	@echo "=== This is a negative example. Compilation will fail. ==="
endif
ifeq ($(TEST),run_time_error)
	@echo "=== This is a negative example. Job will exit with a system error. ==="
endif
	 $(CXX) $(CXXFLAGS) $(FALLBACKFLAG) -o $(EXAMPLE).$(EXE) $<

run:
	$(FALLBACKENVVAR) ./$(EXAMPLE).$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE)
