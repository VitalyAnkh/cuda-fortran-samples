
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

UNAME := $(shell uname -a)
help:
	@echo "To build and run a specific example, do the following:"
	@echo "    "
	@echo C Targets include:
	@echo "make omploop_c1_test            : first simple example"
	@echo "make omploop_c2_test            : second simple example"
	@echo "make omploop_c2a_test           : second simple example, executed 3 times"
	@echo "make omploop_c3_test            : Jacobi relaxation"
	@echo "                              omploop_c3 n m iters "
	@echo "make omploop_c3m_test           : Jacobi relaxation with metadirectives"
	@echo "make nbody_test             : nbody test(double precision)"
	@echo "make nbody_sp_test          : nbody test(single precision)"
	@echo "make nbody2_test            : nbody test with optimized data region(double precision)"
	@echo "make nbody2_sp_test         : nbody test with optimized data region(single precision)"
	@echo ""
	@echo Fortran Targets include:
	@echo "make omploop_f1_test            : first simple example"
	@echo "make omploop_f2_test            : second simple example"
	@echo "make omploop_f2a_test           : second simple example, executed 3 times"
	@echo "make omploop_f3_test            : Jacobi relaxation"
	@echo "                              omploop_f3 n m iters "
	@echo "                              [defaults 1000 1000 10]"
	@echo "make omploop_f3a_test           : Jacobi relaxation with data region"
	@echo "make omploop_f3m_test           : Jacobi relaxation with metadirective"
	@echo ""
	@echo "make nbody_test                 : NBody double precsion Test"
	@echo "make nbody_sp_test              : NBody single precsion Test"
	@echo "make nbody2_test                : NBody double precsion Test w/ data region"
	@echo "make nbody2_sp_test             : NBody single precsion Test w/ data region"
	@echo ""
	@echo "make all                    : all of the above tests"

omploop_c1_test:
	cd omploop_c1; make build; make run; make clean

omploop_c2_test:
	cd omploop_c2; make build; make run; make clean

omploop_c2a_test:
	cd omploop_c2a; make build; make run; make clean

omploop_c3_test:
	cd omploop_c3; make build; make run; make clean

omploop_c3m_test:
	cd omploop_c3meta; make build; make run; make clean

nbody_test:
	cd nbody; make PREC=FP64 build; make run; make clean

nbody_sp_test:
	cd nbody; make PREC=FP32 build; make run; make clean

nbody2_test:
	cd nbody2; make PREC=FP64 build; make run; make clean

nbody2_sp_test:
	cd nbody2; make PREC=FP32 build; make run; make clean


omploop_f1_test:
	cd omploop_f1; make build; make run; make clean

omploop_f2_test:
	cd omploop_f2; make build; make run; make clean

omploop_f2a_test:
	cd omploop_f2a; make build; make run; make clean

omploop_f2i_test:
	cd omploop_f2i; make build; make run; make clean

omploop_f3_test:
	cd omploop_f3; make build; make run; make clean

omploop_f3a_test:
	cd omploop_f3a; make build; make run; make clean

omploop_f3m_test:
	cd omploop_f3meta; make build; make run; make clean

all: omploop_c1_test omploop_c2_test omploop_c2a_test omploop_c3_test    \
     omploop_c3m_test nbody_test nbody_sp_test nbody2_test   \
     nbody2_sp_test omploop_f1_test omploop_f2_test omploop_f2a_test  \
     omploop_f3_test omploop_f3a_test omploop_f3m_test
