
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
	@echo "make cuf_test1   : CUF with simple sort"
	@echo "make cuf_test2   : CUF with select sort"
	@echo "make ftn_test3   : OpenACC FTN with sort calls"
	@echo "make ftn_test4   : OpenACC FTN calls sort with managed data"
	@echo "make ftn_test5   : OpenACC FTN calls, more cudafor-like"
	@echo "make cpp_test6   : OpenACC C++ calls to sort with streams"
	@echo "make cpp_test7   : OpenACC C++ calls to sort and managed data"
	@echo "    "
	@echo "make all         : all of the above tests"

cuf_test1:
	cd test_thrust_cuf; make TEST=tsortcuf1; make clean

cuf_test2:
	cd test_thrust_cuf; make TEST=tsortcuf2; make clean

ftn_test3:
	cd test_thrust_oacc_ftn; make TEST=tsort3; make clean

ftn_test4:
	cd test_thrust_oacc_ftn; make TEST=tsort4; make clean

ftn_test5:
	cd test_thrust_oacc_ftn; make TEST=tsort5; make clean

cpp_test6:
	cd test_thrust_oacc_cpp; make TEST=tsort6; make clean

cpp_test7:
	cd test_thrust_oacc_cpp; make TEST=tsort7; make clean

all: cuf_test1 cuf_test2 ftn_test3 ftn_test4 ftn_test5    \
     cpp_test6 cpp_test7
