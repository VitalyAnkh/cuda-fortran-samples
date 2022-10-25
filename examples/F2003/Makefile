
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
	@echo "make adj1_test    : Test 1"
	@echo "make adj2_test    : Test 2"
	@echo "make adj3_test    : Test 3"
	@echo "    "
	@echo "make all          : all of the above tests"

adj1_test:
	cd adj1; make build; make run; make clean

adj2_test:
	cd adj2; make build; make run; make clean

adj3_test:
	cd adj3; make build; make run; make clean

all: adj1_test adj2_test adj3_test
