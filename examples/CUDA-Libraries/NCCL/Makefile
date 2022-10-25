
# 
#     Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
	@echo "make ftn_all2all : All2All implementation"
	@echo "    "
	@echo "make all         : all of the above tests"

ftn_all2all:
	cd test_nccl_all2all_ftn; make TEST=tncclall2allftn; make clean

all: ftn_all2all
