# 
#     Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
	@echo "make m16n16k16_test<num> : num = 1..8"
	@echo "make m32n8k16_test<num>  : num = 1..8"
	@echo "make m8n32k16_test<num>  : num = 1..8"
	@echo "make all                 : all of the above tests"

m16n16k16_test1:
	cd m16n16k16; make TESTNO=1; make clean

m16n16k16_test2:
	cd m16n16k16; make TESTNO=2; make clean

m16n16k16_test3:
	cd m16n16k16; make TESTNO=3; make clean

m16n16k16_test4:
	cd m16n16k16; make TESTNO=4; make clean

m16n16k16_test5:
	cd m16n16k16; make TESTNO=5; make clean

m16n16k16_test6:
	cd m16n16k16; make TESTNO=6; make clean

m16n16k16_test7:
	cd m16n16k16; make TESTNO=7; make clean

m16n16k16_test8:
	cd m16n16k16; make TESTNO=8; make clean
#
#
m32n8k16_test1:
	cd m32n8k16; make TESTNO=1; make clean

m32n8k16_test2:
	cd m32n8k16; make TESTNO=2; make clean

m32n8k16_test3:
	cd m32n8k16; make TESTNO=3; make clean

m32n8k16_test4:
	cd m32n8k16; make TESTNO=4; make clean

m32n8k16_test5:
	cd m32n8k16; make TESTNO=5; make clean

m32n8k16_test6:
	cd m32n8k16; make TESTNO=6; make clean

m32n8k16_test7:
	cd m32n8k16; make TESTNO=7; make clean

m32n8k16_test8:
	cd m32n8k16; make TESTNO=8; make clean
#
#
m8n32k16_test1:
	cd m8n32k16; make TESTNO=1; make clean

m8n32k16_test2:
	cd m8n32k16; make TESTNO=2; make clean

m8n32k16_test3:
	cd m8n32k16; make TESTNO=3; make clean

m8n32k16_test4:
	cd m8n32k16; make TESTNO=4; make clean

m8n32k16_test5:
	cd m8n32k16; make TESTNO=5; make clean

m8n32k16_test6:
	cd m8n32k16; make TESTNO=6; make clean

m8n32k16_test7:
	cd m8n32k16; make TESTNO=7; make clean

m8n32k16_test8:
	cd m8n32k16; make TESTNO=8; make clean

all: m16n16k16_test1 m16n16k16_test2 m16n16k16_test3 m16n16k16_test4 \
     m16n16k16_test5 m16n16k16_test6 m16n16k16_test7 m16n16k16_test8 \
      m32n8k16_test1  m32n8k16_test2  m32n8k16_test3  m32n8k16_test4 \
      m32n8k16_test5  m32n8k16_test6  m32n8k16_test7  m32n8k16_test8 \
      m8n32k16_test1  m8n32k16_test2  m8n32k16_test3  m8n32k16_test4 \
      m8n32k16_test5  m8n32k16_test6  m8n32k16_test7  m8n32k16_test8
