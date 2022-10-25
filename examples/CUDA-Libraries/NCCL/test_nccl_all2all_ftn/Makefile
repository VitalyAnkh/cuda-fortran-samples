#
#     Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

TEST        = tncclall2allftn
MPIFC       = mpif90
MPIRUN      = mpirun
EXE         = exe

NGPUMAX     = $(shell nvidia-smi -L | wc -l)
NGPU        = $(shell expr $$(( $(NGPUMAX) > 4 ? 4 : $(NGPUMAX) )))
CFLAGS      = -fast -cudalib=nccl
MPIRUNFLAGS = -np $(NGPU)

########## Make rule for test $(TEST)  ########
all: build run verify

build: $(TEST).cuf
	-$(MPIFC) $(CFLAGS) -o $(TEST).$(EXE) $<

run:
	$(MPIRUN) $(MPIRUNFLAGS) ./$(TEST).$(EXE)

verify:

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof
