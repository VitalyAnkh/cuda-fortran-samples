TEST     = tcufft2dompc1
CC       = nvc
EXE      = exe

ifeq ($(TEST),tcufft2dompc1)
CCFLAGS = -fast -mp=gpu -cudalib=cufft
else
CCFLAGS = -fast -mp=gpu -gpu=managed -cudalib=cufft
endif

all: build run verify

build: $(TEST).c
	$(CC) $(CCFLAGS) -o $(TEST).$(EXE) $(TEST).c

run: $(TEST).$(EXE)
	$(RUN) ./$(TEST).$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof
