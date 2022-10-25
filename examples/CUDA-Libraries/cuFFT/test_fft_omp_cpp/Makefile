TEST     = tcufft2dompc3
CXX      = nvc++
EXE      = exe

ifneq ($(TEST),tcufft2dompc4)
CXXFLAGS = -fast -mp=gpu -cudalib=cufft
else
CXXFLAGS = -fast -mp=gpu -gpu=managed -cudalib=cufft
endif

all: build run verify

build: $(TEST).cpp
	$(CXX) $(CXXFLAGS) -o $(TEST).$(EXE) $(TEST).cpp

run: $(TEST).$(EXE)
	$(RUN) ./$(TEST).$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.dwf *.pdb *.mod prof
