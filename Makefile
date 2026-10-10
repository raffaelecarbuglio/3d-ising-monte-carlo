CC = gcc
CFLAGS = -std=c11 -O3 -flto -Wall -Wextra -Wpedantic
LDLIBS = -lm
PROGRAM_OBJECTS = main.o checkpoint.o input.o ising.o rng.o
TEST_OBJECTS = tests.o ising.o rng.o
.PHONY: all test test-cluster test-analysis clean
all: ising
ising: $(PROGRAM_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(PROGRAM_OBJECTS) $(LDLIBS)
tests: $(TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJECTS) $(LDLIBS)
test: ising tests
	./tests

test-cluster: ising
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_cluster.py

test-analysis:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_analysis.py test_analyze_grid.py test_fit_scaling.py test_fit_beta_c.py test_compare_universality.py
main.o: main.c checkpoint.h input.h ising.h rng.h
checkpoint.o: checkpoint.c checkpoint.h input.h ising.h rng.h
input.o: input.c input.h
ising.o: ising.c ising.h rng.h
rng.o: rng.c rng.h
tests.o: tests.c ising.h rng.h
clean:
	rm -f ising tests *.o *.tmp
	rm -rf __pycache__
