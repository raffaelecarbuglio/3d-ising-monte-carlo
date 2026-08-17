CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS = -lm
PROGRAM_OBJECTS = main.o input.o ising.o rng.o
TEST_OBJECTS = tests.o ising.o rng.o
.PHONY: all test test-analysis clean
all: ising
ising: $(PROGRAM_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(PROGRAM_OBJECTS) $(LDLIBS)
tests: $(TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJECTS) $(LDLIBS)
test: ising tests
	./tests

test-analysis:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_analysis.py
main.o: main.c input.h ising.h rng.h
input.o: input.c input.h
ising.o: ising.c ising.h rng.h
rng.o: rng.c rng.h
tests.o: tests.c ising.h rng.h
clean:
	rm -f ising tests *.o *.tmp example_data.dat example_config.dat restart_config.dat
