CC = gcc
CPPFLAGS += -I.
CFLAGS = -std=c11 -O3 -flto -Wall -Wextra -Wpedantic
LDLIBS = -lm
PROGRAM_OBJECTS = main.o input.o ising.o rng.o
TEST_OBJECTS = tests/test_ising.o ising.o rng.o
.PHONY: all test test-analysis clean
all: ising
ising: $(PROGRAM_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(PROGRAM_OBJECTS) $(LDLIBS)
tests/test_ising: $(TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJECTS) $(LDLIBS)
test: ising tests/test_ising
	./tests/test_ising

test-analysis:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v
main.o: main.c input.h ising.h rng.h
input.o: input.c input.h
ising.o: ising.c ising.h rng.h
rng.o: rng.c rng.h
tests/test_ising.o: tests/test_ising.c ising.h rng.h
clean:
	rm -f ising tests/test_ising *.o tests/*.o *.tmp
	rm -rf __pycache__ tests/__pycache__
