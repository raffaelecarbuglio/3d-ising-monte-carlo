CC = gcc
CPPFLAGS += -I.
CFLAGS = -std=c11 -O3 -flto -Wall -Wextra -Wpedantic
LDLIBS = -lm
PROGRAM_OBJECTS = main.o input.o ising.o rng.o
TEST_OBJECTS = tests/test_ising.o ising.o rng.o
Z2_OBJECTS = z2_gauge/main_z2.o z2_gauge/z2.o rng.o
Z2_TEST_OBJECTS = tests/test_z2.o z2_gauge/z2.o rng.o
.PHONY: all test test-z2 test-analysis clean
all: ising z2
ising: $(PROGRAM_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(PROGRAM_OBJECTS) $(LDLIBS)
z2: $(Z2_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(Z2_OBJECTS) $(LDLIBS)
tests/test_ising: $(TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJECTS) $(LDLIBS)
tests/test_z2: $(Z2_TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $(Z2_TEST_OBJECTS) $(LDLIBS)
test: ising tests/test_ising tests/test_z2
	./tests/test_ising
	./tests/test_z2

test-z2: tests/test_z2
	./tests/test_z2

test-analysis:
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v
main.o: main.c input.h ising.h rng.h
input.o: input.c input.h
ising.o: ising.c ising.h rng.h
rng.o: rng.c rng.h
tests/test_ising.o: tests/test_ising.c ising.h rng.h
z2_gauge/main_z2.o: z2_gauge/main_z2.c z2_gauge/z2.h rng.h
z2_gauge/z2.o: z2_gauge/z2.c z2_gauge/z2.h rng.h
tests/test_z2.o: tests/test_z2.c z2_gauge/z2.h rng.h
clean:
	rm -f ising z2 tests/test_ising tests/test_z2 *.o tests/*.o z2_gauge/*.o *.tmp
	rm -rf __pycache__ tests/__pycache__
