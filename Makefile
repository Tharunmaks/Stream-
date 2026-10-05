ifeq ($(origin CC),default)
CC = clang
endif
CFLAGS  ?= -O2 -g -Wall -Wextra -std=c11
LDFLAGS ?=
LDLIBS  ?= -lpthread -lm

# The SDOT kernel needs armv8.2-a+dotprod. It is only called when the CPU
# reports asimddp at runtime, so the binary still runs on older cores.
ARCH ?= $(shell uname -m)
ifeq ($(ARCH),aarch64)
DOTPROD_FLAGS ?= -march=armv8.2-a+dotprod
endif

BIN  = es_gen es_bench es_kbench es_run
HDRS = $(wildcard src/*.h) tools/es_common.h
CORE = src/es_io.c src/es_cpu.c src/es_q2k.c src/es_compute.c src/es_cache.c

all: $(BIN)

es_q2k_neon.o: src/es_q2k_neon.c $(HDRS)
	$(CC) $(CFLAGS) $(DOTPROD_FLAGS) -c -o $@ $<

es_gen: tools/es_gen.c $(CORE) $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_gen.c src/es_io.c src/es_cpu.c $(LDFLAGS) $(LDLIBS)

es_bench: tools/es_bench.c $(CORE) $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_bench.c src/es_io.c src/es_cpu.c $(LDFLAGS) $(LDLIBS)

es_kbench: tools/es_kbench.c $(CORE) es_q2k_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_kbench.c $(CORE) es_q2k_neon.o $(LDFLAGS) $(LDLIBS)

es_run: tools/es_run.c $(CORE) es_q2k_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_run.c $(CORE) es_q2k_neon.o $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(BIN) *.o

.PHONY: all clean
