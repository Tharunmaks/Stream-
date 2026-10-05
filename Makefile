ifeq ($(origin CC),default)
CC = clang
endif
CFLAGS  ?= -O2 -g -Wall -Wextra -std=c11
LDFLAGS ?=
LDLIBS  ?= -lpthread

BIN = es_gen es_bench

all: $(BIN)

es_gen: tools/es_gen.c src/es_io.c src/es_io.h src/es_format.h tools/es_common.h
	$(CC) $(CFLAGS) -o $@ tools/es_gen.c src/es_io.c $(LDFLAGS) $(LDLIBS)

es_bench: tools/es_bench.c src/es_io.c src/es_io.h src/es_format.h tools/es_common.h
	$(CC) $(CFLAGS) -o $@ tools/es_bench.c src/es_io.c $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(BIN)

.PHONY: all clean
