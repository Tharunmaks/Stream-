ifeq ($(origin CC),default)
CC = clang
endif
CFLAGS  ?= -O3 -g -Wall -Wextra -std=c11
LDFLAGS ?=
LDLIBS  ?= -lpthread -lm -ldl

# The SDOT kernel needs armv8.2-a+dotprod. It is only called when the CPU
# reports asimddp at runtime, so the binary still runs on older cores.
ARCH ?= $(shell uname -m)
ifeq ($(ARCH),aarch64)
DOTPROD_FLAGS ?= -march=armv8.2-a+dotprod
endif

BIN  = es_gen es_bench es_kbench es_run es_import es_chat es_serve es_mtest es_qtest es_toktest es_tmpltest es_get es_graft es_lora es_train
HDRS = $(wildcard src/*.h) tools/es_common.h
CORE = src/es_io.c src/es_cpu.c src/es_q2k.c src/es_compute.c src/es_cache.c

all: $(BIN) libesd.so

es_q2k_neon.o: src/es_q2k_neon.c $(HDRS)
	$(CC) $(CFLAGS) $(DOTPROD_FLAGS) -c -o $@ $<
es_quant_neon.o: src/es_quant_neon.c $(HDRS)
	$(CC) $(CFLAGS) $(DOTPROD_FLAGS) -c -o $@ $<

es_gen: tools/es_gen.c $(CORE) $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_gen.c src/es_io.c src/es_cpu.c $(LDFLAGS) $(LDLIBS)

es_bench: tools/es_bench.c $(CORE) $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_bench.c src/es_io.c src/es_cpu.c $(LDFLAGS) $(LDLIBS)

es_kbench: tools/es_kbench.c $(CORE) es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_kbench.c $(CORE) es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)

es_run: tools/es_run.c $(CORE) es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_run.c $(CORE) es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)

es_import: tools/es_import.c src/es_gguf.c src/es_io.c src/es_cpu.c $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_import.c src/es_gguf.c src/es_io.c src/es_cpu.c $(LDFLAGS) $(LDLIBS)

ENGINE = src/es_engine.c src/es_graft.c src/es_gguf.c src/es_quant.c src/es_tok.c src/es_regex.c src/es_fetch.c

es_chat: tools/es_chat.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_chat.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)

es_serve: tools/es_serve.c tools/es_json.h $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_serve.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)

es_mtest: tools/es_mtest.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_mtest.c $(CORE) src/es_quant.c es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)

es_fmtbench: tools/es_fmtbench.c src/es_gguf.c src/es_quant.c src/es_q2k.c src/es_cpu.c es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_fmtbench.c src/es_gguf.c src/es_quant.c src/es_q2k.c src/es_cpu.c es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)

es_qtest: tools/es_qtest.c src/es_gguf.c src/es_quant.c src/es_q2k.c src/es_cpu.c es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_qtest.c src/es_gguf.c src/es_quant.c src/es_q2k.c src/es_cpu.c es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)

es_toktest: tools/es_toktest.c src/es_gguf.c src/es_tok.c src/es_regex.c $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_toktest.c src/es_gguf.c src/es_tok.c src/es_regex.c $(LDFLAGS) $(LDLIBS)

es_lora: tools/es_lora.c tools/es_data.h tools/es_json.h src/es_train.c src/es_amx.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_lora.c src/es_train.c src/es_amx.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)
# es_train: one trainer, modes finetune (es_lora) and moe (moe/es_moe.c); es_lora stays as its own binary too
es_train: tools/es_train.c tools/es_lora.c moe/es_moe.c moe/es_sparse.c moe/es_sparse.h tools/es_data.h tools/es_json.h src/es_train.c src/es_amx.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -c -o es_lora_u.o -DES_TRAIN_UNIFIED tools/es_lora.c
	$(CC) $(CFLAGS) -c -o es_moe_u.o -DES_TRAIN_UNIFIED moe/es_moe.c
	$(CC) $(CFLAGS) -c -o es_sparse.o moe/es_sparse.c
	$(CC) $(CFLAGS) -o $@ tools/es_train.c es_lora_u.o es_moe_u.o es_sparse.o src/es_train.c src/es_amx.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)
es_graft: tools/es_graft.c tools/es_json.h $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_graft.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)

es_tmpltest: tools/es_tmpltest.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_tmpltest.c $(CORE) $(ENGINE) es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)

# libesd.so: es-sparse-v1 decoder for the Run page (es_common.esmoe.SparseRun); install next to es_train in ES_BIN_DIR
libesd.so: moe/esd.c moe/esd_kern.h moe/es_sparse.h
	$(CC) $(CFLAGS) -shared -fPIC -o $@ moe/esd.c -lm -lpthread
# esd_ktest: every SIMD kernel must equal the scalar one exactly
esd_ktest: moe/esd_ktest.c moe/esd_kern.h
	$(CC) $(CFLAGS) -o $@ moe/esd_ktest.c
# esd_bench: decode tok/s for a shape with every expert in RAM (./esd_bench --threads 4 [--bits 2])
esd_bench: moe/esd_bench.c moe/esd.c moe/esd_kern.h moe/es_sparse.h
	$(CC) $(CFLAGS) -DESD_BENCH_SPREAD -o $@ moe/esd_bench.c moe/esd.c -lm -lpthread

clean:
	rm -f $(BIN) libesd.so esd_bench esd_ktest *.o

.PHONY: all clean

es_get: tools/es_get.c src/es_fetch.c src/es_gguf.c src/es_quant.c src/es_q2k.c src/es_cpu.c es_q2k_neon.o es_quant_neon.o $(HDRS)
	$(CC) $(CFLAGS) -o $@ tools/es_get.c src/es_fetch.c src/es_gguf.c src/es_quant.c src/es_q2k.c src/es_cpu.c es_q2k_neon.o es_quant_neon.o $(LDFLAGS) $(LDLIBS)
