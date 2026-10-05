/*
 * ExpertStream parallel expert loader.
 *
 * A fixed pool of reader threads pulls fixed-size chunks from one shared
 * queue. A request (one expert file) is split into chunks, so:
 *   - one expert alone is still read by all threads in parallel (low latency)
 *   - many experts submitted together keep every reader busy (high bandwidth)
 *
 * Usage:
 *   es_pool *p = es_pool_create(4, 1 << 20, 1);
 *   es_req r = { .dst = buf, .cap = cap };
 *   es_submit(p, &r, "experts/L000/E0007.exp");
 *   ... submit more, do compute ...
 *   es_wait(p, &r);   // r.err == 0, r.bytes == file size
 */
#ifndef ES_IO_H
#define ES_IO_H

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

typedef struct es_pool es_pool;

typedef struct es_req {
    /* set by caller before es_submit */
    void  *dst;          /* 4096-aligned (use es_alloc) */
    size_t cap;          /* bytes available at dst */
    /* set by the pool */
    size_t bytes;        /* file size == bytes read on success */
    atomic_int err;      /* 0 or an errno value */
    int    fd;
    int    done;         /* guarded by the pool mutex */
    atomic_int remaining; /* chunks still in flight */
    uint64_t t_submit_ns;
    uint64_t t_done_ns;
} es_req;

/* want_direct: 1 = try O_DIRECT (falls back to buffered if the FS refuses). */
es_pool *es_pool_create(int nthreads, size_t chunk_bytes, int want_direct);
void     es_pool_destroy(es_pool *p);
int      es_pool_is_direct(const es_pool *p);

/* Returns 0 or an errno. On error nothing is queued and r->err is set.
 * Call from one submitting thread (the inference thread). */
int es_submit(es_pool *p, es_req *r, const char *path);
/* Blocks until every chunk of r has finished. Returns r->err. */
int es_wait(es_pool *p, es_req *r);

/* Helpers */
void    *es_alloc(size_t bytes);  /* 4096-aligned, free() to release */
uint64_t es_now_ns(void);
/* 1 if O_DIRECT open + aligned read works on this file, else 0. */
int      es_probe_direct(const char *path);
/* Ask the kernel to drop this file from the page cache (no root needed). */
int      es_evict(const char *path);
/* Fraction [0,1] of the file's pages currently in the page cache, or -1. */
double   es_cached_fraction(const char *path);

#endif
