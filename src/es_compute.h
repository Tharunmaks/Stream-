/*
 * Compute thread pool for matmul. The calling thread is worker 0, so a pool
 * of 4 on the 4 big cores uses exactly those 4 cores. Idle workers spin
 * briefly (dispatch latency of a few microseconds) and then sleep, so they
 * don't burn the big cores while the engine waits on flash.
 */
#ifndef ES_COMPUTE_H
#define ES_COMPUTE_H

typedef struct es_cpool es_cpool;
typedef void (*es_task)(int tid, int nthreads, void *arg);

/* cpus[0] is applied to the calling thread; cpus may be NULL (no pinning). */
es_cpool *es_cpool_create(int nthreads, const int *cpus);
/* Run fn(tid, n, arg) on every worker (caller included); returns when all
 * have finished. */
void      es_cpool_run(es_cpool *p, es_task fn, void *arg);
void      es_cpool_destroy(es_cpool *p);

/* Even split of [0, total) into n parts: part tid is [*lo, *hi). */
static inline void es_split(int total, int tid, int n, int *lo, int *hi) {
    int per = total / n, extra = total % n;
    *lo = tid * per + (tid < extra ? tid : extra);
    *hi = *lo + per + (tid < extra);
}

#endif
