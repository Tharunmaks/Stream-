#define _GNU_SOURCE
#include "es_compute.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

#include "es_cpu.h"

#if defined(__EMSCRIPTEN__)
#define SPIN_ITERS 4000   /* browsers: sleep early, other tabs and the page itself need the cores */
#else
#define SPIN_ITERS 200000 /* ~50-100 us of polling before sleeping */
#endif

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield");
#elif defined(__x86_64__)
    __asm__ volatile("pause");
#endif
}

typedef struct {
    es_cpool *p;
    int tid, cpu;
} wctx;

struct es_cpool {
    int n;
    pthread_t th[ES_MAX_CPUS];
    wctx ctx[ES_MAX_CPUS];
    atomic_uint gen;
    atomic_int pending;
    atomic_int sleepers;
    atomic_int stop;
    es_task fn;
    void *arg;
    pthread_mutex_t mu;
    pthread_cond_t cv;
};

static void *worker(void *a) {
    wctx *c = a;
    es_cpool *p = c->p;
    if (c->cpu >= 0) es_pin_self(c->cpu);
    unsigned seen = 0;
    for (;;) {
        unsigned g;
        int spins = 0;
        while ((g = atomic_load(&p->gen)) == seen) {
            if (++spins < SPIN_ITERS) {
                cpu_relax();
                continue;
            }
            pthread_mutex_lock(&p->mu);
            atomic_fetch_add(&p->sleepers, 1);
            while (atomic_load(&p->gen) == seen) pthread_cond_wait(&p->cv, &p->mu);
            atomic_fetch_sub(&p->sleepers, 1);
            pthread_mutex_unlock(&p->mu);
            spins = 0;
        }
        seen = g;
        if (atomic_load(&p->stop)) return NULL;
        p->fn(c->tid, p->n, p->arg);
        atomic_fetch_sub(&p->pending, 1);
    }
}

es_cpool *es_cpool_create(int n, const int *cpus) {
    if (n < 1 || n > ES_MAX_CPUS) return NULL;
    es_cpool *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->n = n;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    if (cpus) es_pin_self(cpus[0]);
    for (int i = 1; i < n; i++) {
        p->ctx[i] = (wctx){p, i, cpus ? cpus[i] : -1};
        pthread_create(&p->th[i], NULL, worker, &p->ctx[i]);
    }
    return p;
}

static void kick(es_cpool *p) {
    atomic_fetch_add(&p->gen, 1);
    if (atomic_load(&p->sleepers) > 0) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
    }
}

void es_cpool_run(es_cpool *p, es_task fn, void *arg) {
    p->fn = fn;
    p->arg = arg;
    atomic_store(&p->pending, p->n - 1);
    kick(p);
    fn(0, p->n, arg);
    while (atomic_load(&p->pending) > 0) cpu_relax();
}

void es_cpool_destroy(es_cpool *p) {
    if (!p) return;
    atomic_store(&p->stop, 1);
    kick(p);
    /* a worker may be between its spin check and going to sleep: keep
     * broadcasting until everyone has left */
    for (int i = 1; i < p->n; i++) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
        pthread_join(p->th[i], NULL);
    }
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    free(p);
}
