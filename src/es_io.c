#define _GNU_SOURCE
#include "es_io.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "es_cpu.h"
#include "es_format.h"

#define QUEUE_CAP 4096 /* chunks; 4096 x 1 MiB = 4 GiB in flight, plenty */

typedef struct {
    es_req *req;
    uint64_t off;
    size_t len;
} chunk;

struct es_pool {
    int nthreads;
    size_t chunk_bytes;
    int direct;
    pthread_t *threads;

    pthread_mutex_t mu;
    pthread_cond_t cv_work;  /* queue became non-empty, or stop */
    pthread_cond_t cv_space; /* queue has room */
    pthread_cond_t cv_done;  /* some request finished */
    chunk q[QUEUE_CAP];
    size_t head, count;
    int stop;

    int cpus[64];       /* optional pinning targets */
    int ncpus;
    atomic_int next_cpu;
};

uint64_t es_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void *es_alloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, ES_ALIGN, es_round_up(bytes, ES_ALIGN)) != 0) return NULL;
    return p;
}

static void finish_req(es_pool *p, es_req *r) {
    if (!p->direct) posix_fadvise(r->fd, 0, 0, POSIX_FADV_DONTNEED);
    close(r->fd);
    r->fd = -1;
    r->t_done_ns = es_now_ns();
    pthread_mutex_lock(&p->mu);
    r->done = 1;
    pthread_cond_broadcast(&p->cv_done);
    pthread_mutex_unlock(&p->mu);
}

static void *worker(void *arg) {
    es_pool *p = (es_pool *)arg;
    if (p->ncpus > 0) {
        int i = atomic_fetch_add(&p->next_cpu, 1) % p->ncpus;
        es_pin_self(p->cpus[i]);
    }
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (p->count == 0 && !p->stop) pthread_cond_wait(&p->cv_work, &p->mu);
        if (p->count == 0 && p->stop) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        chunk c = p->q[p->head];
        p->head = (p->head + 1) % QUEUE_CAP;
        p->count--;
        pthread_cond_signal(&p->cv_space);
        pthread_mutex_unlock(&p->mu);

        es_req *r = c.req;
        uint8_t *dst = (uint8_t *)r->dst + c.off;
        size_t done = 0;
        while (done < c.len && atomic_load(&r->err) == 0) {
            ssize_t n = pread(r->fd, dst + done, c.len - done, (off_t)(c.off + done));
            if (n > 0) {
                done += (size_t)n;
            } else if (n == 0) {
                atomic_store(&r->err, EIO); /* file shrank under us */
            } else if (errno != EINTR) {
                atomic_store(&r->err, errno);
            }
        }
        if (atomic_fetch_sub(&r->remaining, 1) == 1) finish_req(p, r);
    }
}

es_pool *es_pool_create(int nthreads, size_t chunk_bytes, int want_direct) {
    return es_pool_create_on(nthreads, chunk_bytes, want_direct, NULL, 0);
}

es_pool *es_pool_create_on(int nthreads, size_t chunk_bytes, int want_direct,
                           const int *cpus, int ncpus) {
    if (nthreads < 1 || chunk_bytes == 0 || chunk_bytes % ES_ALIGN) return NULL;
    es_pool *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    if (ncpus > 64) ncpus = 64;
    for (int i = 0; i < ncpus; i++) p->cpus[i] = cpus[i];
    p->ncpus = cpus ? ncpus : 0;
    p->nthreads = nthreads;
    p->chunk_bytes = chunk_bytes;
    p->direct = want_direct;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    pthread_cond_init(&p->cv_space, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    p->threads = calloc((size_t)nthreads, sizeof(pthread_t));
    for (int i = 0; i < nthreads; i++) {
        if (pthread_create(&p->threads[i], NULL, worker, p) != 0) {
            p->nthreads = i;
            es_pool_destroy(p);
            return NULL;
        }
    }
    return p;
}

void es_pool_destroy(es_pool *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->stop = 1;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->nthreads; i++) pthread_join(p->threads[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_space);
    pthread_cond_destroy(&p->cv_done);
    free(p->threads);
    free(p);
}

int es_pool_is_direct(const es_pool *p) { return p->direct; }

int es_submit(es_pool *p, es_req *r, const char *path) {
    atomic_store(&r->err, 0);
    r->bytes = 0;
    r->fd = -1;
    r->done = 1; /* stays 1 if we fail before queueing anything */
    r->t_submit_ns = es_now_ns();
    r->t_done_ns = 0;
    atomic_store(&r->remaining, 0);

    int fd = open(path, O_RDONLY | O_CLOEXEC | (p->direct ? O_DIRECT : 0));
    if (fd < 0 && p->direct && errno == EINVAL) {
        fprintf(stderr, "es_io: O_DIRECT refused for %s, using buffered reads\n", path);
        p->direct = 0;
        fd = open(path, O_RDONLY | O_CLOEXEC);
    }
    if (fd < 0) {
        atomic_store(&r->err, errno);
        return errno;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        int e = errno;
        close(fd);
        atomic_store(&r->err, e);
        return e;
    }
    size_t size = (size_t)st.st_size;
    if (size == 0 || size % ES_ALIGN || size > r->cap ||
        (uintptr_t)r->dst % ES_ALIGN) {
        close(fd);
        int e = (size > r->cap) ? ENOBUFS : EINVAL;
        atomic_store(&r->err, e);
        return e;
    }
    if (!p->direct) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    size_t nchunks = (size + p->chunk_bytes - 1) / p->chunk_bytes;
    r->fd = fd;
    r->bytes = size;
    atomic_store(&r->remaining, (int)nchunks);

    pthread_mutex_lock(&p->mu);
    r->done = 0;
    for (size_t i = 0; i < nchunks; i++) {
        while (p->count == QUEUE_CAP) pthread_cond_wait(&p->cv_space, &p->mu);
        uint64_t off = (uint64_t)i * p->chunk_bytes;
        size_t len = size - off < p->chunk_bytes ? size - off : p->chunk_bytes;
        p->q[(p->head + p->count) % QUEUE_CAP] = (chunk){r, off, len};
        p->count++;
        pthread_cond_signal(&p->cv_work);
    }
    pthread_mutex_unlock(&p->mu);
    return 0;
}

int es_wait(es_pool *p, es_req *r) {
    pthread_mutex_lock(&p->mu);
    while (!r->done) pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
    return atomic_load(&r->err);
}

int es_done(es_pool *p, es_req *r) {
    pthread_mutex_lock(&p->mu);
    int d = r->done;
    pthread_mutex_unlock(&p->mu);
    return d;
}

int es_probe_direct(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0) return 0;
    void *buf = es_alloc(ES_ALIGN);
    ssize_t n = buf ? pread(fd, buf, ES_ALIGN, 0) : -1;
    free(buf);
    close(fd);
    return n == (ssize_t)ES_ALIGN;
}

int es_evict(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return errno;
    int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    return rc;
}

double es_cached_fraction(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size == 0) {
        close(fd);
        return -1;
    }
    size_t len = (size_t)st.st_size;
    void *m = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return -1;
    long pg = sysconf(_SC_PAGESIZE);
    size_t npages = (len + (size_t)pg - 1) / (size_t)pg;
    unsigned char *vec = malloc(npages);
    double frac = -1;
    if (vec && mincore(m, len, vec) == 0) {
        size_t res = 0;
        for (size_t i = 0; i < npages; i++) res += vec[i] & 1;
        frac = (double)res / (double)npages;
    }
    free(vec);
    munmap(m, len);
    return frac;
}
