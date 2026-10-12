#define _GNU_SOURCE
#include "es_fetch.h"

#include <errno.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

/* run curl with args; its stdout comes back on *rfd. Returns the pid, or -1 */
static pid_t run_curl(char *const *argv, int *rfd) {
    int p[2];
    if (pipe(p)) return -1;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, p[1], 1);
    posix_spawn_file_actions_addclose(&fa, p[0]);
    posix_spawn_file_actions_addclose(&fa, p[1]);
    pid_t pid;
    const int r = posix_spawnp(&pid, "curl", &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(p[1]);
    if (r) { close(p[0]); return -1; }
    *rfd = p[0];
    return pid;
}
static int wait_ok(pid_t pid) {
    int st;
    while (waitpid(pid, &st, 0) < 0) if (errno != EINTR) return 0;
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}
static const char *auth_header(char *buf, int cap) {
    const char *t = getenv("HF_TOKEN");
    if (!t || !*t) return NULL;
    snprintf(buf, (size_t)cap, "Authorization: Bearer %s", t);
    return buf;
}

int64_t es_http_size(const char *url) {
    char auth[600];
    const char *h = auth_header(auth, sizeof auth);
    char *argv[12]; int n = 0;
    argv[n++] = "curl"; argv[n++] = "-sSfIL"; argv[n++] = "--retry"; argv[n++] = "3";
    if (h) { argv[n++] = "-H"; argv[n++] = (char *)h; }
    argv[n++] = (char *)url; argv[n] = NULL;
    int fd; const pid_t pid = run_curl(argv, &fd);
    if (pid < 0) return -1;
    FILE *f = fdopen(fd, "r");
    char line[2048]; int64_t size = -1;
    while (f && fgets(line, sizeof line, f)) {   /* the last Content-Length wins: redirects come first */
        if (!strncasecmp(line, "content-length:", 15)) size = strtoll(line + 15, NULL, 10);
        if (!strncasecmp(line, "x-linked-size:", 14)) size = strtoll(line + 14, NULL, 10);
    }
    if (f) fclose(f); else close(fd);
    return wait_ok(pid) ? size : -1;
}

int es_http_range(const char *url, uint64_t off, uint64_t len, int out, uint64_t out_off) {
    if (!len) return 0;
    char range[64], auth[600];
    snprintf(range, sizeof range, "%llu-%llu", (unsigned long long)off, (unsigned long long)(off + len - 1));
    const char *h = auth_header(auth, sizeof auth);
    for (int attempt = 0; attempt < 3; attempt++) {
        char *argv[14]; int n = 0;
        argv[n++] = "curl"; argv[n++] = "-sfL"; argv[n++] = "--retry"; argv[n++] = "3"; argv[n++] = "-r"; argv[n++] = range;
        if (h) { argv[n++] = "-H"; argv[n++] = (char *)h; }
        argv[n++] = (char *)url; argv[n] = NULL;
        int fd; const pid_t pid = run_curl(argv, &fd);
        if (pid < 0) return -1;
        static __thread char buf[1 << 20];
        uint64_t got = 0; int bad = 0;
        for (;;) {
            const ssize_t k = read(fd, buf, sizeof buf);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) break;
            if (got + (uint64_t)k > len) { bad = 1; break; }   /* the server ignored the range */
            for (ssize_t w = 0; w < k;) {
                const ssize_t x = pwrite(out, buf + w, (size_t)(k - w), (off_t)(out_off + got + (uint64_t)w));
                if (x <= 0) { bad = 1; break; }
                w += x;
            }
            if (bad) break;
            got += (uint64_t)k;
        }
        close(fd);
        const int ok = wait_ok(pid);
        if (ok && !bad && got == len) return 0;
    }
    return -1;
}

void es_hf_url(char *out, int cap, const char *repo, const char *file) {
    const char *ep = getenv("HF_ENDPOINT");
    snprintf(out, (size_t)cap, "%s/%s/resolve/main/%s", ep && *ep ? ep : "https://huggingface.co", repo, file);
}
