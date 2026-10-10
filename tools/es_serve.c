/*
 * es_serve: run the ExpertStream web app on the phone itself.
 *
 * Starts one es_chat engine (-J line protocol) and serves:
 *   GET  /             the web app (web/index.html)
 *   GET  /api/status   model info once the engine is ready
 *   POST /api/chat     body = prompt text; streams JSON lines (tokens, stages)
 *   POST /api/reset    start a new conversation
 *
 * Listens on 127.0.0.1 only, so other devices on the network can't use it.
 *
 *   ./es_serve -g ~/models/model.gguf      straight from a GGUF, no import
 *   ./es_serve -m ~/packs/olmoe            then open http://127.0.0.1:8080
 *   ./es_serve -m ~/packs/olmoe -- -C 512  extra options go to es_chat
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static pthread_mutex_t g_engine = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_to, *g_from;          /* pipes to/from es_chat */
static char g_ready[2048];           /* the engine's "ready" line */
static char g_web[PATH_MAX];

static int send_all(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n) {
        ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1;
        p += k;
        n -= (size_t)k;
    }
    return 0;
}

static void reply(int fd, int code, const char *type, const char *body, size_t n) {
    char h[512];
    int hl = snprintf(h, sizeof h,
                      "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                      "Cache-Control: no-store\r\nConnection: close\r\n\r\n",
                      code, code == 200 ? "OK" : code == 404 ? "Not Found" : "Error", type, n);
    send_all(fd, h, (size_t)hl);
    send_all(fd, body, n);
}

static int chunk(int fd, const char *s, size_t n) {
    char h[32];
    int hl = snprintf(h, sizeof h, "%zx\r\n", n);
    return send_all(fd, h, (size_t)hl) || send_all(fd, s, n) || send_all(fd, "\r\n", 2);
}

/* send a line to the engine and stream its JSON lines back until "done" */
static void engine_turn(int fd, const char *line) {
    pthread_mutex_lock(&g_engine);
    const char *h = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"
                    "Cache-Control: no-store\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    int client_ok = send_all(fd, h, strlen(h)) == 0;
    fprintf(g_to, "%s\n", line);
    fflush(g_to);
    static char buf[1 << 16];
    while (fgets(buf, sizeof buf, g_from)) {
        if (client_ok && chunk(fd, buf, strlen(buf))) client_ok = 0; /* keep draining */
        if (!strncmp(buf, "{\"ev\":\"done\"", 12)) break;
    }
    if (client_ok) send_all(fd, "0\r\n\r\n", 5);
    pthread_mutex_unlock(&g_engine);
}

static void serve_file(int fd, const char *name, const char *type) {
    char path[PATH_MAX];
    snprintf(path, sizeof path, "%s/%s", g_web, name);
    FILE *f = fopen(path, "rb");
    if (!f) { reply(fd, 404, "text/plain", "not found\n", 10); return; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n);
    size_t got = fread(b, 1, (size_t)n, f);
    fclose(f);
    reply(fd, 200, type, b, got);
    free(b);
}

static void *client(void *arg) {
    int fd = (int)(intptr_t)arg;
    enum { REQ_MAX = 1 << 17 };
    char *req = malloc(REQ_MAX), *line = malloc(REQ_MAX);
    size_t n = 0;
    char *body = NULL;
    long clen = 0;
    while (n < REQ_MAX - 1) {
        ssize_t k = recv(fd, req + n, REQ_MAX - 1 - n, 0);
        if (k <= 0) break;
        n += (size_t)k;
        req[n] = 0;
        char *e = strstr(req, "\r\n\r\n");
        if (!e) continue;
        body = e + 4;
        char *cl = strcasestr(req, "\r\ncontent-length:");
        clen = cl ? atol(cl + 17) : 0;
        if (clen < 0 || clen > (long)(REQ_MAX - 1 - (size_t)(body - req))) clen = 0;
        while ((long)(n - (size_t)(body - req)) < clen) {
            k = recv(fd, req + n, REQ_MAX - 1 - n, 0);
            if (k <= 0) break;
            n += (size_t)k;
        }
        req[n] = 0;
        break;
    }
    if (!body) { close(fd); free(req); free(line); return NULL; }
    char method[8] = "", path[256] = "";
    sscanf(req, "%7s %255s", method, path);
    char *q = strchr(path, '?');
    if (q) *q = 0;

    if (!strcmp(method, "GET") && (!strcmp(path, "/") || !strcmp(path, "/index.html"))) {
        serve_file(fd, "index.html", "text/html; charset=utf-8");
    } else if (!strcmp(method, "GET") && !strcmp(path, "/api/status")) {
        reply(fd, 200, "application/json", g_ready, strlen(g_ready));
    } else if (!strcmp(method, "POST") && !strcmp(path, "/api/chat")) {
        /* one line for the engine: escape backslashes and newlines */
        size_t o = 0;
        for (long i = 0; i < clen && o + 3 < REQ_MAX; i++) {
            char ch = body[i];
            if (ch == '\r') continue;
            if (ch == '\\') { line[o++] = '\\'; line[o++] = '\\'; }
            else if (ch == '\n') { line[o++] = '\\'; line[o++] = 'n'; }
            else line[o++] = ch;
        }
        line[o] = 0;
        if (o == 0 || line[0] == '/') reply(fd, 400, "text/plain", "empty prompt\n", 13);
        else engine_turn(fd, line);
    } else if (!strcmp(method, "POST") && !strcmp(path, "/api/reset")) {
        engine_turn(fd, "/reset");
    } else {
        reply(fd, 404, "text/plain", "not found\n", 10);
    }
    close(fd);
    free(req);
    free(line);
    return NULL;
}

int main(int argc, char **argv) {
    const char *pack = NULL, *flag = "-m";
    int port = 8080, ai = 1;
    for (; ai < argc; ai++) {
        if (!strcmp(argv[ai], "-m") && ai + 1 < argc) { pack = argv[++ai]; flag = "-m"; }
        else if (!strcmp(argv[ai], "-g") && ai + 1 < argc) { pack = argv[++ai]; flag = "-g"; }
        else if (!strcmp(argv[ai], "-p") && ai + 1 < argc) port = atoi(argv[++ai]);
        else if (!strcmp(argv[ai], "-w") && ai + 1 < argc) snprintf(g_web, sizeof g_web, "%s", argv[++ai]);
        else if (!strcmp(argv[ai], "--")) { ai++; break; }
        else { fprintf(stderr, "usage: es_serve (-m PACKDIR | -g FILE.gguf) [-p port=8080] [-w webdir] [-- es_chat options]\n"); return 2; }
    }
    if (!pack) { fprintf(stderr, "usage: es_serve (-m PACKDIR | -g FILE.gguf) [-p port=8080] [-w webdir] [-- es_chat options]\n"); return 2; }

    /* es_chat and web/ live next to this binary */
    char self[PATH_MAX], dir[PATH_MAX], chat[PATH_MAX + 16];
    ssize_t sl = readlink("/proc/self/exe", self, sizeof self - 1);
    if (sl <= 0) { fprintf(stderr, "cannot find my own path\n"); return 1; }
    self[sl] = 0;
    snprintf(dir, sizeof dir, "%s", self);
    *strrchr(dir, '/') = 0;
    snprintf(chat, sizeof chat, "%s/es_chat", dir);
    if (!g_web[0]) snprintf(g_web, sizeof g_web, "%s/web", dir);

    int to[2], from[2];
    if (pipe(to) || pipe(from)) { perror("pipe"); return 1; }
    pid_t pid = fork();
    if (pid == 0) {
        dup2(to[0], 0);
        dup2(from[1], 1);
        close(to[1]);
        close(from[0]);
        char *args[64];
        int k = 0;
        args[k++] = chat; args[k++] = (char *)flag; args[k++] = (char *)pack; args[k++] = "-J";
        args[k++] = "-n"; args[k++] = "512";   /* room for code answers; later options override */
        for (int i = ai; i < argc && k < 62; i++) args[k++] = argv[i];
        args[k] = NULL;
        execv(chat, args);
        perror(chat);
        _exit(127);
    }
    close(to[0]);
    close(from[1]);
    g_to = fdopen(to[1], "w");
    g_from = fdopen(from[0], "r");
    signal(SIGPIPE, SIG_IGN);

    fprintf(stderr, "Loading %s ...\n", pack);
    static char line[4096];
    if (!fgets(line, sizeof line, g_from) || strncmp(line, "{\"ev\":\"ready\"", 13)) {
        fprintf(stderr, "the engine did not start (see the message above)\n");
        return 1;
    }
    snprintf(g_ready, sizeof g_ready, "%s", line);

    int s = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(s, (struct sockaddr *)&a, sizeof a) || listen(s, 16)) {
        fprintf(stderr, "cannot listen on 127.0.0.1:%d: %s\n", port, strerror(errno));
        return 1;
    }
    fprintf(stderr, "ExpertStream is ready: open http://127.0.0.1:%d in your phone's browser (Ctrl+C to stop)\n", port);
    for (;;) {
        int c = accept(s, NULL, NULL);
        if (c < 0) { if (errno == EINTR) continue; perror("accept"); break; }
        pthread_t th;
        if (pthread_create(&th, NULL, client, (void *)(intptr_t)c) == 0) pthread_detach(th);
        else close(c);
        int st;
        if (waitpid(pid, &st, WNOHANG) == pid) { fprintf(stderr, "the engine stopped\n"); break; }
    }
    return 0;
}
