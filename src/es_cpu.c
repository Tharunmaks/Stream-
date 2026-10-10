#define _GNU_SOURCE
#include "es_cpu.h"

#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sys/auxv.h>
#endif

const char *es_part_name(int impl, int part) {
    if (impl == 0x41) {
        switch (part) {
        case 0xd03: return "Cortex-A53";
        case 0xd04: return "Cortex-A35";
        case 0xd05: return "Cortex-A55";
        case 0xd07: return "Cortex-A57";
        case 0xd08: return "Cortex-A72";
        case 0xd09: return "Cortex-A73";
        case 0xd0a: return "Cortex-A75";
        case 0xd0b: return "Cortex-A76";
        case 0xd0d: return "Cortex-A77";
        case 0xd41: return "Cortex-A78";
        case 0xd44: return "Cortex-X1";
        case 0xd46: return "Cortex-A510";
        case 0xd47: return "Cortex-A710";
        case 0xd48: return "Cortex-X2";
        case 0xd4d: return "Cortex-A715";
        case 0xd4e: return "Cortex-X3";
        case 0xd80: return "Cortex-A520";
        case 0xd81: return "Cortex-A720";
        case 0xd82: return "Cortex-X4";
        }
    } else if (impl == 0x51) {
        switch (part) {
        case 0x800: return "Kryo Gold (A73)";
        case 0x801: return "Kryo Silver (A53)";
        case 0x802: return "Kryo Gold (A75)";
        case 0x803: return "Kryo Silver (A55)";
        case 0x804: return "Kryo Gold (A76)";
        case 0x805: return "Kryo Silver (A55)";
        }
    }
    return "unknown";
}

static long read_long(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    long v = 0;
    if (fscanf(f, "%ld", &v) != 1) v = 0;
    fclose(f);
    return v;
}

static int part_score(int impl, int part);
int es_topo_read(es_topo *t) {
    memset(t, 0, sizeof(*t));
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return -1;
    char line[256];
    int cur = -1;
    while (fgets(line, sizeof line, f)) {
        int v;
        if (sscanf(line, "processor : %d", &v) == 1) {
            cur = v;
            if (cur >= ES_MAX_CPUS) cur = -1;
            else if (cur + 1 > t->ncpu) t->ncpu = cur + 1;
        } else if (cur >= 0 && sscanf(line, "CPU part : %i", &v) == 1) {
            t->part[cur] = v;
        } else if (cur >= 0 && sscanf(line, "CPU implementer : %i", &v) == 1) {
            t->impl[cur] = v;
        }
    }
    fclose(f);

#if defined(__EMSCRIPTEN__)
    int have_mask = 0;   /* no CPU affinity in the browser */
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    int have_mask = sched_getaffinity(0, sizeof set, &set) == 0;
#endif
    for (int c = 0; c < t->ncpu; c++) {
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        t->max_khz[c] = read_long(path);
#if defined(__EMSCRIPTEN__)
        t->allowed[c] = 1;
        (void)have_mask;
#else
        t->allowed[c] = have_mask ? CPU_ISSET(c, &set) : 1;
#endif
        t->order[c] = c;
    }
    /* fastest first: by max frequency, then higher cpu id (big cores are
     * usually numbered last on Android) */
    for (int i = 1; i < t->ncpu; i++)
        for (int j = i; j > 0; j--) {
            int a = t->order[j - 1], b = t->order[j];
            const int sa = part_score(t->impl[a], t->part[a]), sb = part_score(t->impl[b], t->part[b]);
            int swap = t->max_khz[b] > t->max_khz[a] ||
                       (t->max_khz[b] == t->max_khz[a] && (sb > sa || (sb == sa && b > a)));
            if (!swap) break;
            t->order[j - 1] = b;
            t->order[j] = a;
        }
    return 0;
}

/* rough performance class of a core from its MIDR part number (higher = faster); used when cpufreq is unreadable */
static int part_score(int impl, int part) {
    if (impl == 0x41) switch (part) {
        case 0xd03: case 0xd04: case 0xd05: case 0xd46: case 0xd80: return 1;   /* A53 A34 A55 A510 A520 */
        case 0xd07: case 0xd08: case 0xd09: return 2;                          /* A57 A72 A73 */
        case 0xd0a: case 0xd0b: case 0xd0d: case 0xd41: case 0xd47: return 3;   /* A75 A76 A77 A78 A710 */
        case 0xd4d: case 0xd81: return 4;                                       /* A715 A720 */
        case 0xd44: case 0xd48: case 0xd4b: case 0xd4e: case 0xd82: case 0xd84: case 0xd85: return 5;   /* X1 X2 X3 X4 */
        default: return 3;
    }
    return 3;
}

int es_auto_threads(void) {
    es_topo t;
    if (es_topo_read(&t) != 0 || t.ncpu < 1) return 4;
    int best = -1, n = 0;
    for (int i = 0; i < t.ncpu; i++) if (t.allowed[t.order[i]]) { best = t.order[i]; break; }
    if (best < 0) return 4;
    for (int c = 0; c < t.ncpu; c++) {
        if (!t.allowed[c]) continue;
        const int fast = t.max_khz[best] > 0 && t.max_khz[c] > 0 ? t.max_khz[c] * 100 >= t.max_khz[best] * 75
                                                                  : part_score(t.impl[c], t.part[c]) >= part_score(t.impl[best], t.part[best]) - (t.max_khz[best] > 0 ? 0 : 0);
        if (fast) n++;
    }
    return n < 1 ? 1 : n > 8 ? 8 : n;
}

int es_pin_self(int cpu) {
#if defined(__EMSCRIPTEN__)
    (void)cpu;
    return 0;
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof set, &set) == 0 ? 0 : errno;
#endif
}

int es_has_dotprod(void) {
#if defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP) & (1UL << 20)) != 0; /* HWCAP_ASIMDDP */
#else
    return 0;
#endif
}

int es_has_i8mm(void) {
#if defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP2) & (1UL << 13)) != 0; /* HWCAP2_I8MM */
#else
    return 0;
#endif
}
