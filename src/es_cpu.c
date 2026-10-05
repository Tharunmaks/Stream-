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

    cpu_set_t set;
    CPU_ZERO(&set);
    int have_mask = sched_getaffinity(0, sizeof set, &set) == 0;
    for (int c = 0; c < t->ncpu; c++) {
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        t->max_khz[c] = read_long(path);
        t->allowed[c] = have_mask ? CPU_ISSET(c, &set) : 1;
        t->order[c] = c;
    }
    /* fastest first: by max frequency, then higher cpu id (big cores are
     * usually numbered last on Android) */
    for (int i = 1; i < t->ncpu; i++)
        for (int j = i; j > 0; j--) {
            int a = t->order[j - 1], b = t->order[j];
            int swap = t->max_khz[b] > t->max_khz[a] ||
                       (t->max_khz[b] == t->max_khz[a] && b > a);
            if (!swap) break;
            t->order[j - 1] = b;
            t->order[j] = a;
        }
    return 0;
}

int es_pin_self(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof set, &set) == 0 ? 0 : errno;
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
