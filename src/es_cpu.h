/* CPU topology, thread pinning and feature detection (Linux/Android). */
#ifndef ES_CPU_H
#define ES_CPU_H

#define ES_MAX_CPUS 64

typedef struct {
    int  ncpu;
    int  part[ES_MAX_CPUS];     /* MIDR part number, e.g. 0xd05 = Cortex-A55 */
    int  impl[ES_MAX_CPUS];     /* MIDR implementer, 0x41 = Arm, 0x51 = Qualcomm */
    long max_khz[ES_MAX_CPUS];  /* 0 if unreadable */
    int  allowed[ES_MAX_CPUS];  /* 1 if this process may run there */
    int  order[ES_MAX_CPUS];    /* cpu ids, fastest first */
} es_topo;

int         es_topo_read(es_topo *t);
const char *es_part_name(int impl, int part);
int         es_auto_threads(void);  /* number of fast cores (the "big" cluster), 1..8 */
int         es_pin_self(int cpu);   /* 0 on success, else errno */
int         es_has_dotprod(void);   /* sdot/udot (asimddp) */
int         es_has_i8mm(void);      /* smmla (not needed, informational) */

#endif
