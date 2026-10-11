"""Pure-Python layout maths for Flash Experts (no NumPy needed, so the website build can use it too)."""
import math

PAGE = 4096


def hidden_per_page(d, page=PAGE):
    """hidden units per expert so that the expert (weights + scales + optimizer state) fills one page"""
    return max(1, (page - 8) // (16 + 2 * d))


def expert_params(d, mh):
    return 2 * mh * d


def plan_experts(target_params, d, layers=1, page=PAGE):
    """nk (keys per half) and expert count so that layers * nk^2 * params_per_expert reaches target_params"""
    mh = hidden_per_page(d, page); per = expert_params(d, mh)
    nk = max(2, int(round(math.sqrt(target_params / layers / per)))); E = nk * nk
    return dict(nk=nk, experts_per_layer=E, hidden=mh, params_per_expert=per, total_params=layers * E * per, file_bytes=layers * E * page)


def fast_cores():
    """number of 'big' CPU cores (max clock within 80% of the fastest). Phones mix fast and slow cores; spreading the maths evenly over all
    of them makes the fast ones wait for the slow ones, so the trainer uses only the fast ones. Falls back to all cores."""
    import os
    n = os.cpu_count() or 1; f = []
    for i in range(n):
        try: f.append(int(open('/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq' % i).read()))
        except Exception: return n
    top = max(f) if f else 0
    return max(1, sum(1 for x in f if x >= 0.8 * top)) if top else n


def set_threads():
    """call before NumPy is imported: one BLAS thread per fast core unless the user already chose (OPENBLAS_NUM_THREADS etc.)"""
    import os
    k = str(fast_cores())
    for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, k)
    return int(os.environ['OPENBLAS_NUM_THREADS'])
