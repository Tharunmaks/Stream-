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
