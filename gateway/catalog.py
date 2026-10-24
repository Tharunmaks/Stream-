"""The ExpertCloud model catalog: 54 open models this cloud runs on its own computers, like a hosted inference service.

Every model is a real GGUF file on Hugging Face (Q4_K_M quantization, checked to exist when the list was written). The owner deploys a
model to a computer with one click (dashboard > Model catalog, or POST /admin/catalog/{id}/deploy), and from then on anyone calls it
by its short id, for example "llama-3.3-70b" or "qwen3-8b", with an sk-expert- key.

Pricing, in Indian rupees per million tokens:
* free models (8B parameters and smaller) cost nothing: the caller's plan limits (requests a day and a minute) apply;
* bigger models are billed per token from the caller's prepaid credits (bought in the console, paid with Razorpay).

Prices are starting points. To change them, or to add or hide models, put a catalog.json next to the database:
{"llama-3.3-70b": {"price_in": 40, "price_out": 80}, "my-model": {"name": ..., "repo": ..., "file": ..., "size_gb": ...}, "phi-4": null}
"""
import copy
import json
import os
import re

# price bands by download size (Q4_K_M): (largest size in GB, rupees per 1M input tokens, per 1M output tokens)
BANDS = [(6.0, 0, 0), (11.0, 8, 16), (21.0, 20, 40), (1e9, 50, 100)]


def _m(id, name, family, params, repo, file, size_bytes, ctx, tags=()):
    gb = round(size_bytes / 1e9, 2)
    _, pin, pout = next(b for b in BANDS if gb <= b[0])
    return id, dict(id=id, name=name, family=family, params=params, repo=repo, file=file, size_gb=gb, ctx=ctx, tags=list(tags),
                    free=pin == 0 and pout == 0, price_in=pin, price_out=pout)


B = "bartowski/"
MODELS = dict([
    _m("smollm2-135m", "SmolLM2 135M Instruct", "SmolLM", "135M", B + "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q4_K_M.gguf", 105454432, 8192, ["chat", "tiny"]),
    _m("smollm2-360m", "SmolLM2 360M Instruct", "SmolLM", "360M", B + "SmolLM2-360M-Instruct-GGUF", "SmolLM2-360M-Instruct-Q4_K_M.gguf", 270590880, 8192, ["chat", "tiny"]),
    _m("smollm2-1.7b", "SmolLM2 1.7B Instruct", "SmolLM", "1.7B", B + "SmolLM2-1.7B-Instruct-GGUF", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf", 1055609824, 8192, ["chat"]),
    _m("qwen2.5-0.5b", "Qwen2.5 0.5B Instruct", "Qwen", "0.5B", B + "Qwen2.5-0.5B-Instruct-GGUF", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", 397808192, 32768, ["chat", "tiny"]),
    _m("qwen2.5-1.5b", "Qwen2.5 1.5B Instruct", "Qwen", "1.5B", B + "Qwen2.5-1.5B-Instruct-GGUF", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", 986048768, 32768, ["chat"]),
    _m("qwen2.5-3b", "Qwen2.5 3B Instruct", "Qwen", "3B", B + "Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q4_K_M.gguf", 1929903264, 32768, ["chat"]),
    _m("qwen2.5-7b", "Qwen2.5 7B Instruct", "Qwen", "7B", B + "Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf", 4683074240, 32768, ["chat"]),
    _m("qwen2.5-14b", "Qwen2.5 14B Instruct", "Qwen", "14B", B + "Qwen2.5-14B-Instruct-GGUF", "Qwen2.5-14B-Instruct-Q4_K_M.gguf", 8988110976, 32768, ["chat"]),
    _m("qwen2.5-32b", "Qwen2.5 32B Instruct", "Qwen", "32B", B + "Qwen2.5-32B-Instruct-GGUF", "Qwen2.5-32B-Instruct-Q4_K_M.gguf", 19851336576, 32768, ["chat"]),
    _m("qwen2.5-72b", "Qwen2.5 72B Instruct", "Qwen", "72B", B + "Qwen2.5-72B-Instruct-GGUF", "Qwen2.5-72B-Instruct-Q4_K_M.gguf", 47415715488, 32768, ["chat"]),
    _m("qwen2.5-coder-1.5b", "Qwen2.5 Coder 1.5B", "Qwen", "1.5B", B + "Qwen2.5-Coder-1.5B-Instruct-GGUF", "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf", 986048800, 32768, ["code"]),
    _m("qwen2.5-coder-7b", "Qwen2.5 Coder 7B", "Qwen", "7B", B + "Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf", 4683074336, 32768, ["code"]),
    _m("qwen2.5-coder-14b", "Qwen2.5 Coder 14B", "Qwen", "14B", B + "Qwen2.5-Coder-14B-Instruct-GGUF", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", 8988111072, 32768, ["code"]),
    _m("qwen2.5-coder-32b", "Qwen2.5 Coder 32B", "Qwen", "32B", B + "Qwen2.5-Coder-32B-Instruct-GGUF", "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf", 19851336672, 32768, ["code"]),
    _m("qwen2.5-math-7b", "Qwen2.5 Math 7B", "Qwen", "7B", B + "Qwen2.5-Math-7B-Instruct-GGUF", "Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf", 4683074272, 4096, ["math"]),
    _m("qwen3-0.6b", "Qwen3 0.6B", "Qwen", "0.6B", B + "Qwen_Qwen3-0.6B-GGUF", "Qwen_Qwen3-0.6B-Q4_K_M.gguf", 484220320, 32768, ["chat", "reasoning", "tiny"]),
    _m("qwen3-1.7b", "Qwen3 1.7B", "Qwen", "1.7B", B + "Qwen_Qwen3-1.7B-GGUF", "Qwen_Qwen3-1.7B-Q4_K_M.gguf", 1282439584, 32768, ["chat", "reasoning"]),
    _m("qwen3-4b", "Qwen3 4B", "Qwen", "4B", B + "Qwen_Qwen3-4B-GGUF", "Qwen_Qwen3-4B-Q4_K_M.gguf", 2497280960, 32768, ["chat", "reasoning"]),
    _m("qwen3-8b", "Qwen3 8B", "Qwen", "8B", B + "Qwen_Qwen3-8B-GGUF", "Qwen_Qwen3-8B-Q4_K_M.gguf", 5027784224, 32768, ["chat", "reasoning"]),
    _m("qwen3-14b", "Qwen3 14B", "Qwen", "14B", B + "Qwen_Qwen3-14B-GGUF", "Qwen_Qwen3-14B-Q4_K_M.gguf", 9001753632, 32768, ["chat", "reasoning"]),
    _m("qwen3-32b", "Qwen3 32B", "Qwen", "32B", B + "Qwen_Qwen3-32B-GGUF", "Qwen_Qwen3-32B-Q4_K_M.gguf", 19762149696, 32768, ["chat", "reasoning"]),
    _m("qwen3-30b-a3b", "Qwen3 30B A3B (mixture of experts)", "Qwen", "30B (3B active)", B + "Qwen_Qwen3-30B-A3B-GGUF", "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf", 18632184480, 32768, ["chat", "reasoning", "moe"]),
    _m("qwq-32b", "QwQ 32B", "Qwen", "32B", B + "Qwen_QwQ-32B-GGUF", "Qwen_QwQ-32B-Q4_K_M.gguf", 19851336512, 32768, ["reasoning"]),
    _m("llama-3.2-1b", "Llama 3.2 1B Instruct", "Llama", "1B", B + "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", 807694464, 131072, ["chat"]),
    _m("llama-3.2-3b", "Llama 3.2 3B Instruct", "Llama", "3B", B + "Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf", 2019377696, 131072, ["chat"]),
    _m("llama-3.1-8b", "Llama 3.1 8B Instruct", "Llama", "8B", B + "Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 4920739232, 131072, ["chat"]),
    _m("llama-3.3-70b", "Llama 3.3 70B Instruct", "Llama", "70B", B + "Llama-3.3-70B-Instruct-GGUF", "Llama-3.3-70B-Instruct-Q4_K_M.gguf", 42520398816, 131072, ["chat"]),
    _m("hermes-3-llama-3.1-8b", "Hermes 3 (Llama 3.1 8B)", "Llama", "8B", B + "Hermes-3-Llama-3.1-8B-GGUF", "Hermes-3-Llama-3.1-8B-Q4_K_M.gguf", 4920734624, 131072, ["chat", "tools"]),
    _m("gemma-3-1b", "Gemma 3 1B", "Gemma", "1B", B + "google_gemma-3-1b-it-GGUF", "google_gemma-3-1b-it-Q4_K_M.gguf", 806058496, 32768, ["chat"]),
    _m("gemma-3-4b", "Gemma 3 4B", "Gemma", "4B", B + "google_gemma-3-4b-it-GGUF", "google_gemma-3-4b-it-Q4_K_M.gguf", 2489758112, 131072, ["chat"]),
    _m("gemma-3-12b", "Gemma 3 12B", "Gemma", "12B", B + "google_gemma-3-12b-it-GGUF", "google_gemma-3-12b-it-Q4_K_M.gguf", 7300575264, 131072, ["chat"]),
    _m("gemma-3-27b", "Gemma 3 27B", "Gemma", "27B", B + "google_gemma-3-27b-it-GGUF", "google_gemma-3-27b-it-Q4_K_M.gguf", 16546404992, 131072, ["chat"]),
    _m("gemma-2-2b", "Gemma 2 2B", "Gemma", "2B", B + "gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf", 1708582752, 8192, ["chat"]),
    _m("gemma-2-9b", "Gemma 2 9B", "Gemma", "9B", B + "gemma-2-9b-it-GGUF", "gemma-2-9b-it-Q4_K_M.gguf", 5761057728, 8192, ["chat"]),
    _m("phi-3.5-mini", "Phi-3.5 mini Instruct", "Phi", "3.8B", B + "Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf", 2393232672, 131072, ["chat"]),
    _m("phi-4", "Phi-4", "Phi", "14B", B + "phi-4-GGUF", "phi-4-Q4_K_M.gguf", 9053114816, 16384, ["chat", "reasoning"]),
    _m("mistral-7b", "Mistral 7B Instruct v0.3", "Mistral", "7B", B + "Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", 4372812000, 32768, ["chat"]),
    _m("ministral-8b", "Ministral 8B Instruct", "Mistral", "8B", B + "Ministral-8B-Instruct-2410-GGUF", "Ministral-8B-Instruct-2410-Q4_K_M.gguf", 4911500096, 32768, ["chat"]),
    _m("mistral-nemo-12b", "Mistral Nemo 12B Instruct", "Mistral", "12B", B + "Mistral-Nemo-Instruct-2407-GGUF", "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf", 7477208192, 131072, ["chat"]),
    _m("mistral-small-24b", "Mistral Small 24B Instruct", "Mistral", "24B", B + "Mistral-Small-24B-Instruct-2501-GGUF", "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf", 14333908672, 32768, ["chat"]),
    _m("deepseek-r1-distill-qwen-1.5b", "DeepSeek R1 Distill Qwen 1.5B", "DeepSeek", "1.5B", B + "DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", 1117320800, 131072, ["reasoning"]),
    _m("deepseek-r1-distill-qwen-7b", "DeepSeek R1 Distill Qwen 7B", "DeepSeek", "7B", B + "DeepSeek-R1-Distill-Qwen-7B-GGUF", "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf", 4683073504, 131072, ["reasoning"]),
    _m("deepseek-r1-distill-llama-8b", "DeepSeek R1 Distill Llama 8B", "DeepSeek", "8B", B + "DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf", 4920736608, 131072, ["reasoning"]),
    _m("deepseek-r1-distill-qwen-14b", "DeepSeek R1 Distill Qwen 14B", "DeepSeek", "14B", B + "DeepSeek-R1-Distill-Qwen-14B-GGUF", "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf", 8988110240, 131072, ["reasoning"]),
    _m("deepseek-r1-distill-qwen-32b", "DeepSeek R1 Distill Qwen 32B", "DeepSeek", "32B", B + "DeepSeek-R1-Distill-Qwen-32B-GGUF", "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf", 19851335840, 131072, ["reasoning"]),
    _m("deepseek-coder-v2-lite", "DeepSeek Coder V2 Lite (mixture of experts)", "DeepSeek", "16B (2.4B active)", B + "DeepSeek-Coder-V2-Lite-Instruct-GGUF", "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf", 10364416768, 131072, ["code", "moe"]),
    _m("gpt-oss-20b", "gpt-oss 20B (mixture of experts)", "OpenAI", "21B (3.6B active)", B + "openai_gpt-oss-20b-GGUF", "openai_gpt-oss-20b-Q4_K_M.gguf", 11673418816, 131072, ["chat", "reasoning", "moe"]),
    _m("olmoe-1b-7b", "OLMoE 1B-7B (mixture of experts)", "AllenAI", "7B (1B active)", B + "OLMoE-1B-7B-0924-Instruct-GGUF", "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf", 4213513024, 4096, ["chat", "moe"]),
    _m("granite-3.1-2b", "Granite 3.1 2B Instruct", "IBM Granite", "2B", B + "granite-3.1-2b-instruct-GGUF", "granite-3.1-2b-instruct-Q4_K_M.gguf", 1545295424, 131072, ["chat", "tools"]),
    _m("granite-3.1-8b", "Granite 3.1 8B Instruct", "IBM Granite", "8B", B + "granite-3.1-8b-instruct-GGUF", "granite-3.1-8b-instruct-Q4_K_M.gguf", 4942858720, 131072, ["chat", "tools"]),
    _m("aya-expanse-8b", "Aya Expanse 8B (23 languages)", "Cohere", "8B", B + "aya-expanse-8b-GGUF", "aya-expanse-8b-Q4_K_M.gguf", 5056982720, 8192, ["chat", "multilingual"]),
    _m("falcon3-7b", "Falcon3 7B Instruct", "Falcon", "7B", B + "Falcon3-7B-Instruct-GGUF", "Falcon3-7B-Instruct-Q4_K_M.gguf", 4569726400, 32768, ["chat"]),
    _m("internlm2.5-7b", "InternLM2.5 7B Chat", "InternLM", "7B", B + "internlm2_5-7b-chat-GGUF", "internlm2_5-7b-chat-Q4_K_M.gguf", 4712768640, 32768, ["chat"]),
    _m("yi-1.5-9b", "Yi 1.5 9B Chat", "Yi", "9B", B + "Yi-1.5-9B-Chat-GGUF", "Yi-1.5-9B-Chat-Q4_K_M.gguf", 5328957952, 4096, ["chat"]),
])


def load(path=None):
    """The catalog, with catalog.json (if present) applied: a model it names is changed or added; null hides one."""
    cat = copy.deepcopy(MODELS)
    if path and os.path.exists(path):
        with open(path) as f:
            override = json.load(f)
        for mid, m in override.items():
            if m is None: cat.pop(mid, None); continue
            cur = cat.setdefault(mid, dict(id=mid, name=mid, family="", params="", repo="", file="", size_gb=0, ctx=None, tags=[], free=False,
                                           price_in=0, price_out=0))
            cur.update(m)
            cur["free"] = not (cur.get("price_in") or cur.get("price_out"))
    for m in cat.values():
        m["served_as"] = served_name(m["file"])
        m["min_ram_gb"] = round(m["size_gb"] * 1.15 + 1, 1)   # weights plus room for the context
    return cat


def served_name(file):
    """The name a computer reports for a GGUF file: the file name without .gguf (and without a -00001-of-0000N shard suffix)."""
    base = os.path.basename(file or "")
    if base.endswith(".gguf"): base = base[:-5]
    return re.sub(r"-\d{5}-of-\d{5}$", "", base)


def lookup(cat, model):
    """The catalog entry for a model id or the name a computer serves it under, else None."""
    if model in cat: return cat[model]
    return next((m for m in cat.values() if m["served_as"] == model), None)


def cost_paise(m, prompt_tokens, completion_tokens):
    """What a request costs in paise (1/100 rupee), rounded up so a paid request is never free."""
    if not m or m.get("free"): return 0
    raw = (prompt_tokens * float(m["price_in"]) + completion_tokens * float(m["price_out"])) / 1e6 * 100
    return int(raw) + (1 if raw > int(raw) else 0)


def pick_node(m, nodes):
    """The best computer to deploy a model to: enough RAM and disk, not already serving it, most free RAM first."""
    need_mb, size_mb = m["min_ram_gb"] * 1024, m["size_gb"] * 1000
    ok = [n for n in nodes if m["served_as"] not in n.models and float((n.caps or {}).get("ram_mb") or 0) >= need_mb
          and float((n.caps or {}).get("disk_free_mb") or 0) >= size_mb + 2048]
    return max(ok, key=lambda n: float((n.load or {}).get("ram_free_mb") or 0), default=None)
