"""AI model providers: 50+ hosted model APIs reached through the gateway with an API key, next to the cloud's own computers.

A model id "PROVIDER/MODEL" (groq/llama-3.3-70b-versatile, openai/gpt-4.1-mini, openrouter/meta-llama/llama-3.3-70b-instruct:free)
goes to that provider instead of a node. Every provider here speaks the OpenAI chat completions API (Anthropic, Gemini, Cohere and
the rest through their OpenAI-compatible endpoints), so one client serves them all, with streaming.

Keys:
* a user's own key (console > AI providers): any model of that provider, billed by the provider to the user;
* the owner's key (dashboard > AI providers): only the models marked free here, for every user. The owner pays nothing for those.

Keys are encrypted at rest (Fernet) with a secret kept outside the database: EXPERTCLOUD_SECRET, else ~/.expertcloud/provider_secret
(mode 600, made on first use). A copy of expertcloud.db alone does not reveal them. Keys are never sent back to a browser.

Prices are list prices in US dollars per million tokens (input, output), used to estimate spend and enforce a user's monthly budget;
providers change them, so the owner can correct any of them in providers.json next to the database (same shape as PROVIDERS; a
provider or model it names replaces the default, new ones are added). A price of None means "not known": tokens are still counted.
"""
import copy
import ipaddress
import json
import os
import re
import time
from urllib.parse import urlparse

import httpx

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:   # the gateway still runs; saving provider keys says what to install
    Fernet = InvalidToken = None


def M(id, price=None, free=False, ctx=None, note=""):
    """A model: id at the provider, (input, output) USD per 1M tokens, free on the provider's free tier, context window."""
    return dict(id=id, price_in=price[0] if price else None, price_out=price[1] if price else None, free=free, ctx=ctx, note=note)


# kind: "global" (worldwide), "china" (mainland China accounts), "india"; free_tier: what a new account gets without paying.
# fields: values the user enters besides the key, substituted into base_url (checked against FIELD_RE, so they cannot change the host
# part beyond a sub-domain label). auth: how the key is sent ("bearer" or a header name). drop: parameters the provider refuses.
PROVIDERS = {
    "openai": dict(name="OpenAI", base_url="https://api.openai.com/v1", signup="https://platform.openai.com/api-keys", free_tier="",
                   models=[M("gpt-4.1", (2.0, 8.0), ctx=1047576), M("gpt-4.1-mini", (0.4, 1.6), ctx=1047576), M("gpt-4.1-nano", (0.1, 0.4), ctx=1047576),
                           M("gpt-4o", (2.5, 10.0), ctx=128000), M("gpt-4o-mini", (0.15, 0.6), ctx=128000), M("o4-mini", (1.1, 4.4), ctx=200000)]),
    "anthropic": dict(name="Anthropic (Claude)", base_url="https://api.anthropic.com/v1", signup="https://console.anthropic.com/settings/keys", free_tier="",
                      drop=["temperature", "top_p", "top_k", "seed"],
                      models=[M("claude-opus-5-5", (4.0, 20.0), ctx=1000000), M("claude-sonnet-5-5", (2.0, 10.0), ctx=1000000),
                              M("claude-haiku-5-5", (0.10, 0.50), ctx=1000000), M("claude-fable-5-1", (10.0, 50.0), ctx=1000000)]),
    "google": dict(name="Google Gemini", base_url="https://generativelanguage.googleapis.com/v1beta/openai", signup="https://aistudio.google.com/apikey",
                   free_tier="free tier with daily limits on Flash models",
                   models=[M("gemini-2.5-flash", (0.30, 2.50), free=True, ctx=1048576), M("gemini-2.5-flash-lite", (0.10, 0.40), free=True, ctx=1048576),
                           M("gemini-2.0-flash", (0.10, 0.40), free=True, ctx=1048576), M("gemini-2.5-pro", (1.25, 10.0), ctx=1048576)]),
    "groq": dict(name="Groq", base_url="https://api.groq.com/openai/v1", signup="https://console.groq.com/keys", free_tier="free tier, rate limited; very fast (LPU)",
                 models=[M("llama-3.3-70b-versatile", (0.59, 0.79), free=True, ctx=131072), M("llama-3.1-8b-instant", (0.05, 0.08), free=True, ctx=131072),
                         M("openai/gpt-oss-120b", (0.15, 0.75), free=True, ctx=131072), M("openai/gpt-oss-20b", (0.10, 0.50), free=True, ctx=131072),
                         M("qwen/qwen3-32b", (0.29, 0.59), free=True, ctx=131072)]),
    "cerebras": dict(name="Cerebras", base_url="https://api.cerebras.ai/v1", signup="https://cloud.cerebras.ai", free_tier="free tier, rate limited; very fast (wafer-scale)",
                     models=[M("llama-3.3-70b", free=True, ctx=65536), M("llama3.1-8b", free=True, ctx=8192), M("gpt-oss-120b", free=True, ctx=65536),
                             M("qwen-3-32b", free=True, ctx=65536)]),
    "sambanova": dict(name="SambaNova Cloud", base_url="https://api.sambanova.ai/v1", signup="https://cloud.sambanova.ai/apis", free_tier="free tier, rate limited; fast (RDU)",
                      models=[M("Meta-Llama-3.3-70B-Instruct", free=True), M("Meta-Llama-3.1-8B-Instruct", free=True), M("DeepSeek-R1", free=True), M("DeepSeek-V3-0324", free=True)]),
    "openrouter": dict(name="OpenRouter", base_url="https://openrouter.ai/api/v1", signup="https://openrouter.ai/keys", free_tier="models ending in :free cost nothing",
                       models=[M("meta-llama/llama-3.3-70b-instruct:free", (0, 0), free=True), M("deepseek/deepseek-chat-v3-0324:free", (0, 0), free=True),
                               M("google/gemma-3-27b-it:free", (0, 0), free=True), M("qwen/qwen3-235b-a22b:free", (0, 0), free=True),
                               M("mistralai/mistral-small-3.1-24b-instruct:free", (0, 0), free=True), M("openai/gpt-4o-mini", (0.15, 0.6))]),
    "mistral": dict(name="Mistral AI", base_url="https://api.mistral.ai/v1", signup="https://console.mistral.ai/api-keys", free_tier="free Experiment plan, rate limited",
                    models=[M("mistral-small-latest", (0.1, 0.3), free=True), M("open-mistral-nemo", (0.15, 0.15), free=True), M("codestral-latest", (0.3, 0.9), free=True),
                            M("mistral-medium-latest", (0.4, 2.0)), M("mistral-large-latest", (2.0, 6.0))]),
    "cohere": dict(name="Cohere", base_url="https://api.cohere.ai/compatibility/v1", signup="https://dashboard.cohere.com/api-keys", free_tier="free trial key, rate limited",
                   models=[M("command-a-03-2025", (2.5, 10.0), free=True), M("command-r-plus-08-2024", (2.5, 10.0), free=True), M("command-r-08-2024", (0.15, 0.6), free=True),
                           M("command-r7b-12-2024", (0.0375, 0.15), free=True)]),
    "deepseek": dict(name="DeepSeek", base_url="https://api.deepseek.com/v1", signup="https://platform.deepseek.com/api_keys", free_tier="",
                     models=[M("deepseek-chat", (0.27, 1.10), ctx=65536), M("deepseek-reasoner", (0.55, 2.19), ctx=65536)]),
    "xai": dict(name="xAI (Grok)", base_url="https://api.x.ai/v1", signup="https://console.x.ai", free_tier="",
                models=[M("grok-4", (3.0, 15.0)), M("grok-3", (3.0, 15.0)), M("grok-3-mini", (0.3, 0.5))]),
    "together": dict(name="Together AI", base_url="https://api.together.xyz/v1", signup="https://api.together.ai/settings/api-keys", free_tier="a few free models",
                     models=[M("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", (0, 0), free=True), M("deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", (0, 0), free=True),
                             M("meta-llama/Llama-3.3-70B-Instruct-Turbo", (0.88, 0.88)), M("Qwen/Qwen2.5-72B-Instruct-Turbo", (1.2, 1.2))]),
    "fireworks": dict(name="Fireworks AI", base_url="https://api.fireworks.ai/inference/v1", signup="https://fireworks.ai/account/api-keys", free_tier="starter credit",
                      models=[M("accounts/fireworks/models/llama-v3p3-70b-instruct", (0.9, 0.9)), M("accounts/fireworks/models/llama-v3p1-8b-instruct", (0.2, 0.2)),
                              M("accounts/fireworks/models/deepseek-v3", (0.9, 0.9))]),
    "deepinfra": dict(name="DeepInfra", base_url="https://api.deepinfra.com/v1/openai", signup="https://deepinfra.com/dash/api_keys", free_tier="",
                      models=[M("meta-llama/Llama-3.3-70B-Instruct", (0.23, 0.40)), M("meta-llama/Meta-Llama-3.1-8B-Instruct", (0.03, 0.05)), M("Qwen/Qwen2.5-72B-Instruct")]),
    "perplexity": dict(name="Perplexity", base_url="https://api.perplexity.ai", signup="https://www.perplexity.ai/settings/api", free_tier="", drop=["seed", "top_k"],
                       models=[M("sonar", (1.0, 1.0), note="answers with web search"), M("sonar-pro", (3.0, 15.0), note="answers with web search"), M("sonar-reasoning")]),
    "huggingface": dict(name="Hugging Face Inference", base_url="https://router.huggingface.co/v1", signup="https://huggingface.co/settings/tokens",
                        free_tier="monthly free credit",
                        models=[M("meta-llama/Llama-3.1-8B-Instruct", free=True), M("Qwen/Qwen2.5-72B-Instruct", free=True), M("deepseek-ai/DeepSeek-V3-0324", free=True),
                                M("mistralai/Mistral-7B-Instruct-v0.3", free=True)]),
    "github": dict(name="GitHub Models", base_url="https://models.github.ai/inference", signup="https://github.com/settings/tokens", free_tier="free with a GitHub token, rate limited",
                   models=[M("openai/gpt-4.1-mini", free=True), M("openai/gpt-4o-mini", free=True), M("meta/Llama-3.3-70B-Instruct", free=True),
                           M("deepseek/DeepSeek-V3-0324", free=True), M("mistral-ai/mistral-small-2503", free=True)]),
    "nvidia": dict(name="NVIDIA NIM", base_url="https://integrate.api.nvidia.com/v1", signup="https://build.nvidia.com", free_tier="free developer credits",
                   models=[M("meta/llama-3.3-70b-instruct", free=True), M("deepseek-ai/deepseek-r1", free=True), M("nvidia/llama-3.1-nemotron-70b-instruct", free=True),
                           M("qwen/qwen2.5-coder-32b-instruct", free=True)]),
    "cloudflare": dict(name="Cloudflare Workers AI", base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1", signup="https://dash.cloudflare.com/profile/api-tokens",
                       free_tier="10,000 free units a day", fields=["account_id"],
                       models=[M("@cf/meta/llama-3.3-70b-instruct-fp8-fast", free=True), M("@cf/meta/llama-3.1-8b-instruct", free=True), M("@cf/qwen/qwen2.5-coder-32b-instruct", free=True)]),
    "ollama": dict(name="Ollama Cloud", base_url="https://ollama.com/v1", signup="https://ollama.com/settings/keys", free_tier="free tier, usage limited",
                   models=[M("gpt-oss:120b", free=True), M("gpt-oss:20b", free=True), M("deepseek-v3.1:671b", free=True)]),
    "meta": dict(name="Meta Llama API", base_url="https://api.llama.com/compat/v1", signup="https://llama.developer.meta.com", free_tier="free preview",
                 models=[M("Llama-4-Maverick-17B-128E-Instruct-FP8", free=True), M("Llama-4-Scout-17B-16E-Instruct-FP8", free=True), M("Llama-3.3-70B-Instruct", free=True)]),
    "vercel": dict(name="Vercel AI Gateway", base_url="https://ai-gateway.vercel.sh/v1", signup="https://vercel.com/dashboard/ai-gateway", free_tier="monthly free credit",
                   models=[M("openai/gpt-4.1-mini", free=True), M("anthropic/claude-haiku-5-5", free=True), M("google/gemini-2.5-flash", free=True)]),
    "azure": dict(name="Azure OpenAI", base_url="https://{resource}.openai.azure.com/openai/v1", signup="https://portal.azure.com", free_tier="", fields=["resource"], auth="api-key",
                  models=[M("gpt-4.1", (2.0, 8.0), note="your deployment name"), M("gpt-4o-mini", (0.15, 0.6), note="your deployment name")]),
    "bedrock": dict(name="Amazon Bedrock", base_url="https://bedrock-runtime.{region}.amazonaws.com/openai/v1", signup="https://console.aws.amazon.com/bedrock",
                    free_tier="", fields=["region"], models=[M("openai.gpt-oss-120b-1:0"), M("openai.gpt-oss-20b-1:0")]),
    "moonshot": dict(name="Moonshot AI (Kimi)", base_url="https://api.moonshot.ai/v1", signup="https://platform.moonshot.ai/console/api-keys", free_tier="",
                     models=[M("kimi-k2-0711-preview", (0.6, 2.5)), M("moonshot-v1-8k"), M("moonshot-v1-128k")]),
    "zai": dict(name="Z.ai (GLM)", base_url="https://api.z.ai/api/paas/v4", signup="https://z.ai/manage-apikey/apikey-list", free_tier="GLM Flash models are free",
                models=[M("glm-4.5-flash", (0, 0), free=True), M("glm-4.5-air", (0.2, 1.1)), M("glm-4.5", (0.6, 2.2))]),
    "qwen": dict(name="Alibaba Cloud Qwen (DashScope)", base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", signup="https://modelstudio.console.alibabacloud.com",
                 free_tier="free quota for new accounts", models=[M("qwen-turbo", free=True), M("qwen-plus", free=True), M("qwen-max", free=True), M("qwen2.5-coder-32b-instruct", free=True)]),
    "minimax": dict(name="MiniMax", base_url="https://api.minimax.io/v1", signup="https://www.minimax.io/platform", free_tier="",
                    models=[M("MiniMax-M1"), M("MiniMax-Text-01")]),
    "siliconflow": dict(name="SiliconFlow", base_url="https://api.siliconflow.com/v1", signup="https://cloud.siliconflow.com/account/ak", free_tier="several models are free",
                        models=[M("Qwen/Qwen2.5-7B-Instruct", (0, 0), free=True), M("THUDM/glm-4-9b-chat", (0, 0), free=True), M("deepseek-ai/DeepSeek-V3")]),
    "novita": dict(name="Novita AI", base_url="https://api.novita.ai/v3/openai", signup="https://novita.ai/settings/key-management", free_tier="starter credit",
                   models=[M("meta-llama/llama-3.3-70b-instruct"), M("meta-llama/llama-3.1-8b-instruct"), M("deepseek/deepseek-v3-0324")]),
    "hyperbolic": dict(name="Hyperbolic", base_url="https://api.hyperbolic.xyz/v1", signup="https://app.hyperbolic.xyz/settings", free_tier="starter credit",
                       models=[M("meta-llama/Llama-3.3-70B-Instruct"), M("Qwen/Qwen2.5-72B-Instruct"), M("deepseek-ai/DeepSeek-V3")]),
    "nebius": dict(name="Nebius AI Studio", base_url="https://api.studio.nebius.com/v1", signup="https://studio.nebius.com/settings/api-keys", free_tier="starter credit",
                   models=[M("meta-llama/Llama-3.3-70B-Instruct"), M("meta-llama/Meta-Llama-3.1-8B-Instruct"), M("Qwen/Qwen2.5-72B-Instruct")]),
    "scaleway": dict(name="Scaleway Generative APIs", base_url="https://api.scaleway.ai/v1", signup="https://console.scaleway.com/iam/api-keys",
                     free_tier="first million tokens free", models=[M("llama-3.3-70b-instruct", free=True), M("llama-3.1-8b-instruct", free=True), M("mistral-nemo-instruct-2407", free=True)]),
    "ovhcloud": dict(name="OVHcloud AI Endpoints", base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1", signup="https://endpoints.ai.cloud.ovh.net",
                     free_tier="free, rate limited", models=[M("Meta-Llama-3_3-70B-Instruct", free=True), M("Mistral-Nemo-Instruct-2407", free=True)]),
    "ai21": dict(name="AI21 Labs (Jamba)", base_url="https://api.ai21.com/studio/v1", signup="https://studio.ai21.com/account/api-key", free_tier="free trial credit",
                 models=[M("jamba-large", (2.0, 8.0)), M("jamba-mini", (0.2, 0.4))]),
    "upstage": dict(name="Upstage (Solar)", base_url="https://api.upstage.ai/v1", signup="https://console.upstage.ai/api-keys", free_tier="free trial credit",
                    models=[M("solar-pro"), M("solar-mini")]),
    "reka": dict(name="Reka", base_url="https://api.reka.ai/v1", signup="https://platform.reka.ai/apikeys", free_tier="", models=[M("reka-core"), M("reka-flash")]),
    "writer": dict(name="Writer (Palmyra)", base_url="https://api.writer.com/v1", signup="https://app.writer.com", free_tier="", models=[M("palmyra-x5"), M("palmyra-x-004")]),
    "friendli": dict(name="FriendliAI", base_url="https://api.friendli.ai/serverless/v1", signup="https://friendli.ai/suite", free_tier="starter credit",
                     models=[M("meta-llama-3.3-70b-instruct"), M("meta-llama-3.1-8b-instruct")]),
    "baseten": dict(name="Baseten Model APIs", base_url="https://inference.baseten.co/v1", signup="https://app.baseten.co/settings/api_keys", free_tier="starter credit",
                    models=[M("deepseek-ai/DeepSeek-V3-0324"), M("meta-llama/Llama-4-Maverick-17B-128E-Instruct")]),
    "featherless": dict(name="Featherless", base_url="https://api.featherless.ai/v1", signup="https://featherless.ai/account/api-keys", free_tier="flat monthly subscription",
                        models=[M("meta-llama/Meta-Llama-3.1-8B-Instruct"), M("Qwen/Qwen2.5-72B-Instruct")]),
    "chutes": dict(name="Chutes", base_url="https://llm.chutes.ai/v1", signup="https://chutes.ai/app/api", free_tier="", models=[M("deepseek-ai/DeepSeek-V3-0324"), M("Qwen/Qwen3-235B-A22B")]),
    "kluster": dict(name="kluster.ai", base_url="https://api.kluster.ai/v1", signup="https://platform.kluster.ai/apikeys", free_tier="starter credit",
                    models=[M("klusterai/Meta-Llama-3.3-70B-Instruct-Turbo"), M("deepseek-ai/DeepSeek-V3-0324")]),
    "inference": dict(name="Inference.net", base_url="https://api.inference.net/v1", signup="https://inference.net/dashboard/api-keys", free_tier="starter credit",
                      models=[M("meta-llama/llama-3.3-70b-instruct/fp-8"), M("meta-llama/llama-3.1-8b-instruct/fp-8")]),
    "parasail": dict(name="Parasail", base_url="https://api.parasail.io/v1", signup="https://www.saas.parasail.io/keys", free_tier="starter credit",
                     models=[M("parasail-llama-33-70b-fp8"), M("parasail-deepseek-r1")]),
    "venice": dict(name="Venice AI", base_url="https://api.venice.ai/api/v1", signup="https://venice.ai/settings/api", free_tier="", models=[M("llama-3.3-70b"), M("qwen3-235b")]),
    "aimlapi": dict(name="AI/ML API", base_url="https://api.aimlapi.com/v1", signup="https://aimlapi.com/app/keys", free_tier="free tier, rate limited",
                    models=[M("gpt-4o-mini", free=True), M("meta-llama/Llama-3.3-70B-Instruct-Turbo", free=True)]),
    "stepfun": dict(name="StepFun", base_url="https://api.stepfun.com/v1", signup="https://platform.stepfun.com", free_tier="", kind="china", models=[M("step-1-8k"), M("step-2-16k")]),
    "baichuan": dict(name="Baichuan", base_url="https://api.baichuan-ai.com/v1", signup="https://platform.baichuan-ai.com", free_tier="", kind="china", models=[M("Baichuan4-Turbo"), M("Baichuan4")]),
    "yi": dict(name="01.AI (Yi)", base_url="https://api.lingyiwanwu.com/v1", signup="https://platform.lingyiwanwu.com", free_tier="", kind="china", models=[M("yi-lightning")]),
    "volcengine": dict(name="Volcengine Ark (Doubao)", base_url="https://ark.cn-beijing.volces.com/api/v3", signup="https://console.volcengine.com/ark", free_tier="free quota for new accounts",
                       kind="china", models=[M("doubao-1-5-pro-32k-250115", note="or your endpoint id"), M("doubao-1-5-lite-32k-250115", note="or your endpoint id")]),
    "hunyuan": dict(name="Tencent Hunyuan", base_url="https://api.hunyuan.cloud.tencent.com/v1", signup="https://console.cloud.tencent.com/hunyuan", free_tier="hunyuan-lite is free",
                    kind="china", models=[M("hunyuan-lite", (0, 0), free=True), M("hunyuan-turbo"), M("hunyuan-large")]),
    "qianfan": dict(name="Baidu Qianfan (ERNIE)", base_url="https://qianfan.baidubce.com/v2", signup="https://console.bce.baidu.com/iam/#/iam/apikey/list", free_tier="ERNIE Speed and Lite are free",
                    kind="china", models=[M("ernie-speed-128k", (0, 0), free=True), M("ernie-lite-8k", (0, 0), free=True), M("ernie-4.0-8k")]),
    "sarvam": dict(name="Sarvam AI", base_url="https://api.sarvam.ai/v1", signup="https://dashboard.sarvam.ai", free_tier="free credit; Indian languages", kind="india",
                   models=[M("sarvam-m", free=True)]),
    "krutrim": dict(name="Krutrim Cloud", base_url="https://cloud.olakrutrim.com/v1", signup="https://cloud.olakrutrim.com", free_tier="starter credit; Indian languages", kind="india",
                    models=[M("Krutrim-spectre-v2"), M("Meta-Llama-3-8B-Instruct")]),
}
FIELD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,62}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")
MAX_BUDGET_USD = 100000


class ProviderError(Exception):
    def __init__(self, status, message):
        super().__init__(message); self.status = status


def load(path=None):
    """The catalog, with providers.json (if present) applied on top."""
    cat = copy.deepcopy(PROVIDERS)
    if path and os.path.exists(path):
        with open(path) as f:
            override = json.load(f)
        for pid, p in override.items():
            if not ID_RE.match(pid): continue
            cur = cat.setdefault(pid, dict(name=pid, base_url="", signup="", free_tier="", models=[]))
            models = {m["id"]: m for m in cur["models"]}
            for m in p.pop("models", []) or []:
                if m.get("id") in models: models[m["id"]].update(m)
                else: models[m.get("id")] = dict(M(m.get("id")), **m)
            cur.update(p); cur["models"] = list(models.values())
    for pid, p in cat.items():
        p.setdefault("kind", "global"); p.setdefault("fields", []); p.setdefault("auth", "bearer"); p.setdefault("drop", [])
    return cat


def public_view(cat):
    """The catalog for the console and GET /providers (no keys, nothing secret)."""
    out = []
    for pid, p in cat.items():
        out.append(dict(id=pid, name=p["name"], signup=p["signup"], free_tier=p["free_tier"], kind=p["kind"], fields=p["fields"],
                        models=[dict(m, full_id=pid + "/" + m["id"]) for m in p["models"]], free_models=sum(1 for m in p["models"] if m.get("free"))))
    return sorted(out, key=lambda x: (x["kind"] != "global", -x["free_models"], x["name"].lower()))


def split(model, cat):
    """(provider id, model at the provider) for 'provider/model', else (None, None)."""
    if "/" not in model: return None, None
    pid, rest = model.split("/", 1)
    return (pid, rest) if pid in cat and rest else (None, None)


def find_model(cat, pid, mid):
    return next((m for m in cat[pid]["models"] if m["id"] == mid), None)


def cost(model_info, prompt_tokens, completion_tokens):
    """Estimated USD, or None when the price is not known."""
    if not model_info or model_info.get("price_in") is None or model_info.get("price_out") is None: return None
    return (prompt_tokens * float(model_info["price_in"]) + completion_tokens * float(model_info["price_out"])) / 1e6


def base_url(p, fields):
    url = p["base_url"]
    for f in p.get("fields") or []:
        v = str((fields or {}).get(f) or "")
        if not FIELD_RE.match(v): raise ProviderError(400, "%s needs %s (letters, digits and -)" % (p["name"], f))
        url = url.replace("{%s}" % f, v)
    u = urlparse(url)
    if u.scheme != "https" or not u.hostname: raise ProviderError(400, "provider address must be https")
    try:
        if not ipaddress.ip_address(u.hostname).is_global: raise ProviderError(400, "provider address must be public")
    except ValueError:
        pass   # a host name, not an address
    return url.rstrip("/")


# ---------------------------------------------------------------------------------------------------------------- secrets
class Vault:
    """Encrypts provider keys with a secret that is not in the database."""

    def __init__(self, secret_path=None):
        self.f = None
        if Fernet is None: return
        secret = os.environ.get("EXPERTCLOUD_SECRET", "").strip()
        if not secret:
            path = secret_path or os.path.expanduser("~/.expertcloud/provider_secret")
            if os.path.exists(path):
                with open(path) as fh: secret = fh.read().strip()
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                secret = Fernet.generate_key().decode()
                with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as fh: fh.write(secret)
        try:
            self.f = Fernet(secret.encode())
        except ValueError:   # any other string: derive a Fernet key from it
            import base64, hashlib
            self.f = Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()))

    def ready(self):
        return self.f is not None

    def seal(self, text):
        if not self.f: raise ProviderError(503, "provider keys need the cryptography package on the gateway: pip install cryptography")
        return self.f.encrypt(text.encode()).decode()

    def open(self, blob):
        if not self.f: raise ProviderError(503, "provider keys need the cryptography package on the gateway: pip install cryptography")
        try:
            return self.f.decrypt(blob.encode()).decode()
        except InvalidToken:
            raise ProviderError(500, "a saved provider key cannot be read: the gateway's secret changed (EXPERTCLOUD_SECRET); save the key again")


def hint(key):
    key = key.strip()
    return (key[:3] + "…" + key[-4:]) if len(key) > 10 else "…"


# ---------------------------------------------------------------------------------------------------------------- calling a provider
def headers(p, key):
    h = {"Content-Type": "application/json"}
    if p.get("auth", "bearer") == "bearer": h["Authorization"] = "Bearer " + key
    else: h[p["auth"]] = key
    if base_url_host(p) == "openrouter.ai": h.update({"HTTP-Referer": "https://github.com/tharunmaks/stream-", "X-Title": "ExpertCloud"})
    return h


def base_url_host(p):
    return urlparse(p["base_url"]).hostname


def error_text(r):
    try:
        j = r.json(); e = j.get("error") if isinstance(j, dict) else None
        msg = (e.get("message") if isinstance(e, dict) else e) or j.get("message") or j.get("detail")
    except ValueError:
        msg = None
    return str(msg or r.text[:200] or "HTTP %d" % r.status_code)[:300]


async def check_key(p, key, fields):
    """(ok, message): lists the provider's models with the key (no tokens used)."""
    url = base_url(p, fields)
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(url + "/models", headers=headers(p, key))
    except httpx.HTTPError as e:
        return False, "cannot reach %s: %s" % (p["name"], e.__class__.__name__)
    if r.status_code in (401, 403): return False, "%s refused the key: %s" % (p["name"], error_text(r))
    if r.status_code >= 400 and r.status_code != 404: return False, "%s answered HTTP %d: %s" % (p["name"], r.status_code, error_text(r))
    return True, "key works"


async def run(p, key, fields, model, messages, params, timeout=600.0):
    """Async generator like Controller.run: ("delta", text) ... then ("done", info). Streams from the provider."""
    url = base_url(p, fields) + "/chat/completions"
    body = dict(model=model, messages=messages, stream=True, stream_options={"include_usage": True})
    for k, v in params.items():
        if k not in p.get("drop", []): body[k] = v
    t0, first, usage, text_len = time.time(), None, {}, 0
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=20)) as c:
            async with c.stream("POST", url, headers=headers(p, key), json=body) as r:
                if r.status_code >= 400:
                    await r.aread()
                    raise ProviderError(r.status_code if r.status_code in (400, 401, 402, 403, 404, 429) else 502,
                                        "%s: %s" % (p["name"], error_text(r)))
                async for line in r.aiter_lines():
                    if not line.startswith("data:"): continue
                    data = line[5:].strip()
                    if data == "[DONE]": break
                    try: ev = json.loads(data)
                    except ValueError: continue
                    if ev.get("usage"): usage = ev["usage"]
                    for ch in ev.get("choices") or []:
                        d = (ch.get("delta") or {}).get("content")
                        if d:
                            if first is None: first = time.time() - t0
                            text_len += len(d)
                            yield ("delta", d)
    except httpx.HTTPError as e:
        raise ProviderError(502, "cannot reach %s: %s" % (p["name"], e.__class__.__name__))
    gen_s = max(1e-6, time.time() - t0 - (first or 0))
    ct = int(usage.get("completion_tokens") or 0)
    yield ("done", dict(node=p["name"], model=model, usage=usage, ttft=first, tps=(ct / gen_s) if ct else None))

