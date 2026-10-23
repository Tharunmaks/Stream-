"""Plans, in Indian rupees, monthly.

ExpertCloud plans set the limits the gateway enforces (requests a day and a minute, answer length, API keys, database space,
queue priority, training). Expert One is the ExpertStream bundle sold on the ExpertStream site (site/app.py, the only place it is
bought); it includes Expert Cloud, so its tiers unlock Cloud tiers (UNLOCKS below). One account works on both websites.

To change prices or limits without touching code, put a plans.json next to the database (same shape as PLANS below; any plan
or field it names replaces the default).
"""
import copy
import json
import os

PLANS = {
    "cloud": {
        "title": "ExpertCloud",
        "plans": [
            {"id": "cloud_free", "name": "Free", "price_inr": 0,
             "limits": {"requests_per_day": 25, "requests_per_min": 5, "max_tokens": 256, "api_keys": 1, "db_mb": 1024, "databases": 3,
                        "indexes": 5, "snapshots": 1, "priority": 2, "train_steps": 100, "train_moe": False, "train_threads": 2},
             "features": ["25 AI requests a day", "Answers up to 256 tokens", "1 API key", "1 GB database (3 databases)", "1 backup per database",
                          "Training: LoRA runs up to 100 steps"]},
            {"id": "cloud_pro", "name": "Pro", "price_inr": 1700,
             "limits": {"requests_per_day": 2000, "requests_per_min": 60, "max_tokens": 4096, "api_keys": 10, "db_mb": 51200, "databases": 50,
                        "indexes": 50, "snapshots": 10, "priority": 1, "train_steps": 3000, "train_moe": True, "train_threads": 4},
             "features": ["2,000 AI requests a day", "Answers up to 4,096 tokens", "10 API keys", "50 GB database (50 databases)", "10 backups per database",
                          "Training: LoRA up to 3,000 steps, MoE models", "Faster queue, all models"]},
            {"id": "cloud_max", "name": "Max", "price_inr": 4000,
             "limits": {"requests_per_day": 10000, "requests_per_min": 200, "max_tokens": 8192, "api_keys": 50, "db_mb": 1048576, "databases": 500,
                        "indexes": 500, "snapshots": 50, "priority": 0, "train_steps": 20000, "train_moe": True, "train_threads": 8},
             "features": ["10,000 AI requests a day", "Answers up to 8,192 tokens", "50 API keys", "1 TB database (500 databases)", "50 backups per database",
                          "Training: LoRA up to 20,000 steps, MoE models", "Front of the queue, all models"]},
        ],
    },
    "expertone": {   # sold on the ExpertStream site; the ids are the site's (site/app.py PLANS)
        "title": "Expert One",
        "sold_on_site": True,
        "plans": [
            {"id": "es-starter", "name": "Starter (free)", "price_inr": 0, "limits": {},
             "features": ["Run MoE models streamed from disk", "MCP server, 1 API key", "Notebook: 1 job, 30 min CPU"]},
            {"id": "one-starter", "name": "Starter", "price_inr": 2000, "limits": {},
             "features": ["Expert Stream, Cloud, Build, Craft", "Includes Expert Cloud Pro", "Notebook: 2 jobs, 4 h CPU each", "Models up to 10B"]},
            {"id": "one-mid", "name": "Mid", "price_inr": 5000, "limits": {},
             "features": ["Everything in Starter", "Includes Expert Cloud Max", "Notebook: 4 jobs, 12 h CPU each", "Models up to 30B"]},
            {"id": "one-top", "name": "Top", "price_inr": 10000, "limits": {},
             "features": ["Everything in Mid", "Includes Expert Cloud Max", "Notebook: 8 jobs, 24 h CPU each", "Models up to 70B, priority queue"]},
        ],
    },
}
# Expert One includes Expert Cloud: the Cloud tier each Expert One tier unlocks
UNLOCKS = {"one-starter": "cloud_pro", "one-mid": "cloud_max", "one-top": "cloud_max"}
PERIOD_DAYS = 30
# Extra database space on top of any plan, bought in whole GB, for PERIOD_DAYS like a plan.
STORAGE = {"price_inr_per_gb": 30, "min_gb": 1, "max_gb": 100000}


def load(path=None):
    """The plans, with plans.json (if present) applied on top."""
    plans = copy.deepcopy(PLANS)
    if path and os.path.exists(path):
        with open(path) as f:
            override = json.load(f)
        for product, p in override.items():
            if product == "storage":
                STORAGE.update(p); continue
            if product not in plans: plans[product] = {"title": p.get("title", product), "plans": []}
            if "title" in p: plans[product]["title"] = p["title"]
            have = {x["id"]: x for x in plans[product]["plans"]}
            for x in p.get("plans", []):
                if x.get("id") in have: have[x["id"]].update(x)
                else: plans[product]["plans"].append(x)
    return plans


def find(plans, plan_id):
    """(product, plan) for a plan id, or (None, None)."""
    for product, p in plans.items():
        for x in p["plans"]:
            if x["id"] == plan_id: return product, x
    return None, None


def free_plan(plans, product):
    return min(plans[product]["plans"], key=lambda x: x["price_inr"])
