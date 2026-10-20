"""Plans of the two products sold through ExpertCloud, in Indian rupees, monthly.

ExpertCloud plans set the limits the gateway enforces (requests a day and a minute, answer length, API keys, database space,
queue priority). ExpertStream plans are sold and tracked here for the ExpertStream app, which reads a user's plan through the
API (GET /account/plans) and applies its own features.

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
                        "indexes": 5, "snapshots": 1, "priority": 2},
             "features": ["25 AI requests a day", "Answers up to 256 tokens", "1 API key", "1 GB database (3 databases)", "1 backup per database", "Playground"]},
            {"id": "cloud_pro", "name": "Pro", "price_inr": 1700,
             "limits": {"requests_per_day": 2000, "requests_per_min": 60, "max_tokens": 4096, "api_keys": 10, "db_mb": 51200, "databases": 50,
                        "indexes": 50, "snapshots": 10, "priority": 1},
             "features": ["2,000 AI requests a day", "Answers up to 4,096 tokens", "10 API keys", "50 GB database (50 databases)", "10 backups per database", "Faster queue, all models"]},
            {"id": "cloud_max", "name": "Max", "price_inr": 4000,
             "limits": {"requests_per_day": 10000, "requests_per_min": 200, "max_tokens": 8192, "api_keys": 50, "db_mb": 1048576, "databases": 500,
                        "indexes": 500, "snapshots": 50, "priority": 0},
             "features": ["10,000 AI requests a day", "Answers up to 8,192 tokens", "50 API keys", "1 TB database (500 databases)", "50 backups per database", "Front of the queue, all models"]},
        ],
    },
    "expertstream": {
        "title": "ExpertStream",
        "plans": [
            {"id": "es_free", "name": "Free", "price_inr": 0, "limits": {}, "features": ["The ExpertStream app", "Basic features"]},
            {"id": "es_plus", "name": "Plus", "price_inr": 800, "limits": {}, "features": ["Everything in Free", "More usage", "Premium features"]},
            {"id": "es_pro", "name": "Pro", "price_inr": 2000, "limits": {}, "features": ["Everything in Plus", "Highest usage", "All features"]},
        ],
    },
}
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
