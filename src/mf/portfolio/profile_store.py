# src/mf/portfolio/profile_store.py

import os
import json

PROFILE_PATH = "data_cache/mf_profile.json"

DEFAULT_ASSET = "Equity"
DEFAULT_GOAL = "General"

ASSET_CLASSES = ["Equity", "Debt", "Hybrid", "Gold", "International"]
DEFAULT_GOALS = ["General", "Emergency", "House", "Marriage", "Retirement", "Education"]


def load_profile():
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_profile(profile: dict):
    os.makedirs("data_cache", exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def get_scheme_tags(profile: dict, scheme_code: str, scheme_name: str):
    """
    Return (asset_class, goal)
    """
    scode = str(scheme_code)
    row = profile.get(scode, {})
    asset = row.get("asset_class", DEFAULT_ASSET)
    goal = row.get("goal", DEFAULT_GOAL)
    # extra safety
    if asset not in ASSET_CLASSES:
        asset = DEFAULT_ASSET
    if not isinstance(goal, str) or not goal.strip():
        goal = DEFAULT_GOAL
    return asset, goal
