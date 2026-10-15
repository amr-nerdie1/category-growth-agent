"""Deterministic opportunity scorer. The LLM never computes a score; it only reads these.

Each category gets one verdict:
  GROW              demand is growing, competition is fragmented, margin survives freight and returns
  WATCH             decent on some signals, not all
  OVERSATURATED     big but locked up by top brands / Amazon, with flat demand
  MARGIN_TRAP       demand is growing, but heavy freight + returns leave too little contribution margin
  ALREADY_PRESENT   we already sell here; this is a "deepen" decision, not "enter"
  INSUFFICIENT_DATA too few products / sellers to judge; the agent must say so instead of guessing
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PRODUCTS = 30          # below this, market metrics are noise
MIN_SEARCH = 2_000
MARGIN_FLOOR = 0.08        # landed contribution margin we need to enter a category
FREIGHT_PER_LB = 0.00055   # share of price lost per lb of weight (synthetic)


def landed_margin(r) -> float:
    """Our expected CM after heavy-freight and return drag."""
    freight_drag = min(r.avg_weight_lb * FREIGHT_PER_LB, 0.12)
    return round(r.our_cm_pct - freight_drag - 0.5 * r.est_return_rate, 3)


def score(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["evidence_id"] = "E-" + d.node_id
    d["landed_cm_pct"] = d.apply(landed_margin, axis=1)
    d["fragmentation"] = (1 - d.top3_brand_share).round(3)
    d["search_per_seller"] = (d.monthly_search_volume / d.seller_count).round(0)

    def z(s):
        return (s - s.mean()) / (s.std() or 1)

    demand = z(np.log1p(d.monthly_search_volume))
    growth = z(d.search_growth_yoy.clip(-0.2, 0.6))
    comp = z(d.fragmentation) - z(d.amazon_share)
    margin = z(d.landed_cm_pct)
    d["opportunity_score"] = (0.20 * demand + 0.30 * growth + 0.25 * comp + 0.25 * margin).round(2)

    def verdict(r):
        if r.product_count < MIN_PRODUCTS or r.monthly_search_volume < MIN_SEARCH:
            return "INSUFFICIENT_DATA"
        if r.top3_brand_share >= 0.5 and r.search_growth_yoy < 0.08:
            return "OVERSATURATED"
        if r.our_skus > 0:
            return "ALREADY_PRESENT"
        if r.landed_cm_pct < MARGIN_FLOOR:
            return "MARGIN_TRAP"
        if r.search_growth_yoy >= 0.15 and r.top3_brand_share <= 0.30 and r.opportunity_score > 0.4:
            return "GROW"
        return "WATCH"

    d["verdict"] = d.apply(verdict, axis=1)
    return d.sort_values("opportunity_score", ascending=False).reset_index(drop=True)


PUBLIC_COLS = ["evidence_id", "node_id", "subtree", "category", "verdict", "opportunity_score",
               "monthly_search_volume", "search_growth_yoy", "product_count", "seller_count",
               "top3_brand_share", "amazon_share", "avg_price", "avg_weight_lb", "est_return_rate",
               "our_skus", "our_cm_pct", "landed_cm_pct"]
