"""Point-anomaly detection and the anomaly register (WS2 Core/Advanced/Expert).

The register lists both
  * ERRORS that were fixed in cleaning (one entry per affected Date-SKU-Region), and
  * REAL EVENTS found in the demand series (outliers, stockouts, cycle breaks, account changes),
each with type, reason and action (fix / model / exclude / correct).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ERROR_RULES = {
    "DQ-041": ("error:sign_flip", "Negative units (sign error, |units| at normal level)", "fixed"),
    "DQ-050": ("error:price_in_cents", "Unit price keyed in cents (x100)", "fixed"),
    "DQ-051": ("error:unit_of_measure", "Booked in packs of 10 (India 2021-H2)", "fixed"),
    "DQ-052": ("error:currency", "Price/revenue in local currency (Europe 2022-Q1, x0.89)", "fixed"),
    "DQ-055a": ("error:units_x10", "Units keyed x10 (revenue consistent with /10)", "fixed"),
    "DQ-055b": ("error:units_x100", "Units keyed x100 (revenue consistent with /100)", "fixed"),
    "DQ-056": ("error:revenue_x10", "Revenue keyed x10", "fixed"),
    "DQ-057": ("error:units_revenue_conflict", "Units disagreed with revenue; revenue/price kept", "fixed"),
    "DQ-011": ("error:date_typo", "Year typo (e.g. 2062 for 2026)", "fixed"),
    "DQ-022": ("error:region_missing", "Region missing, inferred", "fixed"),
    "DQ-044": ("error:units_missing", "Units missing and unrecoverable", "imputed, weight 0"),
    "DQ-060": ("error:row_missing", "Month missing from ERP extract (data loss)", "imputed, weight 0"),
    "DQ-061": ("error:row_missing", "Month missing; restored from stockout file", "fixed"),
}


def _event_lookup(d: pd.DataFrame, em: pd.DataFrame, verdicts: pd.DataFrame) -> pd.Series:
    """For each row: the id of a *real-effect* event whose scope covers it (or None)."""
    real = verdicts[verdicts["verdict"].str.startswith("real")]["Event_ID"]
    em = em[em["Event_ID"].isin(real) & (em["frac"] >= 0.2)]
    out = pd.Series(None, index=d.index, dtype=object)
    for _, e in em.iterrows():
        m = d["date"] == e["date"]
        if e["Scope_Type"] == "Region":
            m &= d["region"] == e["scope_region"]
        elif e["Scope_Type"] == "Product_Family":
            m &= d["family"] == e["Scope_Value"]
        elif e["Scope_Type"] == "Channel":
            m &= d["channel"] == e["Scope_Value"]
        elif e["Scope_Type"] == "Customer":
            m &= (d["customer"] == e["scope_customer"]) & (d["region"] == e["scope_region"])
        out[m & out.isna()] = e["Event_ID"]
    return out


def detect_outliers(d: pd.DataFrame, em: pd.DataFrame, verdicts: pd.DataFrame, z_thr: float = 4.0):
    d = d.copy()
    ok = (d["order_expected"] == 1) & (d["expected"] > 0) & d["shipped"].notna()
    d["log_resid"] = np.where(ok, np.log((d["demand"] + 1) / (d["expected"] + 1)), np.nan)
    med = d.groupby(["sku", "region"])["log_resid"].transform("median")
    mad = d.groupby(["sku", "region"])["log_resid"].transform(lambda s: (s - s.median()).abs().median())
    d["z"] = (d["log_resid"] - med) / (1.4826 * mad.clip(lower=0.05))
    d["event_id"] = _event_lookup(d, em, verdicts)
    g = d.groupby(["sku", "region"], sort=False)
    zp, zn = g["z"].shift(1), g["z"].shift(-1)

    rows = []
    hi = d["z"] > z_thr
    lo = d["z"] < -z_thr
    off = (d["order_expected"] == 0) & (d["shipped"] > 0)
    miss = (d["order_expected"] == 1) & (d["shipped"] == 0)

    def add(mask, typ, cls, reason, action):
        x = d[mask]
        rows.append(pd.DataFrame({
            "date": x["date"], "sku": x["sku"], "region": x["region"], "customer": x["customer"],
            "family": x["family"], "channel": x["channel"], "type": typ, "classification": cls,
            "raw_value": x["units_in"], "clean_value": x["shipped"], "demand_value": x["demand"],
            "expected_value": x["expected"].round(1), "score": x["z"].round(2), "linked_event_id": x["event_id"],
            "reason": reason, "action": action}))

    ev = d["event_id"].notna()
    add(hi & ev, "spike:event", "real_event", "High vs expected inside a real-effect event window",
        "model via event feature")
    add(lo & ev, "dip:event", "real_event", "Low vs expected inside a real-effect event window",
        "model via event feature")
    bulk = hi & ~ev & (zp.fillna(0) < 2) & (zn.fillna(0) < 2) & (d["demand"] >= 2.5 * d["expected"])
    add(bulk, "spike:one_off_order", "real_one_off",
        "Isolated spike, revenue consistent with units (not a keying error) - one-off bulk order",
        "exclude from training (weight 0), keep in history view")
    add(hi & ~ev & ~bulk, "spike:sustained", "real_level_shift", "High for consecutive months",
        "model (level shift)")
    add(lo & ~ev & ~d["stockout_listed"] & ~d["possible_stockout"], "dip:unexplained", "real_unexplained",
        "Low vs expected, no event / stockout", "down-weight")
    add(d["stockout_listed"], "stockout:listed", "real_supply",
        "Supply-constrained month listed in inventory_stockouts", "correct to Requested_Units")
    add(d["demand_source"] == "backorder_catchup_removed", "stockout:catch_up", "real_supply",
        "Backorder from previous month shipped now", "correct: subtract carried-in units")
    add(d["demand_source"] == "stockout_inferred:chip_shortage", "stockout:inferred_chip", "real_supply",
        "Electrical Assemblies chip allocation, month not listed (file incomplete)", "correct: estimated demand")
    add(d["possible_stockout"], "stockout:possible", "real_supply_uncertain",
        "Dip <50% expected then catch-up >140%: possible unlisted stockout", "down-weight (0.5)")
    add(off & ~ev, "cycle:off_cycle_order", "real_event", "Order in a month the cycle says no order",
        "keep; cycle feature")
    add(miss & ~ev & ~d["stockout_listed"], "cycle:missed_order", "real_event",
        "No order in a month the cycle expects one", "keep; cycle feature")
    reg = pd.concat(rows, ignore_index=True)

    # errors fixed in WS1
    err = []
    for rule, (typ, reason, action) in ERROR_RULES.items():
        x = d[d["dq_rules"].str.contains(rule + ";", regex=False)]
        if x.empty:
            continue
        err.append(pd.DataFrame({
            "date": x["date"], "sku": x["sku"], "region": x["region"], "customer": x["customer"],
            "family": x["family"], "channel": x["channel"], "type": typ, "classification": "error",
            "raw_value": x["units_in"], "clean_value": x["shipped"], "demand_value": x["demand"],
            "expected_value": x["expected"].round(1), "score": np.nan, "linked_event_id": None,
            "reason": reason + f" [{rule}]", "action": action}))
    reg = pd.concat([pd.concat(err, ignore_index=True), reg], ignore_index=True)
    reg = reg.sort_values(["date", "sku", "region"]).reset_index(drop=True)
    reg.insert(0, "anomaly_id", [f"AN-{i:06d}" for i in range(1, len(reg) + 1)])
    d.loc[bulk, "train_weight"] = 0.0
    d["is_one_off"] = bulk
    return reg, d
