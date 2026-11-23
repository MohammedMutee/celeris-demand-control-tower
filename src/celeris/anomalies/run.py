"""Phase-2 stage: anomalies, ordering cycles, unconstrained demand, event effects."""
from __future__ import annotations

import json

import pandas as pd

from celeris import config as C
from celeris.anomalies.cycles import detect_cycles
from celeris.anomalies.demand import build_demand
from celeris.anomalies.detect import detect_outliers
from celeris.anomalies.events import event_effects, holiday_effects, prebuy_analysis
from celeris.anomalies.structure import customer_changes, ev_elasticity, ev_shift, structural_breaks

DEMAND_COLS = ["date", "Year", "Month", "family", "bu", "sku", "description", "customer", "channel", "region",
               "units_in", "shipped", "demand", "demand_source", "expected", "level", "si", "price", "revenue",
               "order_expected", "cycle_period", "cycle_seg", "k", "stockout_listed", "st_reason",
               "lost_units_est", "possible_stockout", "is_one_off", "event_id", "z", "train_weight",
               "row_status", "dq_rules"]


def run_anomalies(run_id: str):
    sales = pd.read_parquet(C.CLEAN / "sales_clean.parquet")
    test = pd.read_parquet(C.CLEAN / "test_clean.parquet")
    stock = pd.read_parquet(C.CLEAN / "stockouts_clean.parquet")
    events = pd.read_parquet(C.CLEAN / "events_clean.parquet")
    em = pd.read_parquet(C.CLEAN / "event_months.parquet")
    master = pd.read_parquet(C.CLEAN / "sku_master.parquet")
    macro = pd.read_parquet(C.CLEAN / "macro_clean.parquet")

    series, segs, flags, fut = detect_cycles(sales, test)
    log = []
    d, metrics, si = build_demand(sales, flags, segs, stock, log)
    eff = event_effects(d, events, em)
    hol = holiday_effects(d, em)
    pre = prebuy_analysis(d, events)
    reg, d = detect_outliers(d, em, eff)
    cust = customer_changes(d, series)
    brk = structural_breaks(d)
    evs = ev_shift(d, master, macro)
    eve = ev_elasticity(d, master, macro)

    d["revenue_demand"] = d["demand"] * d["price"]
    d[DEMAND_COLS + ["revenue_demand"]].to_parquet(C.CLEAN / "demand.parquet", index=False)
    fut.to_parquet(C.CLEAN / "test_cycles.parquet", index=False)
    si.to_parquet(C.CLEAN / "seasonal_index.parquet", index=False)
    series.to_csv(C.OUT / "order_cycles.csv", index=False)
    segs.to_csv(C.OUT / "order_cycle_segments.csv", index=False)
    reg.to_csv(C.OUT / "anomaly_register.csv", index=False)
    eff.drop(columns=["start", "end"]).to_csv(C.OUT / "event_effects.csv", index=False)
    hol.to_csv(C.OUT / "holiday_effects.csv", index=False)
    pre.to_csv(C.OUT / "prebuy_effects.csv", index=False)
    cust.to_csv(C.OUT / "customer_changes.csv", index=False)
    brk.to_csv(C.OUT / "structural_breaks.csv", index=False)
    evs.to_csv(C.OUT / "ev_shift.csv", index=False)
    eve.to_csv(C.OUT / "ev_elasticity.csv", index=False)

    summary = {
        "run_id": run_id,
        "unconstrained_demand_steps": [dict(zip(["id", "step", "rows", "method"], x)) for x in log],
        "validation": metrics,
        "totals": {"shipped_units": float(d["shipped"].sum()), "demand_units": float(d["demand"].sum()),
                   "lost_units_est": float(d["lost_units_est"].sum()),
                   "lost_revenue_est_usd": float((d["lost_units_est"].fillna(0) * d["price"]).sum())},
        "register_counts": reg.groupby(["classification", "type"]).size().rename("n").reset_index()
        .to_dict("records"),
        "cycles": series["current_cycle"].value_counts().to_dict(),
        "cycle_changes": int((segs["seg"] > 0).sum()),
    }
    (C.OUT / "phase2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  anomalies: register {len(reg):,} rows; demand {d['demand'].sum():,.0f} vs shipped "
          f"{d['shipped'].sum():,.0f}; {summary['cycle_changes']} cycle changes")
