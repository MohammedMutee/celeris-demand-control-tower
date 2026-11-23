"""Unit tests for cycle detection + invariants on Phase-2 outputs."""
import json

import numpy as np
import pandas as pd
import pytest

from celeris import config as C
from celeris.anomalies.cycles import describe, segment_series


def _run(pattern: str, n: int = 48):
    o = np.array([float(pattern[i % len(pattern)]) for i in range(n)])
    return segment_series(o, np.arange(n))


@pytest.mark.parametrize("pattern,period,name", [
    ("1", 1, "monthly"), ("10", 2, "every 2nd month"), ("100", 3, "quarterly"),
    ("110", 3, "2 of 3 months"), ("111111111110", 12, "monthly, 1 skip/yr"),
])
def test_cycle_patterns(pattern, period, name):
    segs = _run(pattern)
    assert len(segs) == 1 and segs[0]["period"] == period
    assert describe(segs[0]["period"], segs[0]["mask"]) == name


def test_annual_skip_not_split_by_short_tail():
    # 85 months with an annual skip, then 9 months with no skip month inside: must stay ONE segment
    o = np.array([0.0 if i % 12 == 11 else 1.0 for i in range(94)])
    assert len(segment_series(o, np.arange(94))) == 1


def test_cycle_change_detected():
    o = np.array([1.0] * 40 + [float(i % 3 == 0) for i in range(40)])
    segs = segment_series(o, np.arange(80))
    assert len(segs) == 2 and abs(segs[1]["start"] - 40) <= 2 and segs[1]["period"] == 3


def test_missing_months_are_unknown_not_zero():
    o = np.array([1.0] * 30)
    o[[5, 17, 22]] = np.nan
    segs = segment_series(o, np.arange(30))
    assert len(segs) == 1 and segs[0]["period"] == 1


out = pytest.mark.skipif(not (C.CLEAN / "demand.parquet").exists(), reason="run pipeline first")


@out
def test_demand_frame():
    d = pd.read_parquet(C.CLEAN / "demand.parquet")
    assert not d.duplicated(["date", "sku", "region"]).any()
    assert d["demand"].notna().all() and (d["demand"] >= 0).all()
    # listed stockouts carry the requested quantity
    st = pd.read_parquet(C.CLEAN / "stockouts_clean.parquet").set_index(["date", "sku", "region"])
    x = d[d["stockout_listed"]].set_index(["date", "sku", "region"])
    assert (x["demand"] == st.loc[x.index, "Requested_Units"]).all()
    # imputed data-loss rows never get training weight
    assert (d.loc[d["demand_source"].str.startswith("imputed"), "train_weight"] == 0).all()
    # inferred chip-shortage demand is never below what was shipped
    sup = d["demand_source"] == "stockout_inferred:chip_shortage"
    assert (d.loc[sup, "demand"] >= d.loc[sup, "shipped"]).all()
    # listed months: shipped <= requested + carried-in (shipments can include last month's backorder)
    st2 = st.loc[x.index]
    assert (x["shipped"].fillna(0) <= st2["Requested_Units"] + st2["Carried_In_Units"]).all()


@out
def test_register_has_reason_and_action():
    r = pd.read_csv(C.OUT / "anomaly_register.csv")
    assert r["anomaly_id"].is_unique
    assert r[["type", "classification", "reason", "action"]].notna().all().all()
    assert {"error", "real_event", "real_supply", "real_one_off"} <= set(r["classification"])


@out
def test_chip_estimator_validated():
    s = json.loads((C.OUT / "phase2_summary.json").read_text())
    assert s["validation"]["chip_estimator_wape"]["A_shipped_over_fillrate"] < 0.2
