"""Final forecast stage: train on all history (origin 2026-10), predict test.csv, add P10/P90,
lifecycle rules, forecast-window events, revenue, explanations and what-if scenarios.

Model choice (objective, features, ensemble weights) comes from the backtest: see
outputs/model_selection.json written by celeris.model.select.
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from celeris import config as C
from celeris.model import models as M
from celeris.model.lifecycle import apply_lifecycle, fit_curve, handover_table
from celeris.model.panel import MACRO_COLS, MIDX, MONTHS, Fold, Panel
from celeris.model.quantiles import apply_quantiles, classes
from celeris.model.select import STEP

ORION = {"customer": "Orion Driveline", "region": "North America",
         "months": {"2027-07": None, "2027-08": None, "2027-09": None}}

SCENARIOS = {
    "PMI -3": {"ISM_Manufacturing_PMI": -3.0},
    "PMI +3": {"ISM_Manufacturing_PMI": +3.0},
    "Vehicle inventory +10 days": {"Vehicle_Inventory_Days": +10.0},
    "Capacity utilisation -3 pts": {"Industrial_Util_Rate_Pct": -3.0},
    "EV adoption +10": {"EV_Adoption_Index": +10.0},
}


def _predict_units(F: Fold, models: list, feats_list: list, weights: list, pred: pd.DataFrame,
                   struct_w: float) -> np.ndarray:
    """Weighted ensemble of GBMs (+ structural) in units, for the rows of `pred`."""
    scale = pred["scale"].to_numpy()
    cold = np.isnan(scale)
    fb = M.fallback_scale(F)[pred["s"].to_numpy()]
    sc = np.where(cold, fb, scale)
    y = np.zeros(len(pred))
    for m, feats, w in zip(models, feats_list, weights):
        yh = np.clip(m.predict(pred[feats]), 0, None)
        y += w * yh
    y = y / sum(weights)
    y[cold] = pred.loc[cold, "si_t"].to_numpy()
    units_gbm = y * sc * pred["k_t"].to_numpy() * pred["oe_t"].to_numpy()
    st = M.structural(F)[pred["s"].to_numpy(), pred["t"].to_numpy()]
    st = np.where(np.isnan(st), units_gbm, st)
    return (1 - struct_w) * units_gbm + struct_w * st


def event_adjustments(fc: pd.DataFrame, effects: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Forecast-window events that features cannot learn (never seen in training).
    Orion Driveline NA planned retooling shutdown 2027-07..08: analogue = Nexon strike
    (customer-level plant stoppage): measured effect -77 % in event months, +16 % rebound after."""
    nx = effects.set_index("Event_ID").loc["EV-038"]
    dip = 1 + float(nx["effect_pct"]) / 100
    reb = 1 + float(nx["next_month_effect_pct"]) / 100
    rows = (fc["customer"] == ORION["customer"]) & (fc["region"] == ORION["region"])
    log = []
    for mth, f in [("2027-07", dip), ("2027-08", dip), ("2027-09", reb)]:
        r = rows & (fc["date"] == mth)
        for q in ("p10", "p50", "p90"):
            fc.loc[r, q] = fc.loc[r, q] * f
        fc.loc[r, "adjustment"] = fc.loc[r, "adjustment"] + f"orion_shutdown x{f:.2f};"
        log.append(dict(event="EV-062 Orion NA retooling", month=mth, factor=round(f, 3), rows=int(r.sum())))
    return fc, log


def run_forecast(run_id: str):
    sel = json.loads((C.OUT / "model_selection.json").read_text())
    P = Panel()
    F = Fold(P, "2026-10")
    models, feats_list, cache = [], [], {}
    for spec in sel["ensemble"]:
        st = spec.get("start", sel["train_start"])
        if st not in cache:
            cache[st] = F.training_set(start=st, step=STEP)
        params = dict(spec["params"], seed=spec.get("seed", 0))
        saved = C.OUT / "models" / f"{spec['name']}.txt"
        if os.environ.get("CELERIS_REUSE_MODELS") == "1" and saved.exists():
            m = lgb.Booster(model_file=str(saved))          # skip retraining (dashboard iteration only)
            feats = m.feature_name()
        else:
            m, feats = M.fit_gbm(cache[st], params=params, rounds=spec.get("rounds", sel["rounds"]),
                                 drop_groups=tuple(spec["drop"]))
        models.append(m)
        feats_list.append(feats)
        print(f"  trained {spec['name']}", flush=True)
    weights = [s["weight"] for s in sel["ensemble"]]
    del cache

    pred = F.prediction_set()
    pred = pred[P.TEST_EXISTS[pred["s"].to_numpy(), pred["t"].to_numpy()]].reset_index(drop=True)
    p50 = _predict_units(F, models, feats_list, weights, pred, sel["structural_weight"])

    fc = pd.DataFrame({"s": pred["s"], "t": pred["t"], "h": pred["h"],
                       "date": [MONTHS[t] for t in pred["t"]],
                       "sku": P.static["sku"].to_numpy()[pred["s"]], "region": P.static["region"].to_numpy()[pred["s"]],
                       "family": P.static["family"].to_numpy()[pred["s"]],
                       "channel": P.static["channel"].to_numpy()[pred["s"]],
                       "customer": P.static["customer"].to_numpy()[pred["s"]],
                       "order_expected": pred["oe_t"].to_numpy(), "months_covered": pred["k_t"].to_numpy(),
                       "cold_start": pred["scale"].isna().to_numpy(), "p50": p50, "adjustment": ""})
    # intervals
    q = pd.read_csv(C.OUT / "quantile_table.csv")
    edges = pd.read_csv(C.OUT / "quantile_vol_edges.csv", index_col=0)
    cyc = pd.read_csv(C.OUT / "order_cycles.csv")
    fc = classes(fc, cyc)
    fc = apply_quantiles(fc, "p50", q, edges)
    cold = fc["cold_start"]
    fc.loc[cold, "p10"] = fc.loc[cold, "p50"] * 0.5
    fc.loc[cold, "p90"] = fc.loc[cold, "p50"] * 1.6

    # lifecycle + lost accounts
    d = pd.read_parquet(C.CLEAN / "demand.parquet")
    master = pd.read_parquet(C.CLEAN / "sku_master.parquet")
    curve = fit_curve(handover_table(d, master))
    curve.to_csv(C.OUT / "handover_curve.csv")
    lost = cyc[cyc["current_cycle"] == "no orders"]
    fc, audit = apply_lifecycle(fc, d, master, curve, lost)
    # events in the forecast window
    eff = pd.read_csv(C.OUT / "event_effects.csv")
    fc, evlog = event_adjustments(fc, eff)

    for qq in ("p10", "p50", "p90"):
        fc[qq] = fc[qq].clip(lower=0)
    fc["p10"] = np.minimum(fc["p10"], fc["p50"])
    fc["p90"] = np.maximum(fc["p90"], fc["p50"])

    # prices: last 2026 price x announced Jan increases (+4.5 % 2027, +4.0 % 2028)
    prices = pd.read_parquet(C.CLEAN / "price_table.parquet")
    p26 = prices[prices["pperiod"] == "2026"].set_index("sku")["price_usd"]
    last = prices.sort_values("pperiod").groupby("sku")["price_usd"].last()
    base = fc["sku"].map(p26).fillna(fc["sku"].map(last))
    fam_med = base.groupby(fc["family"]).transform("median")
    base = base.fillna(fam_med)                                    # launches without a price
    growth = np.where(fc["date"] >= "2028-01", 1.045 * 1.04, 1.045)
    fc["price"] = base * growth
    fc["revenue_p50"] = fc["p50"] * fc["price"]
    fc["revenue_p10"] = fc["p10"] * fc["price"]
    fc["revenue_p90"] = fc["p90"] * fc["price"]
    bu = master.groupby("Product_Family")["Business_Unit"].agg(lambda s: s.mode().iloc[0])
    fc["bu"] = fc["family"].map(bu)

    # submission
    test = pd.read_csv(C.RAW / "test.csv")
    key = fc.set_index(["date", "sku", "region"])["p50"]
    sub = test[["Date", "SKU_ID", "Region"]].copy()
    v = key.reindex(pd.MultiIndex.from_frame(sub.rename(columns={"Date": "date", "SKU_ID": "sku",
                                                                   "Region": "region"}))).to_numpy()
    sub["Units_Sold_Forecast"] = np.rint(np.nan_to_num(v, nan=0.0)).clip(0).astype(int)
    assert len(sub) == 58219 and not np.isnan(v).any(), "submission rows missing a forecast"
    sub.to_csv(C.OUT / "submission.csv", index=False)

    cols = ["date", "h", "sku", "region", "family", "bu", "channel", "customer", "order_expected",
            "months_covered", "cold_start", "p10", "p50", "p90", "price", "revenue_p10", "revenue_p50",
            "revenue_p90", "adjustment", "hb", "cyc"]
    fc[cols].to_parquet(C.OUT / "forecast.parquet", index=False)
    audit.to_csv(C.OUT / "lifecycle_audit.csv", index=False)
    pd.DataFrame(evlog).to_csv(C.OUT / "event_adjustments.csv", index=False)

    # explanations: gain share per feature, averaged over the ensemble members
    imps = []
    for m, feats in zip(models, feats_list):
        g = pd.Series(m.feature_importance("gain"), index=feats)
        imps.append(g / g.sum())
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    imp = imp.rename("gain_share").rename_axis("feature").reset_index()
    imp.to_csv(C.OUT / "feature_importance.csv", index=False)
    # models + prediction features are saved so the dashboard can explain any series on demand
    mdir = C.OUT / "models"
    mdir.mkdir(exist_ok=True)
    for spec, m, feats in zip(sel["ensemble"], models, feats_list):
        m.save_model(str(mdir / f"{spec['name']}.txt"))
    keep = sorted(set(sum(feats_list, [])))
    px = pred[keep].copy()
    px.insert(0, "date", fc["date"].values)
    px.insert(1, "sku", fc["sku"].values)
    px["region"] = fc["region"].values                     # region is also a model feature
    px.to_parquet(C.OUT / "forecast_features.parquet", index=False)

    # what-if scenarios: shift the macro scenario path for the forecast window and re-predict
    rows = []
    base_tot = fc.groupby(["date", "family"])["p50"].sum()
    macro_orig = P.macro.copy()
    for name, shift in SCENARIOS.items():
        P.macro = macro_orig.copy()
        for c, v in shift.items():
            j = MACRO_COLS.index(c)
            P.macro[F.O + 1:, j] += v
        ps = F.prediction_set()
        ps = ps[P.TEST_EXISTS[ps["s"].to_numpy(), ps["t"].to_numpy()]].reset_index(drop=True)
        u = _predict_units(F, models, feats_list, weights, ps, sel["structural_weight"])
        ratio = np.divide(u, p50, out=np.ones_like(u), where=p50 > 0)
        tmp = fc[["date", "family"]].assign(p50=fc["p50"] * ratio)
        s = tmp.groupby(["date", "family"])["p50"].sum()
        rows.append(pd.DataFrame({"scenario": name, "p50": s, "base_p50": base_tot}).reset_index())
    P.macro = macro_orig
    pd.concat(rows).to_parquet(C.OUT / "scenarios.parquet", index=False)
    print(f"  forecast: {len(sub):,} rows, total p50 {fc['p50'].sum():,.0f} units, "
          f"revenue ${fc['revenue_p50'].sum() / 1e9:.2f}B")
