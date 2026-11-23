"""Final model selection on the full rolling-origin backtest (7 folds) and calibration.

1. Re-run finalist GBM configs at full strength on all folds (5 full 19-month + 2 partial).
2. Learn ensemble weights (GBM variants + structural) by grid search on mean fold WAPE.
3. Fit P10/P90 ratio quantiles on folds 1-4, check coverage on folds 5-7, then refit on all.
4. Report accuracy by family / region / horizon and at every hierarchy level (bottom-up sums).
Writes: backtest_long.parquet, backtest_metrics.csv, backtest_hierarchy.csv, quantile_table.csv,
        quantile_coverage.json, model_selection.json
"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from celeris import config as C
from celeris.model import models as M
from celeris.model.backtest import FULL_ORIGINS, PARTIAL_ORIGINS, long_frame
from celeris.model.panel import Fold, Panel
from celeris.model.quantiles import apply_quantiles, classes, coverage, fit_quantiles

ROUNDS = 1200
STEP = 2          # every 2nd origin month (same as the deployed model)


def run_folds(finalists: list[dict], train_start: str, origins=FULL_ORIGINS + PARTIAL_ORIGINS):
    P = Panel()
    frames = []
    for origin in origins:
        t0 = time.time()
        F = Fold(P, origin)
        fb = M.structural(F)
        fbm = np.where(np.isnan(fb), M.fallback_scale(F)[:, None] * F.SI * F.K * F.OE, fb)
        preds = {"snaive": M.seasonal_naive(F), "ma12": M.moving_average(F),
                 "croston": M.croston_family(F, method="croston"), "sba": M.croston_family(F, method="sba"),
                 "tsb": M.croston_family(F, method="tsb"), "structural": fbm}
        ps = F.prediction_set()
        cache = {}
        for spec in finalists:
            st = spec.get("start", train_start)
            if st not in cache:
                cache[st] = F.training_set(start=st, step=STEP)
            m, feats = M.fit_gbm(cache[st], params=dict(spec["params"], seed=spec.get("seed", 0)),
                                 rounds=spec.get("rounds", ROUNDS), drop_groups=tuple(spec["drop"]))
            preds[spec["name"]] = M.gbm_forecast(F, m, feats, ps)
        del cache
        df = long_frame(P, F.O, preds, fbm)
        frames.append(df)
        print(f"  fold {origin}: {time.time() - t0:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)


def wape(df, col):
    return (df[col] - df["actual"]).abs().sum() / df["actual"].sum()


def fold_mean_wape(bt, col):
    return bt.groupby("origin").apply(lambda g: wape(g, col), include_groups=False).mean()


def learn_weights(bt: pd.DataFrame, gbm_cols: list[str]):
    """Grid over GBM mixture weights (step 0.25) and structural share (step 0.1)."""
    best = None
    grid = [w for w in itertools.product(*[np.arange(0, 1.01, 0.25)] * len(gbm_cols)) if abs(sum(w) - 1) < 1e-9]
    for w in grid:
        g = sum(wi * bt[c] for wi, c in zip(w, gbm_cols))
        for sw in np.arange(0, 0.51, 0.1):
            bt["_ens"] = (1 - sw) * g + sw * bt["structural"]
            s = fold_mean_wape(bt, "_ens")
            if best is None or s < best[0]:
                best = (s, w, sw)
    s, w, sw = best
    bt["ensemble"] = (1 - sw) * sum(wi * bt[c] for wi, c in zip(w, gbm_cols)) + sw * bt["structural"]
    return dict(zip(gbm_cols, map(float, w))), float(sw), float(s)


def run_selection(finalists: list[dict], train_start: str):
    bt = run_folds(finalists, train_start)
    gcols = [f["name"] for f in finalists]
    w, sw, s = learn_weights(bt, gcols)
    bt = bt.drop(columns="_ens")
    models = ["snaive", "ma12", "croston", "sba", "tsb", "structural"] + gcols + ["ensemble"]

    rows = []
    for o, g in bt.groupby("origin"):
        for m in models:
            rows.append(dict(origin=o, level="overall", key="all", model=m, wape=wape(g, m),
                             bias=(g[m].sum() - g["actual"].sum()) / g["actual"].sum(), actual=g["actual"].sum()))
    for dim in ["family", "region", "channel", "h"]:
        for (k), g in bt.groupby(dim):
            for m in models:
                rows.append(dict(origin="all", level=dim, key=str(k), model=m, wape=wape(g, m),
                                 bias=(g[m].sum() - g["actual"].sum()) / g["actual"].sum(),
                                 actual=g["actual"].sum()))
    met = pd.DataFrame(rows)
    met.to_csv(C.OUT / "backtest_metrics.csv", index=False)

    # hierarchy: bottom-up sums are coherent by construction; accuracy at each level
    bu = pd.read_parquet(C.CLEAN / "sku_master.parquet").groupby("Product_Family")["Business_Unit"].agg(
        lambda x: x.mode().iloc[0])
    bt["bu"] = bt["family"].map(bu)
    hrows = []
    for lvl, keys in [("total", []), ("business_unit", ["bu"]), ("family", ["family"]), ("region", ["region"]),
                      ("family_region", ["family", "region"]), ("sku", ["sku"]), ("sku_region", ["sku", "region"])]:
        agg = bt.groupby(["origin", "date"] + keys)[["actual"] + models].sum().reset_index()
        for m in models:
            e = agg.groupby("origin").apply(lambda g: wape(g, m), include_groups=False).mean()
            hrows.append(dict(level=lvl, model=m, wape=e))
    pd.DataFrame(hrows).to_csv(C.OUT / "backtest_hierarchy.csv", index=False)

    # quantiles: calibrate on first 4 folds, validate on the rest, refit on all
    cyc = pd.read_csv(C.OUT / "order_cycles.csv")
    bt = classes(bt, cyc)
    early = bt["origin"].isin(FULL_ORIGINS[:4])
    q, edges = fit_quantiles(bt[early], "ensemble")
    val = apply_quantiles(bt[~early], "ensemble", q, edges)
    cov = coverage(val)
    q_all, edges_all = fit_quantiles(bt, "ensemble")
    q_all.to_csv(C.OUT / "quantile_table.csv", index=False)
    edges_all.to_csv(C.OUT / "quantile_vol_edges.csv")
    full = apply_quantiles(bt, "ensemble", q_all, edges_all)
    full[["origin", "date", "h", "sku", "region", "family", "channel", "customer", "actual"] + models +
         ["p10", "p90"]].to_parquet(C.OUT / "backtest_long.parquet", index=False)
    (C.OUT / "quantile_coverage.json").write_text(json.dumps({"validated_on_folds": sorted(bt.loc[~early, "origin"].unique().tolist()),
                                                              **cov}, indent=2))
    sel = dict(train_start=train_start, rounds=ROUNDS, structural_weight=sw,
               ensemble=[dict(f, weight=w[f["name"]]) for f in finalists if w[f["name"]] > 0],
               backtest_mean_wape=s,
               ladder={m: float(fold_mean_wape(bt, m)) for m in models})
    (C.OUT / "model_selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel["ladder"], indent=1), "\nweights", w, "structural", sw, "\ncoverage", cov)
    return sel


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    run_selection(spec["finalists"], spec["train_start"])
