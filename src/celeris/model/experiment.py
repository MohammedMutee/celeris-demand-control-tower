"""Experiment runner: run GBM configs over backtest folds, append results to outputs/experiments.csv.

usage: python -m celeris.model.experiment <name> [origins]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from celeris import config as C
from celeris.model import models as M
from celeris.model.backtest import long_frame
from celeris.model.panel import Fold, Panel

CONFIGS = {
    "l1": dict(params={}, drop=(), start="2020-06"),
    "l2": dict(params={"objective": "l2"}, drop=(), start="2020-06"),
    "huber": dict(params={"objective": "huber", "alpha": 1.0}, drop=(), start="2020-06"),
    "tweedie": dict(params={"objective": "tweedie", "tweedie_variance_power": 1.3}, drop=(), start="2020-06"),
    "l1_nomacro": dict(params={}, drop=("mx_", "md_"), start="2020-06"),
    "l2_nomacro": dict(params={"objective": "l2"}, drop=("mx_", "md_"), start="2020-06"),
    "l1_from2021": dict(params={}, drop=(), start="2021-01"),
    "l2_from2021": dict(params={"objective": "l2"}, drop=(), start="2021-01"),
}
FAST = dict(learning_rate=0.05)
ROUNDS = 700
SEL_ORIGINS = ["2023-09", "2024-09", "2025-03"]


def run(names, origins=SEL_ORIGINS, save_preds=True):
    P = Panel()
    out_rows = []
    for origin in origins:
        F = Fold(P, origin)
        fb = M.structural(F)
        fbs = M.fallback_scale(F)
        fbm = np.where(np.isnan(fb), fbs[:, None] * F.SI * F.K * F.OE, fb)
        Xcache = {}
        preds = {"structural": fbm, "snaive": M.seasonal_naive(F)}
        pred_set = F.prediction_set()
        for name in names:
            cfg = CONFIGS[name]
            if cfg["start"] not in Xcache:
                Xcache[cfg["start"]] = F.training_set(start=cfg["start"], step=2)
            X = Xcache[cfg["start"]]
            t = time.time()
            m, feats = M.fit_gbm(X, params=dict(FAST, **cfg["params"]), rounds=ROUNDS, drop_groups=cfg["drop"])
            preds[name] = M.gbm_forecast(F, m, feats, pred_set)
            print(f"  {origin} {name}: {time.time() - t:.0f}s", flush=True)
        df = long_frame(P, F.O, preds, fbm)
        a = df["actual"].sum()
        for k in preds:
            out_rows.append(dict(origin=origin, model=k, wape=(df[k] - df["actual"]).abs().sum() / a,
                                 bias=(df[k].sum() - a) / a, n=len(df)))
        if save_preds:
            df.to_parquet(C.DATA / f"bt_{origin}_{'_'.join(names)[:40]}.parquet")
    res = pd.DataFrame(out_rows)
    path = C.OUT / "experiments.csv"
    res["config"] = res["model"].map(lambda k: json.dumps(CONFIGS.get(k, {}), default=str))
    res["run_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    res.to_csv(path, mode="a", header=not path.exists(), index=False)
    summ = res.groupby("model")[["wape", "bias"]].mean().sort_values("wape")
    print(summ.round(4).to_string())
    return res


if __name__ == "__main__":
    names = sys.argv[1].split(",")
    origins = sys.argv[2].split(",") if len(sys.argv) > 2 else SEL_ORIGINS
    run(names, origins)
