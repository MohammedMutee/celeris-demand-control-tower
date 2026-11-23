"""Experiment 1: model ladder on two folds."""
import sys, time, pandas as pd, numpy as np
from celeris.model.panel import Panel, Fold, MIDX
from celeris.model import models as M
from celeris.model.backtest import long_frame, metrics
P = Panel()
res = []
for origin in sys.argv[1].split(","):
    t0 = time.time(); F = Fold(P, origin)
    fb = M.structural(F)                                   # common fallback for missing values
    fbs = M.fallback_scale(F)
    fbm = np.where(np.isnan(fb), fbs[:, None] * F.SI * F.K * F.OE, fb)
    preds = {"snaive": M.seasonal_naive(F), "ma12": M.moving_average(F), "croston": M.croston_family(F, method="croston"),
             "sba": M.croston_family(F, method="sba"), "tsb": M.croston_family(F, method="tsb"), "structural": fbm}
    X = F.training_set(step=2); t1 = time.time()
    m, feats = M.fit_gbm(X); preds["gbm_l1"] = M.gbm_forecast(F, m, feats); t2 = time.time()
    df = long_frame(P, F.O, preds, fbm); res.append(df)
    print(origin, f"build {t1-t0:.0f}s train {t2-t1:.0f}s rows {len(X):,}", flush=True)
    print(metrics(df, list(preds)).round(3).T.to_string())
pd.concat(res).to_parquet("data/_exp01.parquet")
