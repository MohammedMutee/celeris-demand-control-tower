"""Candidate models. Every model returns a units forecast matrix F[s, t] for t in (O, O+H]."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from celeris.model.panel import CAT_COLS, H, Fold

FEATURES = None    # set lazily from the training frame


# ------------------------------------------------------------------ fallback for series w/o history
def fallback_scale(F: Fold) -> np.ndarray:
    """Cold-start monthly rate: median scale of the same SKU in other regions, else the
    family x region median, else the family median."""
    P, O = F.P, F.O
    scale = np.nanmean(F.DR[:, max(0, O - 11):O + 1], 1)
    st = P.static.assign(scale=scale)
    sku_med = st.groupby("sku")["scale"].transform("median")
    fr_med = st.groupby(["family", "region"])["scale"].transform("median")
    f_med = st.groupby("family")["scale"].transform("median")
    return sku_med.fillna(fr_med).fillna(f_med).to_numpy()


def _future(F: Fold):
    P, O = F.P, F.O
    t1 = min(O + H, P.T - 1)
    return O, t1


# ------------------------------------------------------------------ baselines
def seasonal_naive(F: Fold) -> np.ndarray:
    P, O = F.P, F.O
    out = np.full((P.S, P.T), np.nan)
    for t in range(O + 1, min(O + H, P.T - 1) + 1):
        h = t - O
        out[:, t] = P.D[:, t - 12 * int(np.ceil(h / 12))]
    return out


def moving_average(F: Fold, n=12) -> np.ndarray:
    P, O = F.P, F.O
    m = np.nanmean(P.D[:, O - n + 1:O + 1], 1)
    out = np.full((P.S, P.T), np.nan)
    out[:, O + 1:min(O + H, P.T - 1) + 1] = m[:, None]
    return out


def croston_family(F: Fold, alpha=0.1, method="sba") -> np.ndarray:
    """Croston / SBA / TSB on the raw demand history (missing months skipped)."""
    P, O = F.P, F.O
    D = P.D[:, :O + 1]
    S = P.S
    z = np.full(S, np.nan)      # size
    p = np.full(S, np.nan)      # interval (croston) or probability (tsb)
    q = np.ones(S)              # periods since last demand
    for t in range(O + 1):
        x = D[:, t]
        obs = ~np.isnan(x)
        pos = obs & (x > 0)
        if method == "tsb":
            init = obs & np.isnan(p)
            p[init] = (x[init] > 0).astype(float)
            z[init & pos] = x[init & pos]
            upd = obs & ~init
            p[upd] = p[upd] + alpha * ((x[upd] > 0) - p[upd])
            zp = upd & pos
            z[zp] = np.where(np.isnan(z[zp]), x[zp], z[zp] + alpha * (x[zp] - z[zp]))
        else:
            first = pos & np.isnan(z)
            z[first] = x[first]
            p[first] = q[first]
            upd = pos & ~first
            z[upd] = z[upd] + alpha * (x[upd] - z[upd])
            p[upd] = p[upd] + alpha * (q[upd] - p[upd])
            q[pos] = 1
            q[obs & ~pos] += 1
    if method == "tsb":
        f = p * z
    else:
        f = z / p
        if method == "sba":
            f = f * (1 - alpha / 2)
    out = np.full((P.S, P.T), np.nan)
    out[:, O + 1:min(O + H, P.T - 1) + 1] = f[:, None]
    return out


def structural(F: Fold, n=6) -> np.ndarray:
    """level (deseasonalised rate, last n months) x seasonal index x cycle."""
    P, O = F.P, F.O
    lvl = np.nanmean(F.DR[:, O - n + 1:O + 1], 1)
    lvl = np.where(np.isnan(lvl), np.nanmean(F.DR[:, O - 11:O + 1], 1), lvl)
    out = np.full((P.S, P.T), np.nan)
    for t in range(O + 1, min(O + H, P.T - 1) + 1):
        out[:, t] = lvl * F.SI[:, t] * F.K[:, t] * F.OE[:, t]
    return out


# ------------------------------------------------------------------ global LightGBM
DROP = {"y", "demand", "w_row", "weight", "o", "t", "s", "scale", "oe_t"}

PARAMS = dict(objective="l1", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, verbose=-1, num_threads=8, seed=0)


def feature_list(X: pd.DataFrame, drop_groups=()) -> list[str]:
    cols = [c for c in X.columns if c not in DROP]
    for g in drop_groups:
        cols = [c for c in cols if not c.startswith(g)]
    return cols


def fit_gbm(X: pd.DataFrame, params=None, rounds=1200, drop_groups=()):
    p = dict(PARAMS, **(params or {}))
    feats = feature_list(X, drop_groups)
    ds = lgb.Dataset(X[feats], X["y"], weight=X["weight"],
                     categorical_feature=[c for c in CAT_COLS if c in feats], free_raw_data=True)
    m = lgb.train(p, ds, num_boost_round=rounds)
    return m, feats


def gbm_forecast(F: Fold, model, feats, pred: pd.DataFrame | None = None) -> np.ndarray:
    P = F.P
    pred = F.prediction_set() if pred is None else pred
    scale = pred["scale"].to_numpy()
    fb = fallback_scale(F)[pred["s"].to_numpy()]
    yhat = np.clip(model.predict(pred[feats]), 0, None)
    cold = np.isnan(scale)
    yhat[cold] = pred.loc[cold, "si_t"].to_numpy()        # no history: seasonal shape only
    sc = np.where(cold, fb, scale)
    units = yhat * sc * pred["k_t"].to_numpy() * pred["oe_t"].to_numpy()
    out = np.full((P.S, P.T), np.nan)
    out[pred["s"].to_numpy(), pred["t"].to_numpy()] = units
    return out
