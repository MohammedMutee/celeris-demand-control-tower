"""P10 / P90 from calibrated backtest residuals (WS4 Expert).

ratio = actual / p50 on out-of-sample backtest rows, grouped by
  horizon bucket x cycle class (monthly / periodic) x volume tercile.
p10 = p50 x q10(ratio), p90 = p50 x q90(ratio). Coverage is checked on a held-out fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HB = [0, 3, 6, 12, 19]


def classes(df: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hb"] = pd.cut(df["h"], HB, labels=["1-3", "4-6", "7-12", "13-19"]).astype(str)
    per = cycles.set_index(["sku", "region"])["current_period"]
    p = per.reindex(pd.MultiIndex.from_frame(df[["sku", "region"]])).fillna(1).to_numpy()
    df["cyc"] = np.where(p == 1, "monthly", "periodic")
    return df


def fit_quantiles(bt: pd.DataFrame, pred_col: str, qs=(0.1, 0.9)) -> pd.DataFrame:
    x = bt[bt[pred_col] > 0.5].copy()
    x["ratio"] = x["actual"] / x[pred_col]
    x["vol"] = pd.qcut(x[pred_col].rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    edges = x.groupby("vol")[pred_col].agg(["min", "max"])
    q = x.groupby(["hb", "cyc", "vol"])["ratio"].quantile(list(qs)).unstack()
    q.columns = [f"q{int(c * 100)}" for c in q.columns]
    q.attrs["vol_edges"] = edges
    return q.reset_index(), edges


def vol_class(pred: pd.Series, edges: pd.DataFrame) -> pd.Series:
    lo = edges.loc["low", "max"]
    hi = edges.loc["mid", "max"]
    return pd.Series(np.where(pred <= lo, "low", np.where(pred <= hi, "mid", "high")), index=pred.index)


def apply_quantiles(df: pd.DataFrame, pred_col: str, q: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol"] = vol_class(df[pred_col], edges)
    df = df.merge(q, on=["hb", "cyc", "vol"], how="left")
    df["q10"] = df["q10"].fillna(df["q10"].median())
    df["q90"] = df["q90"].fillna(df["q90"].median())
    df["p10"] = df[pred_col] * df["q10"]
    df["p90"] = df[pred_col] * df["q90"]
    return df


def coverage(df: pd.DataFrame) -> dict:
    inside = (df["actual"] >= df["p10"]) & (df["actual"] <= df["p90"])
    pin = lambda q, p: np.mean(np.maximum(q * (df["actual"] - p), (q - 1) * (df["actual"] - p)))
    return {"coverage_p10_p90": float(inside.mean()), "below_p10": float((df["actual"] < df["p10"]).mean()),
            "above_p90": float((df["actual"] > df["p90"]).mean()),
            "pinball_p10": float(pin(0.1, df["p10"])), "pinball_p90": float(pin(0.9, df["p90"]))}
