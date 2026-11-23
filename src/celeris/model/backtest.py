"""Rolling-origin backtest with a 19-month horizon (WS4 Advanced)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from celeris.model.panel import MIDX, MONTHS, T_HIST_END, Fold, Panel

FULL_ORIGINS = ["2023-03", "2023-09", "2024-03", "2024-09", "2025-03"]
PARTIAL_ORIGINS = ["2025-09", "2026-03"]


def eval_mask(P: Panel, O: int, H: int = 19) -> np.ndarray:
    """Rows that exist, are not data-loss imputations, and fall in (O, O+H] within history."""
    M = np.zeros((P.S, P.T), bool)
    t1 = min(O + H, T_HIST_END)
    imputed = (P.W == 0) & ~P.ONEOFF
    M[:, O + 1:t1 + 1] = P.EXISTS[:, O + 1:t1 + 1] & ~imputed[:, O + 1:t1 + 1]
    return M


def long_frame(P: Panel, O: int, preds: dict[str, np.ndarray], fallback: np.ndarray) -> pd.DataFrame:
    M = eval_mask(P, O)
    s, t = np.nonzero(M)
    df = pd.DataFrame({"s": s, "t": t, "h": t - O, "origin": MONTHS[O], "date": [MONTHS[i] for i in t],
                       "actual": P.D[s, t]})
    for c in ["family", "region", "channel", "customer", "sku"]:
        df[c] = P.static[c].to_numpy()[s]
    for name, F in preds.items():
        v = F[s, t]
        df[name] = np.where(np.isnan(v), fallback[s, t], v)
        df[name] = np.clip(df[name], 0, None)
    return df


def metrics(df: pd.DataFrame, models: list[str], by: list[str] | None = None) -> pd.DataFrame:
    def f(g):
        a = g["actual"].sum()
        return pd.Series({**{f"{m}|wape": (g[m] - g["actual"]).abs().sum() / a for m in models},
                          **{f"{m}|bias": (g[m].sum() - a) / a for m in models}, "actual": a, "n": len(g)})
    if by is None:
        return f(df).to_frame().T
    return df.groupby(by).apply(f, include_groups=False)


def tidy(m: pd.DataFrame) -> pd.DataFrame:
    """wide 'model|metric' columns -> long (model, metric) table."""
    idx = [c for c in m.columns if "|" not in c]
    long = m.reset_index().melt(id_vars=[c for c in m.reset_index().columns if "|" not in c],
                                var_name="mm", value_name="value")
    long[["model", "metric"]] = long["mm"].str.split("|", expand=True)
    return long.drop(columns="mm")
