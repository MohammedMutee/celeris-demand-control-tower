"""Which macro indicator leads which product family, at what lag, and is it stable? (WS3)

Demand side : YoY log growth of family unconstrained demand (MEA excluded: its ramp is not macro).
Indicator   : YoY change (Auto_Production_YoY_Pct is already YoY, used as is).
CCF         : corr(indicator(t - L), demand(t)) for L = 0..12 (positive L = indicator leads),
              computed on PRE-WHITENED series (AR(p) fitted to the indicator, applied to both) so that
              two smooth trending series do not produce spurious correlation.
Granger     : F-test p-value, indicator -> demand, at the best lag.
Stability   : best lag and correlation in rolling 36-month windows; and pre-/post-2022 split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import grangercausalitytests

INDICATORS = ["ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
              "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
              "Fuel_Price_Index", "EV_Adoption_Index"]
MAX_LAG = 12


def _family_growth(d: pd.DataFrame) -> pd.DataFrame:
    x = d[(d["region"] != "Middle East & Africa") & (d["date"] <= "2026-10")]
    f = x.groupby(["date", "family"])["demand"].sum().unstack()
    return np.log(f).diff(12)


def _indicator_change(m: pd.DataFrame) -> pd.DataFrame:
    x = m.set_index("date")[INDICATORS].copy()
    for c in INDICATORS:
        if c != "Auto_Production_YoY_Pct":
            x[c] = x[c].diff(12)
    return x


def _prewhiten(x: pd.Series, y: pd.Series, p: int = 3):
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 24:
        return x, y
    ar = AutoReg(x.values, lags=p, trend="c").fit()
    a = ar.params[1:]

    def filt(s):
        v = s.values.copy()
        out = v[p:] - sum(a[i] * v[p - 1 - i:len(v) - 1 - i] for i in range(p))
        return pd.Series(out, index=s.index[p:])
    return filt(x), filt(y)


def ccf_table(fam: pd.DataFrame, ind: pd.DataFrame, start: str, end: str, prewhiten=True) -> pd.DataFrame:
    rows = []
    for f in fam.columns:
        for c in INDICATORS:
            y = fam[f].loc[start:end]
            x = ind[c].reindex(fam.index)
            for L in range(MAX_LAG + 1):
                xl = x.shift(L).loc[start:end]
                xa, ya = _prewhiten(xl, y) if prewhiten else (xl, y)
                ok = xa.notna() & ya.notna()
                r = float(np.corrcoef(xa[ok], ya[ok])[0, 1]) if ok.sum() > 12 else np.nan
                rows.append(dict(family=f, indicator=c, lag=L, corr=r, n=int(ok.sum())))
    return pd.DataFrame(rows)


def best_lags(t: pd.DataFrame) -> pd.DataFrame:
    i = t.groupby(["family", "indicator"])["corr"].apply(lambda s: s.abs().idxmax())
    b = t.loc[i.values].copy()
    b["sig_threshold"] = 2 / np.sqrt(b["n"])                       # ~95 % white-noise band
    b["significant"] = b["corr"].abs() > b["sig_threshold"]
    return b.reset_index(drop=True)


def granger_p(fam: pd.DataFrame, ind: pd.DataFrame, family: str, indicator: str, lag: int,
              start: str, end: str) -> float:
    if lag == 0:
        return np.nan
    df = pd.concat([fam[family], ind[indicator].reindex(fam.index)], axis=1).loc[start:end].dropna()
    df = df.diff().dropna()
    try:
        res = grangercausalitytests(df.values, maxlag=[lag])
        return float(res[lag][0]["ssr_ftest"][1])
    except Exception:
        return np.nan


def rolling_stability(fam, ind, window: int = 36, step: int = 3) -> pd.DataFrame:
    idx = fam.dropna(how="all").index
    rows = []
    for k in range(0, len(idx) - window + 1, step):
        s, e = idx[k], idx[k + window - 1]
        b = best_lags(ccf_table(fam, ind, s, e))
        b["window_end"] = e
        rows.append(b[["family", "indicator", "lag", "corr", "significant", "window_end"]])
    return pd.concat(rows, ignore_index=True)


def run_leadlag(d: pd.DataFrame, macro: pd.DataFrame):
    fam, ind = _family_growth(d), _indicator_change(macro)
    full = ccf_table(fam, ind, "2020-01", "2026-10")
    best = best_lags(full)
    # post-COVID only (are relationships driven by the 2020 shock?)
    post = best_lags(ccf_table(fam, ind, "2022-01", "2026-10")).rename(
        columns={"lag": "lag_post2022", "corr": "corr_post2022", "significant": "sig_post2022"})
    best = best.merge(post[["family", "indicator", "lag_post2022", "corr_post2022", "sig_post2022"]],
                      on=["family", "indicator"])
    best["granger_p"] = [granger_p(fam, ind, r.family, r.indicator, int(r.lag), "2020-01", "2026-10")
                         for r in best.itertuples()]
    best["stable"] = (best["significant"] & best["sig_post2022"] &
                      (np.sign(best["corr"]) == np.sign(best["corr_post2022"])) &
                      ((best["lag"] - best["lag_post2022"]).abs() <= 2))
    roll = rolling_stability(fam, ind)
    return full, best, roll
