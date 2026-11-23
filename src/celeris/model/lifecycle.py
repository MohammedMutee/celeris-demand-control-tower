"""Lifecycle handling (WS4 Expert): launches via Replaces_SKU, EOL run-down, lost accounts.

Handover curve learned from the 6 historical Replaces_SKU transitions:
  successor(r, m)   = curve_new[m]  x predecessor per-region base rate      (m = months since launch)
  predecessor(r, m) = curve_old[m]  x its own base, for launch <= m <= EOL, then 0
where base = predecessor's mean monthly demand per region 7-12 months before launch.
Validated leave-one-transition-out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAXM = 24


def handover_table(d: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    m = master.set_index("SKU_ID")
    rows = []
    for new, r in m[m["Replaces_SKU"].notna()].iterrows():
        old = r["Replaces_SKU"]
        L = pd.Period(r["Launch_Date"], "M")
        if str(L) > "2026-10":
            continue
        E = pd.Period(m.loc[old, "EOL_Date"], "M")
        o = d[d["sku"] == old]
        n = d[d["sku"] == new]
        base_months = [str(L - k) for k in range(7, 13)]
        base_r = o[o["date"].isin(base_months)].groupby("region")["demand"].mean()
        base = base_r.mean()                                   # per-region base
        for k in range(0, MAXM + 1):
            mth = str(L + k)
            if mth > "2026-10":
                break
            nn = n[n["date"] == mth]
            oo = o[o["date"] == mth]
            rows.append(dict(new=new, old=old, m=k, eol_offset=(E - L).n,
                             new_ratio=nn["demand"].mean() / base if len(nn) else 0.0,
                             old_ratio=oo["demand"].mean() / base if len(oo) else 0.0,
                             n_new_regions=n["region"].nunique(), n_old_regions=o["region"].nunique(),
                             base_per_region=base))
    return pd.DataFrame(rows)


def fit_curve(tab: pd.DataFrame, exclude: str | None = None) -> pd.DataFrame:
    t = tab[tab["new"] != exclude] if exclude else tab
    c = t.groupby("m").agg(new=("new_ratio", "median"), old=("old_ratio", "median")).reindex(range(MAXM + 1))
    c["new"] = c["new"].interpolate().ffill()
    # smooth the plateau: after month 8 use the median of months 8..end
    plateau = c.loc[8:, "new"].median()
    c.loc[9:, "new"] = plateau
    c["new"] = c["new"].cummax().clip(upper=plateau)
    return c


def loo_validation(tab: pd.DataFrame) -> pd.DataFrame:
    out = []
    for new in tab["new"].unique():
        c = fit_curve(tab, exclude=new)
        g = tab[(tab["new"] == new) & (tab["m"] <= 18)]
        pred = c.loc[g["m"], "new"].to_numpy()
        act = g["new_ratio"].to_numpy()
        out.append(dict(new=new, wape=np.abs(pred - act).sum() / act.sum(), bias=(pred.sum() - act.sum()) / act.sum()))
    return pd.DataFrame(out)


def apply_lifecycle(fc: pd.DataFrame, d: pd.DataFrame, master: pd.DataFrame, curve: pd.DataFrame,
                    lost: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fc: test rows with columns date, sku, region, p50 (and optionally p10/p90).
    Returns the adjusted forecast and an audit table of what was changed."""
    fc = fc.copy()
    m = master.set_index("SKU_ID")
    audit = []
    qcols = [c for c in ("p10", "p50", "p90") if c in fc]
    # 1) planned launches with a predecessor: handover curve
    for new, r in m[(m["Lifecycle_Status"] == "Planned Launch") & m["Replaces_SKU"].notna()].iterrows():
        old = r["Replaces_SKU"]
        L = pd.Period(r["Launch_Date"], "M")
        o = d[d["sku"] == old]
        base_months = [str(L - k) for k in range(7, 13)]
        # predecessor base per region = its own forecast/history 7-12 months before launch
        hist = o[o["date"].isin(base_months)].groupby("region")["demand"].mean()
        fut = fc[(fc["sku"] == old) & fc["date"].isin(base_months)].groupby("region")["p50"].mean()
        base_r = pd.concat([hist, fut]).groupby(level=0).mean()
        base = float(base_r.mean())
        rows = fc["sku"] == new
        k = fc.loc[rows, "date"].map(lambda x: (pd.Period(x, "M") - L).n)
        reg_share = (base_r / base).reindex(fc.loc[rows, "region"]).fillna(1.0).to_numpy()
        val = curve["new"].reindex(k.clip(0, MAXM)).to_numpy() * base * reg_share
        val = np.where(k.to_numpy() < 0, 0.0, val)
        for q in qcols:
            mult = 1.0 if q == "p50" else (0.5 if q == "p10" else 1.6)       # wide band: no history
            fc.loc[rows, q] = val * mult
        audit.append(dict(sku=new, action="launch_handover_curve", rows=int(rows.sum()), predecessor=old,
                          base_per_region=round(base, 1)))
    # 2) EOL: predecessor run-down after successor launch, zero after EOL (rule 5)
    for sku, r in m[m["EOL_Date"].notna()].iterrows():
        rows = fc["sku"] == sku
        if not rows.any():
            continue
        after = rows & (fc["date"] > r["EOL_Date"])
        for q in qcols:
            fc.loc[after, q] = 0.0
        succ = m[m["Replaces_SKU"] == sku]
        n_run = 0
        if len(succ):
            L = pd.Period(succ["Launch_Date"].iloc[0], "M")
            run = rows & (fc["date"] >= str(L)) & (fc["date"] <= r["EOL_Date"])
            k = fc.loc[run, "date"].map(lambda x: (pd.Period(x, "M") - L).n)
            f = curve["old"].reindex(k).fillna(0.5).to_numpy()
            for q in qcols:
                fc.loc[run, q] = fc.loc[run, q] * f
            n_run = int(run.sum())
        audit.append(dict(sku=sku, action="eol_zero_after", rows=int(after.sum()), rundown_rows=n_run))
    # 3) lost accounts: series whose current cycle is 'no orders'
    key = set(zip(lost["sku"], lost["region"]))
    rows = pd.Series([(s, r) in key for s, r in zip(fc["sku"], fc["region"])], index=fc.index)
    for q in qcols:
        fc.loc[rows, q] = 0.0
    audit.append(dict(sku="(16 series)", action="lost_account_zero", rows=int(rows.sum())))
    return fc, pd.DataFrame(audit)
