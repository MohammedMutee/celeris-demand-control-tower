"""Account changes, lost accounts, structural breaks and the EV shift (WS2/WS3 Expert)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt


def customer_changes(d: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    """Customer switches within a series and the monthly-rate change around them."""
    o = d[d["row_status"] == "observed"].sort_values(["sku", "region", "date"])
    prev = o.groupby(["sku", "region"])["customer"].shift()
    sw = o[prev.notna() & (prev != o["customer"])].assign(prev_customer=prev)
    rows = []
    for _, r in sw.iterrows():
        s = d[(d["sku"] == r.sku) & (d["region"] == r.region)].set_index("date")["demand"]
        before = s[s.index < r.date].tail(12)
        after = s[s.index >= r.date].head(12)
        rows.append(dict(type="customer_switch", sku=r.sku, region=r.region, date=r.date,
                         from_customer=r.prev_customer, to_customer=r.customer, channel=r.channel,
                         rate_before=round(before.mean(), 1), rate_after=round(after.mean(), 1),
                         level_change_pct=round(100 * (after.mean() / before.mean() - 1), 1) if before.mean() else None,
                         action="model: series continues; level from post-switch months"))
    # lost accounts: current ordering cycle = no orders (rows keep arriving with 0 units)
    lost = cycles[cycles["current_cycle"] == "no orders"]
    for _, r in lost.iterrows():
        s = d[(d["sku"] == r.sku) & (d["region"] == r.region)]
        before = s[s["date"] < r.current_since]["demand"].tail(12)
        rows.append(dict(type="lost_account", sku=r.sku, region=r.region, date=r.current_since,
                         from_customer=s["customer"].iloc[-1], to_customer=None, channel=s["channel"].iloc[-1],
                         rate_before=round(before.mean(), 1), rate_after=0.0, level_change_pct=-100.0,
                         action="forecast 0 unless orders resume (rows still present in test)"))
    return pd.DataFrame(rows)


def structural_breaks(d: pd.DataFrame, penalty: float = 8.0) -> pd.DataFrame:
    """PELT on log demand (deseasonalised by 12-month differencing of the level) per family and
    region aggregate. Reports break dates with the level change across each break."""
    rows = []
    for dim in ["family", "region", "channel"]:
        for key, g in d.groupby(dim):
            s = g.groupby("date")["demand"].sum().sort_index()
            s = s[s > 0]
            if len(s) < 36:
                continue
            # remove calendar-month seasonality (ratio to the series' calendar-month median)
            cal = s.index.str[5:]
            seas = s.groupby(cal).transform("median") / s.median()
            y = np.log(s / seas).to_numpy()
            algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(y.reshape(-1, 1))
            bkps = algo.predict(pen=penalty * np.var(np.diff(y)) * 2)
            prev = 0
            for b in bkps[:-1]:
                lb = float(np.mean(y[max(prev, b - 6):b]))
                la = float(np.mean(y[b:min(len(y), b + 6)]))
                rows.append(dict(dimension=dim, key=key, break_date=s.index[b],
                                 level_change_pct=round(100 * (np.exp(la - lb) - 1), 1)))
                prev = b
    return pd.DataFrame(rows)


def ev_shift(d: pd.DataFrame, master: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Annual demand by powertrain application and family, with the EV index."""
    pt = master.set_index("SKU_ID")["Powertrain_Application"]
    x = d.assign(powertrain=d["sku"].map(pt), year=d["date"].str[:4])
    x = x[x["date"] <= "2026-10"]
    a = x.groupby(["year", "powertrain"])["demand"].sum().unstack()
    share = a.div(a.sum(axis=1), axis=0)
    ev = macro.assign(year=macro["date"].str[:4]).groupby("year")["EV_Adoption_Index"].mean()
    out = share.add_prefix("share_").join(a.add_prefix("units_")).join(ev)
    # per-SKU growth by powertrain (2023->2025 like-for-like, Active SKUs only)
    return out.reset_index()


def ev_elasticity(d: pd.DataFrame, master: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Log-log slope of monthly demand (per powertrain, deseasonalised YoY) on the EV index."""
    pt = master.set_index("SKU_ID")["Powertrain_Application"]
    x = d.assign(powertrain=d["sku"].map(pt))
    x = x[(x["date"] >= "2022-01") & (x["date"] <= "2026-10") & (x["family"] != "Electrical Assemblies")]
    m = macro.set_index("date")["EV_Adoption_Index"]
    rows = []
    tot = x.groupby("date")["demand"].sum()
    for p, g in x.groupby("powertrain"):
        s = g.groupby("date")["demand"].sum()
        share = np.log(s / tot)                     # share of total removes the common cycle
        e = np.log(m.reindex(share.index))
        b = np.polyfit(e, share, 1)[0]
        rows.append(dict(powertrain=p, share_elasticity_to_ev_index=round(float(b), 2),
                         share_2022=round(float(np.exp(share.iloc[:12]).mean()), 3),
                         share_2026=round(float(np.exp(share.iloc[-10:]).mean()), 3)))
    return pd.DataFrame(rows)
