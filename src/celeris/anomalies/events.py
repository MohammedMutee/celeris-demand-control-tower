"""Measure the demand effect of every calendar event (WS2 Advanced).

Method (difference-in-differences on ratio-to-trend):
  R_X(t)  = demand_X(t) / centred 13-month moving average of demand_X
  effect  = R_scope(t) / R_control(t) / typical_ratio(calendar month) - 1
For events that recur on the same calendar month every year (NA July retooling, EU August
shutdown) the typical ratio is NOT removed (it would cancel the effect); for everything else
the typical scope/control ratio of that calendar month in non-event years is removed.
Significance: z-score of the effect against the same statistic in all non-event months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOVING = {"Lunar New Year holiday period", "Ramadan"}
FIXED_ANNUAL = {"North American summer retooling weeks", "European summer plant shutdowns"}


def _ratio_to_trend(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    ma = s.rolling(13, center=True, min_periods=7).mean()
    return s / ma


def _scope_masks(d: pd.DataFrame, ev: pd.Series):
    st, sv = ev["Scope_Type"], ev["Scope_Value"]
    if st == "Global":
        return pd.Series(True, index=d.index), None
    if st == "Region":
        sc = d["region"] == ev["scope_region"]
        return sc, ~sc
    if st == "Product_Family":
        sc = d["family"] == sv
        return sc, ~sc
    if st == "Channel":
        sc = d["channel"] == sv
        return sc, ~sc
    if st == "Customer":
        same_reg = d["region"] == ev["scope_region"]
        sc = (d["customer"] == ev["scope_customer"]) & same_reg
        return sc, same_reg & ~sc
    raise ValueError(st)


def _method(ev) -> tuple[str, str]:
    """(method, measured column) per event type."""
    col = "shipped" if ev["Event_Type"] in ("Supply Constraint", "Logistics") else "demand"
    if ev["Event_Type"] == "System":
        return "data_loss", "shipped"
    if ev["Event_Type"] == "Disruption":
        return "yoy", col
    if ev["Event_Type"] == "Supply Constraint":
        return "did_level", col
    if bool(ev.get("open_ended")) or ev["Event_Type"] == "Market Entry":
        return "step", col
    return "did_trend", col


def _yoy_effect(A: pd.Series, months: list) -> tuple[float, list]:
    """Event months vs the same months a year earlier, relative to the YoY growth of the
    6 months before the event."""
    yoy = (A / A.shift(12)).dropna()
    pre_end = str(pd.Period(months[0], freq="M") - 1)
    pre = yoy[yoy.index <= pre_end].tail(6)
    base = float(pre.median())
    em = (yoy.reindex(months) / base - 1).dropna()
    return float(em.mean()), em


def _step_effect(d, sc, ctl, start: str, col: str):
    """Permanent level change: scope/control ratio, 12 months after vs 12 months before."""
    s = d[sc].groupby("date")[col].sum()
    c = d[ctl].groupby("date")[col].sum() if ctl is not None else None
    r = (s / c) if c is not None else s
    p0 = pd.Period(start, freq="M")
    before = r[(r.index < start) & (r.index >= str(p0 - 12))]
    after = r[(r.index >= start) & (r.index < str(p0 + 12))]
    if before.empty or after.empty:
        return np.nan, np.nan
    eff = after.mean() / before.mean() - 1
    # noise: same statistic at all other split points
    sims = []
    for k in range(12, len(r) - 12):
        b, a = r.iloc[k - 12:k], r.iloc[k:k + 12]
        sims.append(a.mean() / b.mean() - 1)
    sd = np.std(sims) if sims else np.nan
    return float(eff), float(eff / sd) if sd else np.nan


def event_effects(d: pd.DataFrame, events: pd.DataFrame, event_months: pd.DataFrame) -> pd.DataFrame:
    """One row per event: mean effect over its months, z-score, verdict."""
    d = d[d["date"] <= "2026-10"]
    out = []
    for _, ev in events.iterrows():
        months = sorted(event_months.loc[(event_months["Event_ID"] == ev["Event_ID"]) &
                                         (event_months["frac"] >= 0.2), "date"])
        months = [m for m in months if m <= "2026-10"]
        if not months:
            out.append(dict(Event_ID=ev["Event_ID"], in_history=False))
            continue
        sc, ctl = _scope_masks(d, ev)
        method, col = _method(ev)
        if method == "data_loss":
            lost = int((sc & d["date"].isin(months) & d["shipped"].isna()).sum())
            per_month = (d[sc & ~d["date"].isin(months)].groupby("date")["shipped"]
                         .apply(lambda s: s.isna().sum()))
            base = float(per_month.mean()) * len(months)
            z = (lost - base) / max(float(per_month.std()) * np.sqrt(len(months)), 1e-9)
            out.append(dict(Event_ID=ev["Event_ID"], in_history=True, n_months=len(months), method=method,
                            measured_on=col, rows_lost=lost, rows_lost_baseline=round(base, 1),
                            effect_pct=np.nan, z=round(z, 1)))
            continue
        if method == "step":
            start = months[0] if ev["Event_Type"] != "Market Entry" else months[0]
            if ev["Event_Type"] == "Market Entry":
                A = d[sc].groupby("date")[col].sum()
                first12 = A[(A.index >= months[0])].head(12)
                out.append(dict(Event_ID=ev["Event_ID"], in_history=True, n_months=len(months), method="ramp",
                                measured_on=col, effect_pct=np.nan, ramp_multiple_12m=round(first12.iloc[-1] / first12.iloc[0], 1),
                                z=None, monthly="ramp first->12th month: " +
                                ";".join(f"{k}:{v:.0f}" for k, v in first12.items())))
                continue
            eff, z = _step_effect(d, sc, ctl, start, col)
            out.append(dict(Event_ID=ev["Event_ID"], in_history=True, n_months=len(months), method=method,
                            measured_on=col, effect_pct=round(100 * eff, 1), z=round(z, 2)))
            continue
        if method == "did_level":
            # long event: scope/control ratio vs its own 12 pre-event months (trend MA would absorb it)
            S = d[sc].groupby("date")[col].sum()
            Cc = d[ctl].groupby("date")[col].sum()
            r = S / Cc
            pre = r[(r.index < months[0])].tail(12)
            em = (r.reindex(months) / pre.mean() - 1).dropna()
            post = r[(r.index > months[-1])].head(3) / pre.mean() - 1
            out.append(dict(Event_ID=ev["Event_ID"], in_history=True, n_months=len(em), method=method,
                            measured_on=col, effect_pct=round(100 * float(em.mean()), 1),
                            min_month_effect_pct=round(100 * float(em.min()), 1),
                            z=round(float(em.mean() / (pre.std() / pre.mean() / np.sqrt(len(em)))), 2),
                            next_month_effect_pct=round(100 * float(post.iloc[0]), 1) if len(post) else None,
                            monthly=";".join(f"{k}:{100 * v:+.0f}%" for k, v in em.items())))
            continue
        if method == "yoy":
            A = d[sc].groupby("date")[col].sum()
            mean, em = _yoy_effect(A, months)
            out.append(dict(Event_ID=ev["Event_ID"], in_history=True, n_months=len(em), method=method,
                            measured_on=col, effect_pct=round(100 * mean, 1), z=None,
                            min_month_effect_pct=round(100 * em.min(), 1),
                            monthly=";".join(f"{k}:{100 * v:+.0f}%" for k, v in em.items())))
            continue
        A = d[sc].groupby("date")[col].sum()
        if ev["Scope_Type"] == "Region" and ev["scope_region"] == "Middle East & Africa":
            A = A[A.index >= "2022-01"]
        rs = _ratio_to_trend(A)
        if ctl is not None:
            rc = _ratio_to_trend(d[ctl].groupby("date")[col].sum())
            rel = (rs / rc).dropna()
        else:
            rel = rs.dropna()
        cal = pd.Series(rel.index.str[5:].astype(int), index=rel.index)
        same_type = event_months[event_months["Description"] == ev["Description"]]["date"]
        others = rel[~rel.index.isin(same_type)]
        if ev["Description"] in FIXED_ANNUAL or ev["Scope_Type"] == "Global":
            typical = pd.Series(1.0, index=rel.index) if ev["Scope_Type"] != "Global" else \
                others.groupby(cal[others.index]).median().reindex(cal).set_axis(rel.index)
        else:
            typical = others.groupby(cal[others.index]).median().reindex(cal).set_axis(rel.index)
        eff = (rel / typical - 1).dropna()
        base = eff[~eff.index.isin(same_type)]
        sd = float(base.std()) if len(base) > 6 else np.nan
        em = eff.reindex(months).dropna()
        if em.empty:
            out.append(dict(Event_ID=ev["Event_ID"], in_history=True, n_months=0))
            continue
        mean = float(em.mean())
        z = mean / (sd / np.sqrt(len(em))) if sd and sd > 0 else np.nan
        # post-event rebound (catch-up) in the month after
        nxt = str(pd.Period(months[-1], freq="M") + 1)
        out.append(dict(Event_ID=ev["Event_ID"], in_history=True, n_months=len(em), method=method,
                        measured_on=col, effect_pct=round(100 * mean, 1), min_month_effect_pct=round(100 * em.min(), 1),
                        max_month_effect_pct=round(100 * em.max(), 1), z=round(z, 2) if z == z else None,
                        next_month_effect_pct=round(100 * float(eff.get(nxt, np.nan)), 1),
                        monthly=";".join(f"{k}:{100 * v:+.0f}%" for k, v in em.items())))
    res = events.merge(pd.DataFrame(out), on="Event_ID", how="left")

    def verdict(r):
        if r["start"] > pd.Timestamp("2026-10-31"):
            return "future event (forecast window)"
        if r.get("method") == "data_loss":
            if r["z"] is not None and r["z"] > 3:
                return (f"data-quality event: {int(r['rows_lost'])} rows lost vs {r['rows_lost_baseline']} "
                        "normal (imputed, weight 0)")
            return f"no effect: {int(r['rows_lost'])} rows lost = normal background ({r['rows_lost_baseline']})"
        if r.get("method") == "ramp":
            return f"real effect: new region ramps x{r['ramp_multiple_12m']} over first 12 months"
        if r["Event_Type"] == "Price Change":
            return "see pre-buy analysis"
        if r["Event_Type"] == "Holiday" and r["Description"] in MOVING:
            return "real effect (see holiday regression; spread over 2 months)"
        if pd.isna(r.get("effect_pct")):
            return "not measurable (no history in scope)"
        z = r.get("z")
        if abs(r["effect_pct"]) >= 5 and (z is None or pd.isna(z) or abs(z) >= 2):
            return "real effect"
        if abs(r["effect_pct"]) >= 3 and z is not None and not pd.isna(z) and abs(z) >= 1.5:
            return "weak effect"
        return "no detectable effect"
    res["verdict"] = res.apply(verdict, axis=1)
    return res


def prebuy_analysis(d: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Pull-forward before each January price increase, by channel.
    uplift = (Nov+Dec)/(Sep+Oct) of the year before vs the median of the other years;
    dip    = (Jan+Feb)/(Sep+Oct) likewise. Electrical Assemblies excluded (chip shortage)."""
    x = d[(d["family"] != "Electrical Assemblies") & (d["date"] <= "2026-10")]
    pc = events[(events["Event_Type"] == "Price Change") & events["Description"].str.contains("Annual")]
    inc = {int(s.year): p for s, p in zip(pc["start"], pc["pct"])}
    ann = events[events["Event_Type"] == "Announcement"]
    announced = {int(s.year) + 1 for s in ann["start"]}
    rows = []
    for ch in ["All", "Direct OEM", "Tier-1 Distributor", "Aftermarket", "Export"]:
        y = x if ch == "All" else x[x["channel"] == ch]
        m = y.groupby("date")["demand"].sum()
        r = []
        for Y, p in inc.items():
            if Y < 2020 or Y > 2026:
                continue
            so = m.get(f"{Y-1}-09", np.nan) + m.get(f"{Y-1}-10", np.nan)
            r.append(dict(year=Y, pct=p, announced=Y in announced,
                          nd=(m.get(f"{Y-1}-11", np.nan) + m.get(f"{Y-1}-12", np.nan)) / so,
                          jf=(m.get(f"{Y}-01", np.nan) + m.get(f"{Y}-02", np.nan)) / so))
        r = pd.DataFrame(r)
        normal = r[(r["year"] != 2021)]              # 2021: post-COVID rebound distorts Q4-2020 base
        for _, row in r.iterrows():
            ref = normal[normal["year"] != row["year"]]
            rows.append(dict(channel=ch, year=int(row["year"]), price_increase_pct=row["pct"],
                             announced_in_advance=bool(row["announced"]),
                             novdec_uplift_pct=round(100 * (row["nd"] / ref["nd"].median() - 1), 1),
                             janfeb_dip_pct=round(100 * (row["jf"] / ref["jf"].median() - 1), 1),
                             distorted_by_covid=row["year"] == 2021))
    return pd.DataFrame(rows)


def holiday_effects(d: pd.DataFrame, event_months: pd.DataFrame) -> pd.DataFrame:
    """Moving holidays (Lunar New Year, Ramadan): regress the scope/control ratio-to-trend on the
    fraction of the month covered, so the effect is per 'full month of holiday'."""
    rows = []
    for desc, region in [("Lunar New Year holiday period", "Asia Pacific"), ("Ramadan", "Middle East & Africa"),
                         ("North American summer retooling weeks", "North America"),
                         ("European summer plant shutdowns", "Europe")]:
        x = d[d["date"] <= "2026-10"]
        if region == "Middle East & Africa":
            x = x[x["date"] >= "2022-03"]
        rs = _ratio_to_trend(x[x["region"] == region].groupby("date")["demand"].sum())
        rc = _ratio_to_trend(x[x["region"] != region].groupby("date")["demand"].sum())
        rel = (rs / rc).dropna()
        em = event_months[event_months["Description"] == desc].groupby("date")["frac"].max()
        f = em.reindex(rel.index).fillna(0.0)
        X = np.column_stack([np.ones(len(f)), f.values])
        y = np.log(rel.values)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        se = np.sqrt(resid.var(ddof=2) / max(((f - f.mean()) ** 2).sum(), 1e-9))
        nxt = f.shift(1).fillna(0)                          # catch-up the month after
        X2 = np.column_stack([np.ones(len(f)), f.values, nxt.values])
        b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
        rows.append(dict(holiday=desc, region=region, effect_per_full_month_pct=round(100 * (np.exp(beta[1]) - 1), 1),
                         t_stat=round(beta[1] / se, 1), next_month_rebound_pct=round(100 * (np.exp(b2[2]) - 1), 1),
                         n_event_months=int((f > 0).sum())))
    return pd.DataFrame(rows)
