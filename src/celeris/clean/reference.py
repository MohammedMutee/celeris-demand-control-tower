"""Cleaning for the reference files: macro indicators, SKU master, events, stockouts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from celeris.clean.normalize import map_unique, normalize_region, normalize_sku, parse_month
from celeris.dq import DQLog, profile


# --------------------------------------------------------------------------- macro
def clean_macro(raw: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    F = "macro_indicators.csv"
    log.file_summary(F, rows_in=len(raw), before=profile(raw))
    df = raw.copy()
    df["date"] = df["Date"].map(parse_month)
    cols = [c for c in df.columns if c not in ("Date", "Year", "Month", "date")]

    # Duplicate month (2022-06 appears twice with different PMI / orders):
    # keep the row closest to the average of its neighbours.
    dup = df["date"].duplicated(keep=False)
    if dup.any():
        keep = []
        for d, g in df[dup].groupby("date"):
            nb = df[~dup & (df["date"].isin([str(pd.Period(d) - 1), str(pd.Period(d) + 1)]))][cols].mean()
            z = ((g[cols] - nb).abs() / df[cols].std()).sum(axis=1)
            keep.append(z.idxmin())
        drop = df.index[dup & ~df.index.isin(keep)]
        log.log(F, "DQ-104", "duplicate_month", "Date", "kept_row_closest_to_neighbour_mean",
                df.index.isin(drop), df, id_col="date")
        df = df.drop(index=drop)

    # Auto_Production_YoY_Pct is a fraction in 2019 (-0.015) and a percent afterwards.
    c = "Auto_Production_YoY_Pct"
    frac = (df["date"] < "2020-01") & (df[c].abs() < 1)
    log.log(F, "DQ-102", "unit_inconsistency", c, "x100 (fraction -> percent)", frac, df, id_col="date",
            before=df[c], after=df[c] * 100)
    df.loc[frac, c] = df.loc[frac, c] * 100

    # Decimal shifts: a value that is ~10x off its neighbours and lands in range when rescaled.
    for c in cols:
        s = df[c]
        med = s.rolling(7, center=True, min_periods=3).median()
        for k in (10, 0.1):
            m = ((s * k / med - 1).abs() < 0.1) & ((s / med - 1).abs() > 0.5)
            if m.any():
                log.log(F, "DQ-101", "decimal_shift", c, f"x{k}", m, df, id_col="date",
                        before=s, after=s * k)
                df.loc[m, c] = s[m] * k

    na = df[cols].isna()
    for c in cols:
        log.log(F, "DQ-100", "missing_value", c, "linear_interpolation_in_time", na[c], df, id_col="date")
    df[cols] = df[cols].interpolate(limit_direction="both")
    df = df.sort_values("date").reset_index(drop=True)
    df["Year"] = df["date"].str[:4].astype(int)
    df["Month"] = df["date"].str[5:].astype(int)
    assert df["date"].is_unique and not df[cols].isna().any().any()
    log.file_summary(F, rows_out=len(df))
    return df[["date"] + cols]


# --------------------------------------------------------------------------- master
def check_master(master: pd.DataFrame, train: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    """Master data vs transactions. Returns a conflicts table (Expert WS2)."""
    F = "sku_master.csv"
    log.file_summary(F, rows_in=len(master), before=profile(master))
    rows = []
    fam_bu = master.groupby("Product_Family")["Business_Unit"].agg(lambda s: s.mode().iloc[0])
    bad_bu = master["Business_Unit"] != master["Product_Family"].map(fam_bu)
    log.log(F, "DQ-110", "hierarchy_conflict", "Business_Unit", "transactions_use_family_BU", bad_bu, master,
            id_col="SKU_ID", before=master["Business_Unit"], after=master["Product_Family"].map(fam_bu))
    for _, r in master[bad_bu].iterrows():
        rows.append(dict(sku=r.SKU_ID, conflict="business_unit", master_value=r.Business_Unit,
                         transaction_value=fam_bu[r.Product_Family],
                         resolution="Trust transactions + family->BU rule; master entry is wrong"))

    obs = train[train["units"].fillna(0) > 0].groupby("sku")["date"].agg(["min", "max"])
    m = master.set_index("SKU_ID").join(obs)
    pre = m["min"].notna() & (m["min"] < m["Launch_Date"])
    for s, r in m[pre].iterrows():
        rows.append(dict(sku=s, conflict="sales_before_launch", master_value=r.Launch_Date,
                         transaction_value=r["min"],
                         resolution="Sales are real (all regions, normal levels); master Launch_Date too late"))
    log.log(F, "DQ-111", "lifecycle_conflict", "Launch_Date", "flagged; sales kept", pre.values, m.reset_index(),
            id_col="SKU_ID")
    post = m["EOL_Date"].notna() & m["max"].notna() & (m["max"] > m["EOL_Date"])
    for s, r in m[post].iterrows():
        rows.append(dict(sku=s, conflict="sales_after_eol", master_value=r.EOL_Date, transaction_value=r["max"],
                         resolution="flag"))
    disc = (m["Lifecycle_Status"] == "Discontinued") & (m["EOL_Date"].isna() | (m["EOL_Date"] > "2026-10"))
    for s, r in m[disc].iterrows():
        rows.append(dict(sku=s, conflict="status_vs_eol", master_value=r.Lifecycle_Status,
                         transaction_value=r.EOL_Date, resolution="flag"))
    planned_hist = (m["Lifecycle_Status"] == "Planned Launch") & m["min"].notna()
    for s, r in m[planned_hist].iterrows():
        rows.append(dict(sku=s, conflict="planned_launch_has_history", master_value=r.Launch_Date,
                         transaction_value=r["min"], resolution="flag"))
    # Replacement chains
    for _, r in master[master["Replaces_SKU"].notna()].iterrows():
        old = master.set_index("SKU_ID").loc[r.Replaces_SKU]
        if pd.isna(old.EOL_Date):
            rows.append(dict(sku=r.SKU_ID, conflict="predecessor_has_no_eol", master_value=r.Replaces_SKU,
                             transaction_value="", resolution="flag"))
        elif r.Launch_Date > old.EOL_Date:
            rows.append(dict(sku=r.SKU_ID, conflict="gap_in_replacement", master_value=r.Launch_Date,
                             transaction_value=old.EOL_Date, resolution="flag"))
    # Regions per SKU: does an SKU sell in a region in train but not test (lost) or vice versa?
    log.file_summary(F, rows_out=len(master), conflicts=len(rows))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- events
def clean_events(raw: pd.DataFrame, log: DQLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (events, event_months). event_months has one row per event x month with the
    fraction of the month covered, ready to join on scope."""
    F = "events_calendar.csv"
    log.file_summary(F, rows_in=len(raw))
    ev = raw.copy()
    ev["start"] = pd.to_datetime(ev["Start_Date"].where(ev["Start_Date"].str.len() > 7, ev["Start_Date"] + "-01"))
    end_s = ev["End_Date"].fillna("2028-12")
    ev["end"] = pd.to_datetime(end_s.where(end_s.str.len() > 7, None))
    month_end = (pd.to_datetime(end_s.where(end_s.str.len() <= 7) + "-01") + pd.offsets.MonthEnd(0))
    ev["end"] = ev["end"].fillna(month_end)
    ev["open_ended"] = raw["End_Date"].isna()
    log.log(F, "DQ-120", "open_ended_event", "End_Date", "extended_to_end_of_forecast_window",
            ev["open_ended"], ev, id_col="Event_ID")
    sv = ev["Scope_Value"].str.split("|", expand=True)
    ev["scope_customer"] = np.where(ev["Scope_Type"] == "Customer", sv[0].str.strip(), None)
    ev["scope_region"] = np.where(ev["Scope_Type"] == "Region", ev["Scope_Value"],
                                  np.where(ev["Scope_Type"] == "Customer", sv[1].str.strip() if 1 in sv else None, None))
    ev["scope_region"] = ev["scope_region"].map(lambda v: normalize_region(v) if isinstance(v, str) else None)
    ev["pct"] = ev["Description"].str.extract(r"(\d+(?:\.\d+)?)%")[0].astype(float)
    rows = []
    for _, r in ev.iterrows():
        for p in pd.period_range(r.start, r.end, freq="M"):
            ms, me = p.start_time, p.end_time
            cov = (min(me, r.end + pd.Timedelta(hours=23, minutes=59)) - max(ms, r.start)).total_seconds()
            frac = max(0.0, cov) / (me - ms).total_seconds()
            rows.append(dict(Event_ID=r.Event_ID, date=str(p), frac=round(min(frac, 1.0), 3)))
    em = pd.DataFrame(rows).merge(ev[["Event_ID", "Scope_Type", "Scope_Value", "scope_region", "scope_customer",
                                      "Event_Type", "Description", "pct"]], on="Event_ID")
    log.file_summary(F, rows_out=len(ev), event_months=len(em))
    return ev, em


# --------------------------------------------------------------------------- stockouts
def clean_stockouts(raw: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    F = "inventory_stockouts.csv"
    log.file_summary(F, rows_in=len(raw), before=profile(raw))
    st = raw.copy()
    st["date"] = st["Date"].map(parse_month)
    st["sku"] = map_unique(st["SKU_ID"], normalize_sku)
    st["region"] = map_unique(st["Region"], normalize_region)
    ident = st.Requested_Units + st.Carried_In_Units - st.Shipped_Units - st.Backordered_Units - st.Lost_Units
    log.log(F, "DQ-130", "identity_violation", "Lost_Units", "flagged", ident != 0, st)
    fr = 100 * st.Shipped_Units / (st.Requested_Units + st.Carried_In_Units)
    off = (fr - st.Fill_Rate_Pct).abs() > 2.0
    log.log(F, "DQ-131", "derived_field_inconsistent", "Fill_Rate_Pct",
            "recomputed as Shipped/(Requested+Carried_In)", off, st, before=st.Fill_Rate_Pct, after=fr.round(1),
            note="Fill_Rate_Pct is only approximate in the source (|diff| <= 2pp tolerated)")
    st["fill_rate"] = fr
    st = st.sort_values(["sku", "region", "date"])
    st["p"] = pd.PeriodIndex(st["date"], freq="M")
    prev = st[["sku", "region", "p", "Backordered_Units"]].rename(columns={"Backordered_Units": "bo_prev"})
    prev["p"] = prev["p"] + 1
    st = st.merge(prev, on=["sku", "region", "p"], how="left")
    br = (st.Carried_In_Units > 0) & st.bo_prev.notna() & (st.Carried_In_Units != st.bo_prev)
    log.log(F, "DQ-132", "backorder_chain_break", "Carried_In_Units", "flagged", br, st)
    log.file_summary(F, rows_out=len(st))
    return st.drop(columns=["p"])


def reconcile_stockouts(st: pd.DataFrame, train: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    """Shipped_Units should equal the cleaned Units_Sold: an independent check of our fixes."""
    F = "inventory_stockouts.csv"
    t = train.set_index(["date", "sku", "region"])
    k = pd.MultiIndex.from_frame(st[["date", "sku", "region"]])
    st = st.copy()
    st["train_units_clean"] = t["units"].reindex(k).values
    st["train_units_raw"] = t["units_in"].reindex(k).values
    st["train_row_status"] = t["row_status"].reindex(k).values
    agree_raw = st.train_units_raw == st.Shipped_Units
    agree_clean = st.train_units_clean == st.Shipped_Units
    log.file_summary(F, reconciliation={
        "rows": int(len(st)),
        "shipped_eq_raw_units": int(agree_raw.sum()),
        "shipped_eq_clean_units": int(agree_clean.sum()),
        "no_train_row_restored": int((st.train_row_status == "restored_from_stockout_file").sum()),
    })
    mism = st.train_units_clean.notna() & ~agree_clean
    log.log(F, "DQ-133", "cross_file_mismatch", "Shipped_Units vs train Units_Sold", "flagged for review", mism, st,
            before=st.Shipped_Units, after=st.train_units_clean)
    return st
