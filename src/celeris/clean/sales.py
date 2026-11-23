"""Clean train.csv / test.csv into one row per Date-SKU-Region.

Each rule is logged to the DQ log (rule ids DQ-0xx). The diagnosis behind
every rule is documented in docs/03-data-cleaning.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from celeris.config import FIRST_MONTH, LAST_TRAIN_MONTH
from celeris.clean.normalize import (
    map_unique, normalize_channel, normalize_customer, normalize_region, normalize_sku,
    parse_month, to_number,
)
from celeris.dq import DQLog, profile

F = "train.csv"
SURCHARGE = ("2022-07", "2022-12")          # EV-026: temporary 6 % raw-material surcharge


def _tag(df, mask, rule):
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    df.loc[mask, "dq_rules"] = df.loc[mask, "dq_rules"] + rule + ";"


def _near(x, target, tol=0.02):
    return (x / target - 1).abs() <= tol


def price_period(date: pd.Series) -> pd.Series:
    """Prices change on 1 Jan; the 2022-H2 surcharge is its own price period."""
    yr = date.str[:4]
    s = (date >= SURCHARGE[0]) & (date <= SURCHARGE[1])
    return yr.where(~s, yr + "S")


# --------------------------------------------------------------------------- keys & labels
def normalise_keys(df: pd.DataFrame, log: DQLog, master: pd.DataFrame, file=F) -> pd.DataFrame:
    df = df.copy()
    df["dq_rules"] = ""

    # Dates ------------------------------------------------------------------
    df["date"] = map_unique(df["Date"], parse_month)
    nonstd = ~df["Date"].str.fullmatch(r"\d{4}-\d{2}", na=False)
    log.log(file, "DQ-010", "date_format", "Date", "parsed_to_YYYY-MM", nonstd, df,
            id_col="Record_ID" if "Record_ID" in df else None, before=df["Date"], after=df["date"],
            note="7 formats seen: YYYY-MM, YYYY-MM-DD, MM/YYYY, YYYY/MM, Mon YYYY, Mon-YYYY, MM-YYYY")
    _tag(df, nonstd, "DQ-010")
    if "Load_Timestamp" in df:
        load_m = (pd.to_datetime(df["Load_Timestamp"]).dt.to_period("M") - 1).astype(str)
        bad = df["date"].isna() | (df["date"] < FIRST_MONTH) | (df["date"] > LAST_TRAIN_MONTH)
        before = df["date"].copy()
        # Load_Timestamp month == Date + 1 month for 99.99 % of rows -> reliable fallback.
        df.loc[bad, "date"] = load_m[bad]
        log.log(file, "DQ-011", "date_out_of_range", "Date", "replaced_by_load_timestamp_month_minus_1",
                bad, df, id_col="Record_ID", before=before, after=df["date"],
                note="Typo years such as 2062-01 (should be 2026-01); Year/Month and Load_Timestamp agree")
        _tag(df, bad, "DQ-011")
    ym = df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2)
    mism = ym != df["date"]
    log.log(file, "DQ-012", "year_month_mismatch", "Year/Month", "rebuilt_from_Date (rule 1)", mism, df,
            id_col="Record_ID" if "Record_ID" in df else None, before=ym, after=df["date"],
            note="Load_Timestamp confirms Date in 100 % of these rows")
    _tag(df, mism, "DQ-012")
    df["Year"] = df["date"].str[:4].astype(int)
    df["Month"] = df["date"].str[5:].astype(int)

    # SKU ------------------------------------------------------------------------
    df["sku"] = map_unique(df["SKU_ID"], normalize_sku)
    ch = df["sku"] != df["SKU_ID"]
    log.log(file, "DQ-020", "sku_format", "SKU_ID", "normalised_to_AAA-NNN", ch, df,
            id_col="Record_ID" if "Record_ID" in df else None, before=df["SKU_ID"], after=df["sku"])
    _tag(df, ch, "DQ-020")
    unknown = ~df["sku"].isin(master["SKU_ID"])
    log.log(file, "DQ-020b", "sku_not_in_master", "SKU_ID", "quarantined", unknown, df,
            id_col="Record_ID" if "Record_ID" in df else None, before=df["SKU_ID"])

    # Region / channel / customer --------------------------------------------------
    for col, new, fn, rule in [("Region", "region", normalize_region, "DQ-021"),
                               ("Sales_Channel", "channel", normalize_channel, "DQ-023")]:
        df[new] = map_unique(df[col], fn)
        ch = df[col].notna() & (df[new] != df[col])
        log.log(file, rule, "label_variant", col, "mapped_to_canonical", ch, df,
                id_col="Record_ID" if "Record_ID" in df else None, before=df[col], after=df[new])
        _tag(df, ch, rule)
    cust = map_unique(df["Customer_Name"], lambda v: normalize_customer(v)[0])
    df["customer"] = cust
    ch = df["Customer_Name"] != df["customer"]
    log.log(file, "DQ-024", "label_variant", "Customer_Name", "fuzzy_matched_to_12_canonical", ch, df,
            id_col="Record_ID" if "Record_ID" in df else None, before=df["Customer_Name"], after=df["customer"],
            note="typos, case, double spaces, missing spaces, 'Inc.' suffix")
    _tag(df, ch, "DQ-024")

    # Hierarchy from master -------------------------------------------------------
    m = master.set_index("SKU_ID")
    fam = df["sku"].map(m["Product_Family"])
    miss = df["Product_Family"].isna()
    log.log(file, "DQ-025", "missing_value", "Product_Family", "filled_from_sku_master", miss, df,
            id_col="Record_ID" if "Record_ID" in df else None, after=fam)
    _tag(df, miss, "DQ-025")
    desc = df["sku"].map(m["SKU_Description"])
    miss = df["SKU_Description"].isna()
    log.log(file, "DQ-026", "missing_value", "SKU_Description", "filled_from_sku_master", miss, df,
            id_col="Record_ID" if "Record_ID" in df else None, after=desc)
    _tag(df, miss, "DQ-026")
    df["family"] = fam
    df["description"] = desc
    # Family -> BU is one-to-one; the master has one wrong BU (PMP-004), so derive BU from family.
    fam_bu = master.groupby("Product_Family")["Business_Unit"].agg(lambda s: s.mode().iloc[0])
    df["bu"] = df["family"].map(fam_bu)
    if "Business_Unit" in df:
        ch = df["Business_Unit"] != df["bu"]
        log.log(file, "DQ-027", "hierarchy_conflict", "Business_Unit", "set_from_family (family->BU is 1:1)",
                ch, df, id_col="Record_ID" if "Record_ID" in df else None,
                before=df["Business_Unit"], after=df["bu"])
    return df


def dedupe(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df = df.assign(_ts=pd.to_datetime(df["Load_Timestamp"])).sort_values("_ts", kind="stable")
    dup = df.duplicated("Record_ID", keep="last")
    exact = df.duplicated([c for c in df.columns if c not in ("dq_rules",)], keep="last")
    log.log(F, "DQ-030", "duplicate_record_exact", "Record_ID", "dropped_reloaded_copy", dup & exact, df,
            id_col="Record_ID")
    log.log(F, "DQ-031", "duplicate_record_restated", "Record_ID",
            "kept_latest_Load_Timestamp (rule 2)", dup & ~exact, df, id_col="Record_ID",
            before=df["Units_Sold"], note="Same record re-loaded with corrected Units/Revenue")
    return df.loc[~dup].drop(columns="_ts")


def impute_region(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    """Missing Region: the region that is absent for that SKU in that month AND matches the
    (SKU, customer, channel) combination. Unique for all observed cases."""
    df = df.copy()
    ok = df[df["region"].notna()]
    present = ok.groupby(["sku", "date"])["region"].agg(set)
    combo = ok.groupby(["sku", "customer", "channel"])["region"].agg(set)
    allreg = ok.groupby("sku")["region"].agg(set)
    fill, conf = {}, {}
    for i, r in df[df["region"].isna()].iterrows():
        cand = allreg.get(r.sku, set()) - present.get((r.sku, r.date), set())
        both = cand & combo.get((r.sku, r.customer, r.channel), set())
        pick = both if len(both) == 1 else (cand if len(cand) == 1 else set())
        if len(pick) == 1:
            fill[i] = next(iter(pick))
            conf[i] = "high" if len(both) == 1 else "medium"
    df.loc[list(fill), "region"] = pd.Series(fill)
    miss = df.index.isin(list(fill))
    log.log(F, "DQ-022", "missing_value", "Region", "inferred_from_sku_month_gap_and_customer_channel",
            miss, df, id_col="Record_ID", after=df["region"])
    _tag(df, miss, "DQ-022")
    still = df["region"].isna()
    log.log(F, "DQ-022b", "missing_value_unresolved", "Region", "quarantined", still, df, id_col="Record_ID")
    return df[~still]


# --------------------------------------------------------------------------- numbers
def fix_numbers(df: pd.DataFrame, log: DQLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    units_raw = df["Units_Sold"]
    df["units"] = to_number(units_raw)
    sep = units_raw.str.contains(",", na=False)
    log.log(F, "DQ-040", "number_format", "Units_Sold", "removed_thousands_separator", sep, df,
            id_col="Record_ID", before=units_raw, after=df["units"])
    dash = units_raw.str.strip().eq("-")
    log.log(F, "DQ-040b", "missing_value", "Units_Sold", "dash_treated_as_missing", dash, df, id_col="Record_ID")
    df["price"] = to_number(df["Unit_Price_USD"])
    df["rev"] = to_number(df["Revenue_USD"])
    df["units_in"], df["price_in"], df["rev_in"] = df["units"], df["price"], df["rev"]

    # Sign errors: |units| sits at the series' normal level, revenue negative too,
    # spread evenly over all years -> sign flip, not returns.
    neg = df["units"] < 0
    log.log(F, "DQ-041", "sign_error", "Units_Sold", "absolute_value", neg, df, id_col="Record_ID",
            before=df["units"], after=df["units"].abs(),
            note="|units|/series median ~1.0 (p10 0.57, p90 1.6); evenly spread 2019-2026")
    _tag(df, neg, "DQ-041")
    df["units"] = df["units"].abs()
    df["rev"] = df["rev"].abs()

    # Canonical price per SKU x price-period (rule 4: same in every region, changes 1 Jan).
    df["pperiod"] = price_period(df["date"])
    good = df["price"] > 0
    canon = (df[good].assign(p=df.loc[good, "price"].round(2))
             .groupby(["sku", "pperiod"])["p"].agg(lambda s: s.mode().iloc[0]).rename("pc"))
    df = df.join(canon, on=["sku", "pperiod"])
    # fallback for SKU-periods with no valid price: neighbour year adjusted (rare)
    df["pc"] = df.groupby("sku")["pc"].transform(lambda s: s.ffill().bfill())
    ratio = df["price"] / df["pc"]

    level = (df[df["units"] > 0].groupby(["sku", "region"])["units"].median().rename("lvl"))
    df = df.join(level, on=["sku", "region"])
    rel = df["units"] / df["lvl"]

    cents = _near(ratio, 100)
    log.log(F, "DQ-050", "price_unit_error", "Unit_Price_USD", "divided_by_100 (price keyed in cents)",
            cents, df, id_col="Record_ID", before=df["price"], after=df["pc"],
            note="ratio to canonical = 100 (or 106 / 94.3 around the 2022 surcharge); revenue already correct")
    _tag(df, cents, "DQ-050")

    pack = _near(ratio, 10) & (rel < 0.3)
    log.log(F, "DQ-051", "unit_of_measure_error", "Units_Sold/Unit_Price_USD",
            "units x10, price /10 (booked in packs of 10); revenue unchanged", pack, df, id_col="Record_ID",
            before=df["units"], after=df["units"] * 10,
            note="India 2021-07..12 only; units ~0.1x normal and price 10x, revenue consistent")
    _tag(df, pack, "DQ-051")
    df.loc[pack, "units"] = df.loc[pack, "units"] * 10

    zero_price = (df["price"] == 0) & (df["units"] > 0)
    log.log(F, "DQ-053", "missing_value", "Unit_Price_USD", "price_0_on_sold_row -> canonical price",
            zero_price, df, id_col="Record_ID", before=df["price"], after=df["pc"])
    _tag(df, zero_price, "DQ-053")

    other = (df["price"] > 0) & ~_near(ratio, 1, 0.005) & ~cents & ~pack
    # Constant ratio within one region-quarter -> currency (EUR keyed as USD, Europe 2022-Q1).
    log.log(F, "DQ-052", "currency_error", "Unit_Price_USD/Revenue_USD",
            "converted to USD via canonical price; revenue recomputed", other, df, id_col="Record_ID",
            before=ratio.round(3), after=pd.Series(1.0, index=df.index),
            note="Europe 2022-01..03: price & revenue both x0.89 (local currency), units correct")
    _tag(df, other, "DQ-052")
    df.loc[other, "rev"] = df.loc[other, "units"] * df.loc[other, "pc"]
    df["price"] = df["pc"]

    # Revenue triangle (price now trusted): q = revenue / (units * price)
    q = df["rev"] / (df["units"] * df["price"])
    has = df["units"].notna() & (df["units"] > 0) & df["rev"].notna()
    ok = has & _near(q, 1, 0.011)
    for k, rule in [(0.1, "DQ-055a"), (0.01, "DQ-055b"), (0.001, "DQ-055c")]:
        m = has & _near(q, k, 0.03) & (rel > 1 / (k * 4))     # units inflated by 1/k
        log.log(F, rule, "unit_scale_error", "Units_Sold", f"divided_by_{int(round(1/k))}", m, df,
                id_col="Record_ID", before=df["units"], after=(df["units"] * k).round(),
                note="units >> series level and revenue/(units*price) = %g" % k)
        _tag(df, m, rule)
        df.loc[m, "units"] = (df.loc[m, "units"] * k).round()
        ok |= m
    r10 = has & ~ok & _near(q, 10, 0.03)
    log.log(F, "DQ-056", "revenue_scale_error", "Revenue_USD", "recomputed (revenue x10, units at normal level)",
            r10, df, id_col="Record_ID", before=df["rev"], after=df["units"] * df["price"])
    _tag(df, r10, "DQ-056")
    ok |= r10
    # Other disagreements: keep whichever of (units, revenue/price) is integer and nearer the level.
    rest = has & ~ok & ~pack
    alt = df["rev"] / df["price"]
    alt_int = (alt - alt.round()).abs() < 0.02
    d_units = np.log((df["units"] + 1) / (df["lvl"] + 1)).abs()
    d_alt = np.log((alt + 1) / (df["lvl"] + 1)).abs()
    use_alt = rest & alt_int & (d_alt < d_units)
    log.log(F, "DQ-057", "units_revenue_conflict", "Units_Sold", "units_replaced_by_revenue/price",
            use_alt, df, id_col="Record_ID", before=df["units"], after=alt.round())
    log.log(F, "DQ-058", "units_revenue_conflict", "Revenue_USD", "revenue_recomputed_from_units",
            rest & ~use_alt, df, id_col="Record_ID", before=df["rev"], after=df["units"] * df["price"])
    _tag(df, use_alt, "DQ-057")
    _tag(df, rest & ~use_alt, "DQ-058")
    df.loc[use_alt, "units"] = alt[use_alt].round()

    # Missing units ----------------------------------------------------------------
    nanu = df["units"].isna()
    rec = nanu & (df["rev"] > 0) & ((df["rev"] / df["price"] - (df["rev"] / df["price"]).round()).abs() < 0.02)
    log.log(F, "DQ-042", "missing_value", "Units_Sold", "recovered_as_revenue/price", rec, df,
            id_col="Record_ID", before=df["Units_Sold"], after=(df["rev"] / df["price"]).round())
    _tag(df, rec, "DQ-042")
    df.loc[rec, "units"] = (df.loc[rec, "rev"] / df.loc[rec, "price"]).round()
    z = df["units"].isna() & (df["rev"] == 0)
    log.log(F, "DQ-043", "missing_value", "Units_Sold", "set_to_0 (revenue = 0 -> no-order row)", z, df,
            id_col="Record_ID", before=df["Units_Sold"], after=pd.Series(0, index=df.index))
    _tag(df, z, "DQ-043")
    df.loc[z, "units"] = 0
    still = df["units"].isna()
    log.log(F, "DQ-044", "missing_value_unrecoverable", "Units_Sold",
            "kept_as_missing (imputed later for lags, excluded from training loss)", still, df, id_col="Record_ID",
            note="Includes the Latin America ERP migration 2023-03..05 (EV-033)")
    _tag(df, still, "DQ-044")

    revna = df["rev_in"].isna()
    log.log(F, "DQ-059", "missing_value", "Revenue_USD", "recomputed_from_units_x_price", revna, df,
            id_col="Record_ID")
    df["revenue"] = df["units"] * df["price"]
    df["units_missing"] = still
    price_table = canon.reset_index().rename(columns={"pc": "price_usd"})
    return df, price_table


# --------------------------------------------------------------------------- grid
def build_grid(df: pd.DataFrame, stock: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    """One row per Date-SKU-Region for each series' active span. Months with no ERP row are
    data loss (rule 3: missing row != zero). Restore from the stockout file where possible."""
    df = df.copy()
    df["p"] = pd.PeriodIndex(df["date"], freq="M")
    span = df.groupby(["sku", "region"])["p"].agg(["min", "max"])
    keys = [(s, r, p) for (s, r), row in span.iterrows() for p in pd.period_range(row["min"], row["max"], freq="M")]
    full = pd.DataFrame(keys, columns=["sku", "region", "p"])
    out = full.merge(df, on=["sku", "region", "p"], how="left", indicator=True)
    missing = out["_merge"] == "left_only"
    out["row_status"] = np.where(missing, "missing_row", "observed")
    out["date"] = out["p"].astype(str)
    # carry static attributes within series
    for c in ["family", "bu", "description", "customer", "channel"]:
        out[c] = out.groupby(["sku", "region"])[c].transform(lambda s: s.ffill().bfill())
    out["dq_rules"] = out["dq_rules"].fillna("")
    st = stock.set_index(["date", "sku", "region"])["Shipped_Units"]
    k = pd.MultiIndex.from_frame(out[["date", "sku", "region"]])
    shipped = pd.Series(st.reindex(k).values, index=out.index)
    restore = missing & shipped.notna()
    out.loc[restore, "units"] = shipped[restore]
    out.loc[restore, "row_status"] = "restored_from_stockout_file"
    pc = out.groupby(["sku", "date"])["price"].transform("max")
    out["price"] = out["price"].fillna(pc)
    out["price"] = out.groupby("sku")["price"].transform(lambda s: s.ffill().bfill())
    out["revenue"] = out["units"] * out["price"]
    log.log(F, "DQ-060", "missing_row", "(row)", "added_to_grid_as_missing (data loss, not zero)",
            missing & ~restore, out, n=None, id_col=None,
            before=out["date"] + "|" + out["sku"] + "|" + out["region"])
    log.log(F, "DQ-061", "missing_row", "(row)", "restored_from_inventory_stockouts.Shipped_Units",
            restore, out, before=out["date"] + "|" + out["sku"] + "|" + out["region"], after=out["units"])
    out["units_missing"] = out["units"].isna()
    out.loc[restore, "dq_rules"] = "DQ-061;"
    out.loc[missing & ~restore, "dq_rules"] = "DQ-060;"
    return out.drop(columns=["_merge", "p"])


TRAIN_COLS = ["date", "Year", "Month", "family", "bu", "sku", "description", "customer", "channel", "region",
              "units", "price", "revenue", "units_missing", "row_status", "dq_rules",
              "Record_ID", "units_in", "price_in", "rev_in"]


def clean_train(raw: pd.DataFrame, master: pd.DataFrame, stock: pd.DataFrame, log: DQLog):
    log.file_summary(F, rows_in=len(raw), before=profile(raw))
    df = normalise_keys(raw, log, master)
    df = df[df["sku"].isin(master["SKU_ID"])]
    df = dedupe(df, log)
    df = impute_region(df, log)
    df, prices = fix_numbers(df, log)
    dupkey = df.duplicated(["date", "sku", "region"], keep=False)
    log.log(F, "DQ-032", "duplicate_key", "Date-SKU-Region", "none_after_Record_ID_dedupe", dupkey, df,
            id_col="Record_ID")
    assert not dupkey.any(), "duplicate Date-SKU-Region after dedupe"
    df = build_grid(df, stock, log)
    # Packs-of-10 rows were stored as round(units/10): x10 is only accurate to +-5 units.
    # Where the stockout file holds the exact shipped quantity, use it.
    sh = stock.set_index(["date", "sku", "region"])["Shipped_Units"]
    exact = pd.Series(sh.reindex(pd.MultiIndex.from_frame(df[["date", "sku", "region"]])).values, index=df.index)
    m = df["dq_rules"].str.contains("DQ-051") & exact.notna() & (exact != df["units"])
    log.log(F, "DQ-062", "lossy_fix_refined", "Units_Sold", "replaced x10 estimate by exact Shipped_Units "
            "from inventory_stockouts", m, df, before=df["units"], after=exact)
    df.loc[m, "units"] = exact[m]
    df.loc[m, "revenue"] = df.loc[m, "units"] * df.loc[m, "price"]
    df.loc[m, "dq_rules"] += "DQ-062;"
    df = df[TRAIN_COLS].sort_values(["sku", "region", "date"]).reset_index(drop=True)
    df["Year"] = df["date"].str[:4].astype(int)
    df["Month"] = df["date"].str[5:].astype(int)
    log.file_summary(F, rows_out=len(df), after=profile(df.drop(columns=["dq_rules"])))
    return df, prices


def clean_test(raw: pd.DataFrame, master: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    log.file_summary("test.csv", rows_in=len(raw), before=profile(raw))
    df = normalise_keys(raw, log, master, file="test.csv")
    df = df.rename(columns={})
    out = df[["date", "Year", "Month", "family", "bu", "sku", "description", "customer", "channel", "region",
              "Date", "SKU_ID", "Region"]].copy()
    assert out[["date", "sku", "region"]].notna().all().all()
    assert not out.duplicated(["date", "sku", "region"]).any()
    log.file_summary("test.csv", rows_out=len(out))
    return out
