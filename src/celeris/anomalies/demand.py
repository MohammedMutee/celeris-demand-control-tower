"""Expected-demand model and unconstrained-demand estimation (WS2 Advanced).

Demand structure discovered in the data:
    units_t = monthly_rate_t x months_covered x order_expected_t
where order_expected / months_covered come from the customer's ordering cycle (cycles.py),
monthly_rate_t = level_t x seasonal_index[family, region, month].

Unconstrained demand (the training target, rule 7):
  1. listed stockout month         -> Requested_Units
  2. month after a listed backorder -> shipped - backorder carried in (catch-up removed)
  3. unlisted chip-shortage months (Electrical Assemblies 2021-03..2022-09) -> estimated
  4. unlisted stockouts elsewhere (dip + catch-up signature) -> estimated
  5. data loss (missing units / rows) -> imputed from the expected model, weight 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

CHIP = ("2021-03", "2022-09")
COVID = ("2020-02", "2020-12")
HALF = 6                         # neighbour window: 6 months each side


# --------------------------------------------------------------------------- helpers
def months_covered(mask: str, period: int) -> float:
    n = mask.count("1")
    return period / n if n else np.nan


def _neighbour_median(x: np.ndarray, half: int = HALF) -> np.ndarray:
    """Median of the `half` values on each side, excluding the centre (NaNs ignored)."""
    pad = np.concatenate([np.full(half, np.nan), x, np.full(half, np.nan)])
    w = sliding_window_view(pad, 2 * half + 1).copy()
    w[:, half] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(w, axis=1)
            cnt = np.sum(~np.isnan(w), axis=1)
    med[cnt < 3] = np.nan
    return med


def seasonal_index(df: pd.DataFrame) -> pd.DataFrame:
    """Family x region x calendar-month multiplicative index from 'normal' months.
    Ratio of the monthly rate to its own centred 13-month mean, median across series."""
    d = df[df["rate"].notna() & df["normal"]].copy()
    d["ma"] = d.groupby(["sku", "region"])["rate"].transform(
        lambda s: s.rolling(13, center=True, min_periods=9).mean())
    d = d[d["ma"] > 0]
    d["r"] = d["rate"] / d["ma"]
    si = d.groupby(["family", "region", "Month"])["r"].median().rename("si").reset_index()
    si["si"] = si["si"] / si.groupby(["family", "region"])["si"].transform("mean")
    return si


# --------------------------------------------------------------------------- main
def build_demand(sales: pd.DataFrame, flags: pd.DataFrame, cycles: pd.DataFrame,
                 stock: pd.DataFrame, log: list) -> tuple[pd.DataFrame, dict]:
    df = sales.merge(flags, on=["sku", "region", "date"], how="left")
    seg_mask = cycles.set_index(["sku", "region", "seg"])["mask"]
    df["mask"] = seg_mask.reindex(pd.MultiIndex.from_frame(df[["sku", "region", "cycle_seg"]])).values
    df["k"] = [months_covered(m, p) for m, p in zip(df["mask"], df["cycle_period"])]
    df["shipped"] = df["units"]

    # ---- stockout file joins
    st = stock.set_index(["date", "sku", "region"])
    key = pd.MultiIndex.from_frame(df[["date", "sku", "region"]])
    df["st_requested"] = st["Requested_Units"].reindex(key).values
    df["st_carried_in"] = st["Carried_In_Units"].reindex(key).values
    df["st_lost"] = st["Lost_Units"].reindex(key).values
    df["st_reason"] = st["Reason"].reindex(key).values
    df["stockout_listed"] = df["st_requested"].notna()
    df = df.sort_values(["sku", "region", "date"]).reset_index(drop=True)
    g = df.groupby(["sku", "region"], sort=False)
    bo = st["Backordered_Units"]
    prev_date = (pd.PeriodIndex(df["date"], freq="M") - 1).astype(str)
    df["bo_prev"] = bo.reindex(pd.MultiIndex.from_arrays([prev_date, df["sku"], df["region"]])).values

    in_chip = (df["family"] == "Electrical Assemblies") & df["date"].between(*CHIP)
    in_covid = df["date"].between(*COVID)
    df["normal"] = ~df["stockout_listed"] & ~in_chip & ~in_covid & df["units"].notna() & df["bo_prev"].isna()

    # ---- monthly-rate view
    df["rate"] = np.where((df["order_expected"] == 1) & df["units"].notna(), df["units"] / df["k"], np.nan)
    si = seasonal_index(df)
    df = df.merge(si, on=["family", "region", "Month"], how="left")
    df["si"] = df["si"].fillna(1.0)
    df = df.sort_values(["sku", "region", "date"]).reset_index(drop=True)
    df["drate"] = df["rate"] / df["si"]

    # ---- local level from clean neighbours (constrained months excluded)
    clean_d = df["drate"].where(df["normal"])
    df["level"] = np.concatenate([_neighbour_median(v.to_numpy()) for _, v in
                                  clean_d.groupby([df["sku"], df["region"]], sort=False)])
    # wider fallback for long constrained spans (chip shortage = 19 months)
    lvl_wide = df.groupby(["sku", "region"], sort=False)["level"].transform(
        lambda s: s.interpolate(limit_direction="both"))
    df["level"] = df["level"].fillna(lvl_wide)
    df["expected"] = np.where(df["order_expected"] == 0, 0.0, df["level"] * df["si"] * df["k"])

    # ---- unconstrained demand
    df["demand"] = df["units"].astype(float)
    df["demand_source"] = np.where(df["units"].notna(), "shipped", "missing")

    m = df["stockout_listed"]
    df.loc[m, "demand"] = df.loc[m, "st_requested"]
    df.loc[m, "demand_source"] = "stockout_listed:requested"
    log.append(("UD-1", "stockout_listed", int(m.sum()), "demand = Requested_Units"))

    m = df["bo_prev"].fillna(0).gt(0) & ~df["stockout_listed"] & df["units"].notna()
    df.loc[m, "demand"] = (df.loc[m, "units"] - df.loc[m, "bo_prev"]).clip(lower=0)
    df.loc[m, "demand_source"] = "backorder_catchup_removed"
    log.append(("UD-2", "backorder_catchup", int(m.sum()), "demand = shipped - backorder carried in"))

    metrics = {}
    # ---- chip shortage: unlisted Electrical Assemblies months
    chip_rows = in_chip & df["units"].notna()
    lst = chip_rows & df["stockout_listed"] & (df["shipped"] > 0)
    # Estimator A: shipped / fill-rate(month, region) learned from listed rows
    fr = (df[lst].groupby(["date", "region"])
          .apply(lambda x: x["shipped"].sum() / (x["st_requested"].sum() + x["st_carried_in"].fillna(0).sum()),
                 include_groups=False).rename("fr"))
    # Estimator B: level x seasonal x cycle x family-month demand index from listed rows
    idx_b = (df[lst].assign(r=lambda x: x["st_requested"] / (x["level"] * x["si"] * x["k"]))
             .groupby("date")["r"].median().rename("ib"))
    df = df.join(fr, on=["date", "region"]).join(idx_b, on="date")
    estA = df["shipped"] / df["fr"]
    estB = df["expected"] * df["ib"]
    # validate both on listed rows (they hold the truth) with leave-one-SKU-out for the indices
    v = df[lst].copy()
    rng = np.random.default_rng(0)
    folds = {s: rng.integers(5) for s in v["sku"].unique()}
    v["fold"] = v["sku"].map(folds)
    pa, pb = pd.Series(np.nan, index=v.index), pd.Series(np.nan, index=v.index)
    for f in range(5):
        tr, te = v[v["fold"] != f], v[v["fold"] == f]
        fr_f = (tr.groupby(["date", "region"]).apply(
            lambda x: x["shipped"].sum() / (x["st_requested"].sum() + x["st_carried_in"].fillna(0).sum()),
            include_groups=False))
        ib_f = (tr["st_requested"] / (tr["level"] * tr["si"] * tr["k"])).groupby(tr["date"]).median()
        pa[te.index] = te["shipped"] / fr_f.reindex(pd.MultiIndex.from_frame(te[["date", "region"]])).values
        pb[te.index] = te["expected"] * ib_f.reindex(te["date"]).values
    truth = v["st_requested"]

    def wape(p):
        ok = p.notna()
        return float((p[ok] - truth[ok]).abs().sum() / truth[ok].sum())

    pab = np.sqrt(pa * pb)
    metrics["chip_estimator_wape"] = {"A_shipped_over_fillrate": wape(pa), "B_expected_x_index": wape(pb),
                                      "geo_mean_AB": wape(pab), "n_validation_rows": int(len(v))}
    best = min(metrics["chip_estimator_wape"], key=lambda k: metrics["chip_estimator_wape"][k]
               if k != "n_validation_rows" else 9)
    est = {"A_shipped_over_fillrate": estA, "B_expected_x_index": estB, "geo_mean_AB": np.sqrt(estA * estB)}[best]
    est = est.fillna(estA).fillna(estB)
    m = chip_rows & ~df["stockout_listed"] & (df["demand_source"] == "shipped") & est.notna()
    df.loc[m, "demand"] = np.maximum(est[m].round(), df.loc[m, "shipped"])
    df.loc[m, "demand_source"] = "stockout_inferred:chip_shortage"
    metrics["chip_estimator_used"] = best
    log.append(("UD-3", "stockout_inferred_chip", int(m.sum()), f"estimator {best}"))
    df = df.drop(columns=["fr", "ib"])

    # ---- unlisted stockouts elsewhere: dip followed by catch-up
    nxt_units = df.groupby(["sku", "region"], sort=False)["shipped"].shift(-1)
    nxt_exp = df.groupby(["sku", "region"], sort=False)["expected"].shift(-1)
    short = df["expected"] - df["shipped"]
    excess = nxt_units - nxt_exp

    def signature(ratio_dip, ratio_up):
        return ((df["order_expected"] == 1) & (df["shipped"] < ratio_dip * df["expected"]) &
                (nxt_units > ratio_up * nxt_exp) & (excess > 0.3 * short) & (excess < 1.5 * short))

    # Validation on listed non-chip stockouts (truth known): sweep thresholds.
    listed_nc = df["stockout_listed"] & ~in_chip & (df["order_expected"] == 1)
    normal_o = df["normal"] & (df["order_expected"] == 1)
    sweep = []
    for a in (0.5, 0.6, 0.7):
        for b in (1.2, 1.3, 1.4):
            s = signature(a, b)
            sweep.append(dict(dip=a, catchup=b, recall_listed=float(s[listed_nc].mean()),
                              false_flag_rate_normal=float(s[normal_o].mean())))
    base_rate = float(listed_nc.sum() / (listed_nc.sum() + normal_o.sum())) * 0.45  # ~31% unlisted (chip ratio)
    for r in sweep:
        tp = base_rate * r["recall_listed"]
        r["est_precision"] = tp / (tp + r["false_flag_rate_normal"]) if tp else 0.0
    metrics["unlisted_detector_sweep"] = sweep
    # Shipment noise (CV ~0.4) makes isolated unlisted stockouts undetectable with useful precision
    # (best ~8%), so we do NOT impute them. We flag the strictest signature and down-weight it.
    sig = signature(0.5, 1.4)
    m = sig & ~df["stockout_listed"] & ~in_chip & (df["demand_source"] == "shipped")
    nm = m.groupby([df["sku"], df["region"]], sort=False).shift(1, fill_value=False)
    df["possible_stockout"] = m
    df["possible_catchup"] = nm & (df["demand_source"] == "shipped")
    log.append(("UD-4", "possible_unlisted_stockout_flagged", int(m.sum()),
                "dip<50% expected then >140% next month; NOT imputed (precision too low), weight 0.5"))

    # ---- data loss
    m = df["units"].isna() & ~df["stockout_listed"]          # a listed Requested_Units beats imputation
    df.loc[m, "demand"] = df.loc[m, "expected"].round()
    df.loc[m, "demand_source"] = np.where(df.loc[m, "row_status"] == "missing_row", "imputed:missing_row",
                                          "imputed:missing_units")
    log.append(("UD-5", "data_loss_imputed", int(m.sum()), "expected model; train_weight 0"))

    df["demand"] = df["demand"].clip(lower=0)
    df["lost_units_est"] = (df["demand"] - df["shipped"]).clip(lower=0).where(
        df["demand_source"].str.startswith("stockout"))
    df["train_weight"] = np.select(
        [df["demand_source"].str.startswith("imputed"),
         df["demand_source"].str.contains("inferred") | df["possible_stockout"] | df["possible_catchup"]],
        [0.0, 0.5], 1.0)
    return df, metrics, si
