"""Series x month panel and the leakage-safe, fold-aware feature builder.

Everything a fold uses is derived from data <= its origin O:
  * ordering cycles (re-detected on truncated history) -> order_expected / months_covered,
    extrapolated into the future from the segment that is current at O;
  * the family x region seasonal index;
  * all lag / rolling features.

Model target (see Obsidian note 14):
    y = rate(t) / scale(o)        rate = demand / months_covered on order months
    units forecast = y_hat * scale(o) * months_covered(t) * order_expected(t)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from celeris import config as C
from celeris.anomalies.cycles import detect_cycles

warnings.filterwarnings("ignore", category=RuntimeWarning)

MONTHS = [str(p) for p in pd.period_range("2019-01", "2028-05", freq="M")]
MIDX = {m: i for i, m in enumerate(MONTHS)}
T_HIST_END = MIDX["2026-10"]
H = 19
MACRO_COLS = ["ISM_Manufacturing_PMI", "Auto_Production_YoY_Pct", "Vehicle_Inventory_Days",
              "Raw_Material_Cost_Index", "New_Vehicle_Orders_Index", "Industrial_Util_Rate_Pct",
              "Fuel_Price_Index", "EV_Adoption_Index"]
CAT_COLS = ["family", "region", "channel", "customer", "powertrain", "size_class"]
COVID = (MIDX["2020-02"], MIDX["2020-12"])


class Panel:
    """Static wide matrices over all series (train series + test-only series)."""

    def __init__(self):
        d = pd.read_parquet(C.CLEAN / "demand.parquet")
        test = pd.read_parquet(C.CLEAN / "test_clean.parquet")
        master = pd.read_parquet(C.CLEAN / "sku_master.parquet").set_index("SKU_ID")
        keys = pd.concat([d[["sku", "region"]], test[["sku", "region"]]]).drop_duplicates()
        keys = keys.sort_values(["sku", "region"]).reset_index(drop=True)
        self.keys = keys
        self.S, self.T = len(keys), len(MONTHS)
        kidx = pd.MultiIndex.from_frame(keys)
        self.sidx = pd.Series(np.arange(self.S), index=kidx)
        self.d = d
        self.test = test

        def mat(frame, col, fill=np.nan):
            M = np.full((self.S, self.T), fill, dtype=np.float64)
            s = self.sidx.reindex(pd.MultiIndex.from_frame(frame[["sku", "region"]])).to_numpy()
            t = frame["date"].map(MIDX).to_numpy()
            M[s, t] = frame[col].to_numpy(dtype=float)
            return M

        self.D = mat(d, "demand")                         # unconstrained demand (history)
        self.W = mat(d, "train_weight", 0.0)
        self.EXISTS = ~np.isnan(mat(d, "demand"))
        self.ONEOFF = mat(d.assign(o=d["is_one_off"].astype(float)), "o", 0.0) > 0
        self.TEST_EXISTS = ~np.isnan(mat(test.assign(x=1.0), "x"))

        # static attributes (latest known; test overrides customer/channel for the future)
        last = d.sort_values("date").groupby(["sku", "region"]).last()
        tfirst = test.groupby(["sku", "region"]).first()
        st = pd.concat([last[["family", "customer", "channel"]], tfirst[["family", "customer", "channel"]]])
        st = st[~st.index.duplicated(keep="last")].reindex(kidx)
        self.static = pd.DataFrame({
            "sku": keys["sku"].values, "region": keys["region"].values,
            "family": st["family"].values, "customer": st["customer"].values, "channel": st["channel"].values,
            "powertrain": keys["sku"].map(master["Powertrain_Application"]).values,
            "size_class": keys["sku"].map(master["Size_Class"]).values,
        })
        lt = keys["sku"].map(master["Launch_Date"]).map(lambda x: MIDX.get(x, -999 if x < MONTHS[0] else 999))
        self.launch_idx = lt.to_numpy(dtype=float)
        eol = keys["sku"].map(master["EOL_Date"])
        self.eol_idx = eol.map(lambda x: MIDX.get(x, np.nan) if isinstance(x, str) else np.nan).to_numpy(float)
        first = np.where(self.EXISTS.any(1), self.EXISTS.argmax(1), np.nan)
        self.first_idx = first

        # macro (known for the whole window, a scenario path in the future)
        m = pd.read_parquet(C.CLEAN / "macro_clean.parquet").set_index("date").reindex(MONTHS)
        self.macro = m[MACRO_COLS].to_numpy(float)

        # calendar events: holiday fractions by region, price increase by year
        em = pd.read_parquet(C.CLEAN / "event_months.parquet")
        self.hol = {}
        for desc, reg in [("Lunar New Year holiday period", "Asia Pacific"), ("Ramadan", "Middle East & Africa")]:
            f = em[em["Description"] == desc].groupby("date")["frac"].max().reindex(MONTHS).fillna(0).to_numpy()
            self.hol[desc] = (self.static["region"].to_numpy() == reg)[:, None] * f[None, :]
        pc = em[(em["Event_Type"] == "Price Change") & em["Description"].str.contains("Annual")]
        yr_pct = {d_[:4]: p for d_, p in zip(pc["date"], pc["pct"])}
        self.price_pct = np.array([yr_pct.get(m_[:4], 0.0) for m_ in MONTHS])
        self.month_of = np.array([int(m_[5:]) for m_ in MONTHS])
        self.cat_codes = {c: pd.Categorical(self.static[c]) for c in CAT_COLS}


class Fold:
    """Fold-specific derived matrices (cycles, seasonal index, rate) using data <= origin O."""

    def __init__(self, P: Panel, origin: str):
        self.P, self.O = P, MIDX[origin]
        O = self.O
        hist = P.d[P.d["date"] <= origin]
        fut = pd.DataFrame([(k.sku, k.region, m) for k in P.keys.itertuples() for m in MONTHS[O + 1:O + 1 + H]],
                           columns=["sku", "region", "date"]) if O < P.T - 1 else None
        if origin == MONTHS[T_HIST_END]:
            fut = P.test[["sku", "region", "date"]]
        series, segs, flags, futc = detect_cycles(hist.rename(columns={"demand": "units_dem"})
                                                  .assign(units=hist["demand"]), fut)
        self.cycles = series
        OE = np.full((P.S, P.T), np.nan)
        K = np.full((P.S, P.T), np.nan)
        segmask = segs.set_index(["sku", "region", "seg"])["mask"]
        fl = flags.copy()
        fl["mask"] = segmask.reindex(pd.MultiIndex.from_frame(fl[["sku", "region", "cycle_seg"]])).values
        fl["k"] = [p / m.count("1") if m.count("1") else np.nan for m, p in zip(fl["mask"], fl["cycle_period"])]
        s = P.sidx.reindex(pd.MultiIndex.from_frame(fl[["sku", "region"]])).to_numpy()
        t = fl["date"].map(MIDX).to_numpy()
        OE[s, t] = fl["order_expected"].to_numpy()
        K[s, t] = fl["k"].to_numpy()
        if futc is not None:
            cur = series.set_index(["sku", "region"])
            fc = futc.merge(cur[["current_mask", "current_period"]], left_on=["sku", "region"],
                            right_index=True, how="left")
            fc["k"] = [p / m.count("1") if isinstance(m, str) and m.count("1") else 1.0
                       for m, p in zip(fc["current_mask"], fc["current_period"].fillna(1))]
            s = P.sidx.reindex(pd.MultiIndex.from_frame(fc[["sku", "region"]])).to_numpy()
            t = fc["date"].map(MIDX).to_numpy()
            OE[s, t] = np.where(fc["cycle_known"], fc["order_expected"], 1)
            K[s, t] = fc["k"].to_numpy()
        self.OE, self.K = OE, K
        self.months_since_cycle = np.full(P.S, np.nan)
        cs = series.set_index(["sku", "region"])["current_since"].map(MIDX)
        si_ = P.sidx.reindex(cs.index).to_numpy()
        self.months_since_cycle[si_] = O - cs.to_numpy()

        # monthly rate on order months, history only
        R = np.where((OE == 1) & (P.W > 0) & ~np.isnan(P.D), P.D / K, np.nan)
        R[:, O + 1:] = np.nan
        self.R = R
        # seasonal index (family x region x calendar month) from normal months <= O
        normal = R.copy()
        normal[:, COVID[0]:COVID[1] + 1] = np.nan
        ela = (P.static["family"] == "Electrical Assemblies").to_numpy()
        normal[ela, MIDX["2021-03"]:MIDX["2022-09"] + 1] = np.nan
        ma = pd.DataFrame(normal).T.rolling(13, center=True, min_periods=9).mean().T.to_numpy()
        ratio = normal / ma
        rows = pd.DataFrame({"fr": np.repeat(P.static["family"] + "|" + P.static["region"], P.T).values,
                             "m": np.tile(P.month_of, P.S), "r": ratio.ravel()}).dropna()
        si = rows.groupby(["fr", "m"])["r"].median()
        si = si / si.groupby(level=0).transform("mean")
        fr = (P.static["family"] + "|" + P.static["region"]).to_numpy()
        SIm = si.unstack().reindex(pd.Index(fr)).to_numpy()            # S x 12
        SIm = np.where(np.isnan(SIm), 1.0, SIm)
        self.SI = SIm[:, P.month_of - 1]                                 # S x T
        self.DR = R / self.SI                                            # deseasonalised rate

    # ------------------------------------------------------------------ features
    def origin_block(self, o: int, targets: np.ndarray | None = None, need_target=True):
        """Rows for origin o and horizons 1..H. Returns a DataFrame (features + meta)."""
        P = self
        Pp = self.P
        DR, R = self.DR[:, :o + 1], self.R[:, :o + 1]

        def win(a, n):
            return a[:, max(0, o + 1 - n):o + 1]
        scale = np.nanmean(win(DR, 12), 1)
        nobs12 = np.sum(~np.isnan(win(DR, 12)), 1)
        scale = np.where(nobs12 >= 3, scale, np.nan)
        sc = scale[:, None]
        f = {}
        f["log_scale"] = np.log1p(scale)
        for L in range(3):
            f[f"l{L+1}"] = DR[:, o - L] / scale if o - L >= 0 else np.full(Pp.S, np.nan)
        f["m3"] = np.nanmean(win(DR, 3), 1) / scale
        f["m6"] = np.nanmean(win(DR, 6), 1) / scale
        f["m24"] = np.nanmean(win(DR, 24), 1) / scale
        prev = DR[:, max(0, o - 23):max(0, o - 11)]
        f["yoy_growth"] = scale / np.nanmean(prev, 1) if prev.shape[1] else np.full(Pp.S, np.nan)
        w = win(DR, 12)
        x = np.arange(w.shape[1]) - (w.shape[1] - 1) / 2
        ok = ~np.isnan(w)
        num = np.nansum(np.where(ok, (w - np.nanmean(w, 1, keepdims=True)) * x, 0), 1)
        den = np.sum(np.where(ok, x ** 2, 0), 1)
        f["slope12"] = num / np.where(den > 0, den, np.nan) / scale
        f["cv12"] = np.nanstd(w, 1) / scale
        f["nobs12"] = nobs12
        zs = win(Pp.D, 12)
        f["zero_share12"] = np.nanmean(zs == 0, 1)
        f["months_since_cycle"] = self.months_since_cycle if o == self.O else np.full(Pp.S, np.nan)
        f["age"] = o - Pp.first_idx
        f["oneoff_recent"] = win(Pp.ONEOFF.astype(float), 12).sum(1)
        base = pd.DataFrame(f)

        blocks = []
        for h in range(1, H + 1):
            t = o + h
            if t >= Pp.T:
                break
            b = base.copy()
            b["h"] = h
            b["t"] = t
            b["s"] = np.arange(Pp.S)
            b["month"] = Pp.month_of[t]
            b["si_t"] = self.SI[:, t]
            b["si_o"] = self.SI[:, o]
            ly = t - 12 * int(np.ceil(h / 12))
            b["ly_rate"] = R[:, ly] / scale if ly >= 0 else np.nan
            b["ly2_rate"] = R[:, ly - 12] / scale if ly - 12 >= 0 else np.nan
            b["k_t"] = self.K[:, t]
            b["oe_t"] = self.OE[:, t]
            b["lny_frac"] = Pp.hol["Lunar New Year holiday period"][:, t]
            b["ramadan_frac"] = Pp.hol["Ramadan"][:, t]
            b["price_pct_year"] = Pp.price_pct[t]
            b["months_since_launch"] = t - Pp.launch_idx
            b["months_to_eol"] = Pp.eol_idx - t
            b["mea_age"] = np.where(Pp.static["region"] == "Middle East & Africa", t - MIDX["2022-01"], np.nan)
            for j, c in enumerate(MACRO_COLS):
                b[f"mx_{c[:10]}"] = Pp.macro[t, j]
                b[f"md_{c[:10]}"] = Pp.macro[t, j] - Pp.macro[o, j]
            b["mx_PMI_l1"] = Pp.macro[t - 1, 0]
            b["mx_InvDays_l1"] = Pp.macro[t - 1, 2]
            b["scale"] = scale
            if need_target:
                b["y"] = self.R[:, t] / scale if t <= self.O else np.nan
                b["demand"] = Pp.D[:, t]
                b["w_row"] = Pp.W[:, t]
            blocks.append(b)
        out = pd.concat(blocks, ignore_index=True)
        out["o"] = o
        for c in CAT_COLS:
            out[c] = pd.Categorical.from_codes(Pp.cat_codes[c].codes[out["s"].to_numpy()],
                                               categories=Pp.cat_codes[c].categories)
        return out

    def training_set(self, start="2020-06", step=1):
        O = self.O
        rows = []
        for o in range(MIDX[start], O, step):
            b = self.origin_block(o)
            b = b[(b["t"] <= O) & (b["oe_t"] == 1) & b["y"].notna() & b["scale"].notna() & (b["w_row"] > 0)]
            rows.append(b)
        X = pd.concat(rows, ignore_index=True)
        covid = (X["t"] >= COVID[0]) & (X["t"] <= COVID[1])
        X["weight"] = X["scale"] * X["k_t"] * X["w_row"] * np.where(covid, 0.3, 1.0)
        return X

    def prediction_set(self):
        b = self.origin_block(self.O)
        return b
