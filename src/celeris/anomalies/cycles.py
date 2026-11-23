"""Ordering-cycle detection (Expert WS2 / rule 6).

Customers order on a deterministic pattern: every month, every 2nd month, 2 of every 3 months,
quarterly, monthly with one skipped month a year, ... A cycle is (period p, mask) where
mask[k] = 1 if an order is expected when month_index % p == k. Missing months are unknown,
not zero.

Segmentation: penalised binary segmentation. The cost of a segment is the number of months the
best (simplest adequate) pattern gets wrong; a split is accepted only if it removes more
mismatches than the penalty. Costs are O(1) via cumulative counts per residue class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = (1, 2, 3, 4, 6, 12)
MIN_SEG = 6
PENALTY = 2.5
T0 = pd.Period("2019-01", freq="M")


class _Counts:
    """Cumulative counts of observed ones / zeros for each (period, residue)."""

    def __init__(self, o: np.ndarray, idx: np.ndarray):
        self.n = len(o)
        obs = ~np.isnan(o)
        one = (o == 1) & obs
        zero = (o == 0) & obs
        self.c = {}
        for p in PERIODS:
            r = idx % p
            ones = np.zeros((p, self.n + 1))
            zeros = np.zeros((p, self.n + 1))
            for k in range(p):
                ones[k, 1:] = np.cumsum(one & (r == k))
                zeros[k, 1:] = np.cumsum(zero & (r == k))
            self.c[p] = (ones, zeros)
        self.nobs = np.concatenate([[0], np.cumsum(obs)])

    def best(self, a: int, b: int):
        """Best pattern on [a, b): returns (mismatches, period, mask)."""
        nobs = self.nobs[b] - self.nobs[a]
        if nobs == 0:
            return 0.0, 1, (1,)
        cands = []
        for p in PERIODS:
            if p == 12 and nobs < 18:          # an annual pattern needs > 1 year of evidence
                continue
            if p > 1 and nobs < 2 * p:
                continue
            ones, zeros = self.c[p]
            o = ones[:, b] - ones[:, a]
            z = zeros[:, b] - zeros[:, a]
            mask = tuple(int(oi >= zi) if (oi + zi) > 0 else 1 for oi, zi in zip(o, z))
            mism = float(np.minimum(o, z).sum())
            cands.append((mism, p, mask))
        # simplest pattern within 5 % of the best accuracy (avoid over-fitting long periods)
        bestm = min(c[0] for c in cands)
        for m, p, mask in cands:
            if m <= bestm + 0.05 * nobs * (p > 1) + 1e-9 and m <= bestm + max(1.0, 0.03 * nobs):
                return m, p, mask
        return min(cands)


def _binseg(cnt: _Counts, a: int, b: int, out: list):
    m0, p0, k0 = cnt.best(a, b)
    best = None
    if m0 > 0 and b - a >= 2 * MIN_SEG:
        for s in range(a + MIN_SEG, b - MIN_SEG + 1):
            ml = cnt.best(a, s)[0]
            mr = cnt.best(s, b)[0]
            gain = m0 - (ml + mr)
            if best is None or gain > best[0]:
                best = (gain, s)
    if best is not None and best[0] > PENALTY:
        _binseg(cnt, a, best[1], out)
        _binseg(cnt, best[1], b, out)
    else:
        out.append((a, b, m0, p0, k0))


def segment_series(o: np.ndarray, idx: np.ndarray) -> list[dict]:
    cnt = _Counts(o, idx)
    out = []
    _binseg(cnt, 0, len(o), out)
    segs = []
    for a, b, m, p, k in sorted(out):
        nobs = cnt.nobs[b] - cnt.nobs[a]
        if segs and segs[-1]["period"] == p and segs[-1]["mask"] == k:
            segs[-1]["end"] = b
            continue
        segs.append(dict(start=a, end=b, period=p, mask=k, acc=1 - m / max(nobs, 1)))
    return segs


def describe(p: int, mask: tuple) -> str:
    n = sum(mask)
    if n == 0:
        return "no orders"
    if n == p:
        return "monthly"
    names = {(12, 11): "monthly, 1 skip/yr", (2, 1): "every 2nd month", (3, 1): "quarterly",
             (3, 2): "2 of 3 months", (4, 1): "every 4th month", (6, 1): "half-yearly",
             (6, 5): "5 of 6 months", (12, 10): "monthly, 2 skips/yr"}
    return names.get((p, n), f"{n} of {p} months")


def month_index(dates: pd.Series) -> np.ndarray:
    return (pd.PeriodIndex(dates, freq="M") - T0).map(lambda x: x.n).to_numpy()


def detect_cycles(sales: pd.DataFrame, test: pd.DataFrame | None = None):
    """Returns (series summary, segments, per-row train flags, per-row test flags)."""
    seg_rows, ser_rows, flags = [], [], []
    for (sku, reg), g in sales.groupby(["sku", "region"], sort=False):
        g = g.sort_values("date")
        idx = month_index(g["date"])
        o = np.where(g["units"].isna(), np.nan, (g["units"] > 0).astype(float))
        segs = segment_series(o, idx)
        for j, s in enumerate(segs):
            seg_rows.append(dict(sku=sku, region=reg, seg=j, start=g["date"].iloc[s["start"]],
                                 end=g["date"].iloc[s["end"] - 1], months=s["end"] - s["start"],
                                 period=s["period"], mask="".join(map(str, s["mask"])),
                                 cycle=describe(s["period"], s["mask"]), fit=round(s["acc"], 3)))
            exp = np.array([s["mask"][i % s["period"]] for i in idx[s["start"]:s["end"]]])
            flags.append(pd.DataFrame({"sku": sku, "region": reg,
                                       "date": g["date"].iloc[s["start"]:s["end"]].values,
                                       "order_expected": exp, "cycle_period": s["period"], "cycle_seg": j}))
        last = segs[-1]
        ser_rows.append(dict(sku=sku, region=reg, n_segments=len(segs),
                             current_cycle=describe(last["period"], last["mask"]),
                             current_period=last["period"], current_mask="".join(map(str, last["mask"])),
                             current_since=g["date"].iloc[last["start"]], current_fit=round(last["acc"], 3),
                             zero_share=float(np.nanmean(o == 0))))
    segs = pd.DataFrame(seg_rows)
    series = pd.DataFrame(ser_rows)
    flags = pd.concat(flags, ignore_index=True)
    fut = None
    if test is not None:
        cur = series.set_index(["sku", "region"])
        tt = test[["date", "sku", "region"]].copy()
        i = month_index(tt["date"])
        c = cur.reindex(pd.MultiIndex.from_frame(tt[["sku", "region"]]))
        per = c["current_period"].fillna(1).astype(int).to_numpy()
        msk = c["current_mask"].fillna("1").to_numpy()
        tt["order_expected"] = [int(m[k % p]) for m, k, p in zip(msk, i, per)]
        tt["cycle_period"] = per
        tt["cycle_known"] = c["current_period"].notna().to_numpy()
        fut = tt
    return series, segs, flags, fut
