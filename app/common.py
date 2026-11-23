"""Shared data loading, filters and charts for the Celeris Planning Control Tower.

Every number shown comes from pipeline outputs (outputs/, data/clean/). Nothing is typed in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CLEAN = ROOT / "data" / "clean"
FORECAST_START = "2026-11"
LAST_HIST = "2026-10"

FAMILY_COLORS = {
    "Brake & Safety Systems": "#4C78A8", "Chassis & Structural": "#F58518", "Electrical Assemblies": "#54A24B",
    "Hydraulic Systems": "#E45756", "Powertrain Components": "#72B7B2", "Precision Machined Parts": "#B279A2",
    "Thermal Management": "#9D755D",
}
HIST_COLOR, DEMAND_COLOR, FC_COLOR, BAND_COLOR = "#7f8c8d", "#1f4e79", "#d35400", "rgba(211,84,0,0.18)"
FILTERS = ["family", "sku", "region", "channel", "customer"]


# ------------------------------------------------------------------ loaders
@st.cache_data(show_spinner=False)
def history() -> pd.DataFrame:
    d = pd.read_parquet(CLEAN / "demand.parquet")
    d["revenue_shipped"] = d["shipped"].fillna(0) * d["price"]
    d["revenue_demand"] = d["demand"] * d["price"]
    for c in ["family", "region", "channel", "customer", "sku", "bu"]:
        d[c] = d[c].astype("category")
    return d


@st.cache_data(show_spinner=False)
def forecast() -> pd.DataFrame:
    f = pd.read_parquet(OUT / "forecast.parquet")
    for c in ["family", "region", "channel", "customer", "sku", "bu"]:
        f[c] = f[c].astype("category")
    return f


@st.cache_data(show_spinner=False)
def master() -> pd.DataFrame:
    return pd.read_parquet(CLEAN / "sku_master.parquet")


@st.cache_data(show_spinner=False)
def csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(OUT / name, **kw)


@st.cache_data(show_spinner=False)
def parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / name)


@st.cache_data(show_spinner=False)
def js(name: str) -> dict:
    return json.loads((OUT / name).read_text())


@st.cache_data(show_spinner=False)
def band_lambda() -> float:
    """Calibrate how row-level P10/P90 intervals combine when summed over many series.
    half-width = (1-l) * sqrt(sum d^2) + l * sum d  (l=0: independent errors, l=1: fully correlated).
    l is chosen so that family x month aggregates in the backtest have ~80 % coverage."""
    bt = pd.read_parquet(OUT / "backtest_long.parquet")
    bt["d_hi"] = (bt["p90"] - bt["ensemble"]).clip(lower=0)
    bt["d_lo"] = (bt["ensemble"] - bt["p10"]).clip(lower=0)
    g = bt.groupby(["origin", "date", "family"]).agg(a=("actual", "sum"), p=("ensemble", "sum"),
                                                      hs=("d_hi", "sum"), ls=("d_lo", "sum"),
                                                      hq=("d_hi", lambda x: np.sqrt((x ** 2).sum())),
                                                      lq=("d_lo", lambda x: np.sqrt((x ** 2).sum())))
    best = (1.0, 0.0)
    for lam in np.linspace(0, 1, 21):
        hi = g["p"] + (1 - lam) * g["hq"] + lam * g["hs"]
        lo = g["p"] - (1 - lam) * g["lq"] - lam * g["ls"]
        cov = ((g["a"] >= lo) & (g["a"] <= hi)).mean()
        if abs(cov - 0.8) < best[0]:
            best = (abs(cov - 0.8), lam)
    return float(best[1])


def aggregate_forecast(f: pd.DataFrame, by: list[str], value="p50") -> pd.DataFrame:
    """Sum p50 and combine intervals with the calibrated lambda."""
    lam = band_lambda()
    x = f.assign(dh=(f["p90"] - f["p50"]).clip(lower=0), dl=(f["p50"] - f["p10"]).clip(lower=0))
    x["dh2"], x["dl2"] = x["dh"] ** 2, x["dl"] ** 2
    g = x.groupby(by, observed=True).agg(p50=("p50", "sum"), dh=("dh", "sum"), dl=("dl", "sum"),
                                         dh2=("dh2", "sum"), dl2=("dl2", "sum"),
                                         revenue=("revenue_p50", "sum")).reset_index()
    g["p90"] = g["p50"] + (1 - lam) * np.sqrt(g["dh2"]) + lam * g["dh"]
    g["p10"] = (g["p50"] - (1 - lam) * np.sqrt(g["dl2"]) - lam * g["dl"]).clip(lower=0)
    rr = np.divide(g["revenue"], g["p50"], out=np.zeros(len(g)), where=g["p50"] > 0)
    g["revenue_p10"], g["revenue_p90"] = g["p10"] * rr, g["p90"] * rr
    return g.drop(columns=["dh", "dl", "dh2", "dl2"])


# ------------------------------------------------------------------ shared filters
def sidebar_filters() -> dict:
    """Five filters shared by every page. Values persist across pages via session_state."""
    h = history()
    st.sidebar.markdown("### Filters")
    st.sidebar.caption("Shared across all pages")
    sel = {}
    opts = {
        "family": sorted(h["family"].cat.categories),
        "region": sorted(h["region"].cat.categories),
        "channel": sorted(h["channel"].cat.categories),
        "customer": sorted(h["customer"].cat.categories),
    }
    for f in ["family", "region", "channel", "customer"]:
        sel[f] = _persisted_multiselect(f.capitalize(), f, opts[f])
    skus = h["sku"].cat.categories
    if sel["family"]:
        skus = h.loc[h["family"].isin(sel["family"]), "sku"].unique()
    fc = forecast()
    extra = fc.loc[~fc["sku"].isin(h["sku"].cat.categories), "sku"].unique()
    sel["sku"] = _persisted_multiselect("SKU", "sku", sorted(set(map(str, skus)) | set(map(str, extra))))
    if st.sidebar.button("Clear filters"):
        for f in FILTERS:
            st.session_state[f"_f_{f}"] = []
            st.session_state.pop(f"w_{f}", None)
        st.rerun()
    return sel


def _persisted_multiselect(label, name, options):
    store = f"_f_{name}"
    default = [v for v in st.session_state.get(store, []) if v in options]
    val = st.sidebar.multiselect(label, options, default=default, key=f"w_{name}")
    st.session_state[store] = val
    return val


def apply_filters(df: pd.DataFrame, sel: dict) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for f, v in sel.items():
        if v and f in df:
            m &= df[f].astype(str).isin(v)
    return df[m]


def filter_caption(sel: dict) -> str:
    parts = [f"{k}: {', '.join(v[:3])}{'…' if len(v) > 3 else ''}" for k, v in sel.items() if v]
    return "Filters: " + (" · ".join(parts) if parts else "none (all data)")


# ------------------------------------------------------------------ charts
def timeline(hist: pd.DataFrame, fc: pd.DataFrame | None, value="units", title="", show_shipped=True,
             height=420) -> go.Figure:
    """One continuous timeline: history (shipped + unconstrained demand) then forecast P50 with P10-P90."""
    fig = go.Figure()
    if value == "units":
        hcols, fcols = ("shipped", "demand"), ("p50", "p10", "p90")
    else:
        hcols, fcols = ("revenue_shipped", "revenue_demand"), ("revenue", "revenue_p10", "revenue_p90")
    hh = hist.groupby("date", observed=True)[list(hcols)].sum().reset_index()
    if show_shipped:
        fig.add_trace(go.Scatter(x=hh["date"], y=hh[hcols[0]], name="Shipped (history)",
                                 line=dict(color=HIST_COLOR, width=1.5)))
    fig.add_trace(go.Scatter(x=hh["date"], y=hh[hcols[1]], name="Unconstrained demand (history)",
                             line=dict(color=DEMAND_COLOR, width=2.2)))
    if fc is not None and len(fc):
        g = aggregate_forecast(fc, ["date"])
        # bridge the last history point to the first forecast point
        last = hh.iloc[-1]
        x = [last["date"]] + g["date"].tolist()
        p50 = [last[hcols[1]]] + g[fcols[0]].tolist()
        p10 = [last[hcols[1]]] + g[fcols[1]].tolist()
        p90 = [last[hcols[1]]] + g[fcols[2]].tolist()
        fig.add_trace(go.Scatter(x=x + x[::-1], y=p90 + p10[::-1], fill="toself", fillcolor=BAND_COLOR,
                                 line=dict(width=0), name="Forecast P10–P90", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=p50, name="Forecast P50", line=dict(color=FC_COLOR, width=2.4, dash="dash")))
    fig.add_vline(x=LAST_HIST, line=dict(color="#555", dash="dot"))
    fig.add_annotation(x=FORECAST_START, y=1.02, yref="paper", text="Forecast →", showarrow=False,
                       xanchor="left", font=dict(color=FC_COLOR))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15), yaxis_title="Units" if value == "units" else "USD")
    return fig


def kpi(col, label, value, help_=None, delta=None):
    col.metric(label, value, delta=delta, help=help_)


def fmt_units(x):
    return f"{x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"{x / 1e3:,.1f}k" if abs(x) >= 1e4 else f"{x:,.0f}"


def fmt_usd(x):
    return f"${x / 1e9:,.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:,.1f}M" if abs(x) >= 1e6 else f"${x:,.0f}"


def page_header(title: str, purpose: str):
    st.title(title)
    st.caption(purpose)
