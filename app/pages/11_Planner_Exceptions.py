import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Planner exceptions", layout="wide")
from common import (apply_filters, csv, filter_caption, forecast, history, page_header, parquet,  # noqa: E402
                    sidebar_filters)

page_header("Exception list for planners", "Series that deserve a human look, ranked by revenue at stake. "
            "Each row says why it was flagged.")
sel = sidebar_filters()
st.caption(filter_caption(sel))


@st.cache_data(show_spinner=False)
def exceptions() -> pd.DataFrame:
    h, f = history(), forecast()
    last12 = h[h["date"] > "2025-10"].groupby(["sku", "region"], observed=True).agg(
        hist12=("demand", "sum"), price=("price", "last"))
    f12 = f[f["date"] <= "2027-10"].groupby(["sku", "region"], observed=True).agg(
        fc12=("p50", "sum"), p10=("p10", "sum"), p90=("p90", "sum"), rev=("revenue_p50", "sum"),
        family=("family", "first"), customer=("customer", "first"), channel=("channel", "first"),
        adjustment=("adjustment", lambda s: ";".join(sorted(set(";".join(s).split(";")) - {""}))))
    e = f12.join(last12, how="left")
    e["change"] = e["fc12"] / e["hist12"] - 1
    e["band_width"] = (e["p90"] - e["p10"]) / e["fc12"].replace(0, np.nan)
    bt = parquet("backtest_long.parquet")
    err = bt.groupby(["sku", "region"]).apply(lambda g: (g["ensemble"] - g["actual"]).abs().sum() /
                                              max(g["actual"].sum(), 1), include_groups=False).rename("backtest_wape")
    e = e.join(err)
    cc = csv("customer_changes.csv")
    lost = set(zip(cc.loc[cc["type"] == "lost_account", "sku"], cc.loc[cc["type"] == "lost_account", "region"]))
    sw = set(zip(cc.loc[cc["type"] == "customer_switch", "sku"], cc.loc[cc["type"] == "customer_switch", "region"]))
    cyc = csv("order_cycles.csv").set_index(["sku", "region"])
    e = e.join(cyc[["current_cycle", "current_since"]])
    reasons = []
    for (sku, reg), r in e.iterrows():
        why = []
        if (sku, reg) in lost:
            why.append("lost account: no orders since " + str(r["current_since"]))
        if pd.isna(r["hist12"]) or r["hist12"] == 0:
            why.append("new series / no recent history")
        elif abs(r["change"]) > 0.3:
            why.append(f"forecast {r['change']:+.0%} vs last 12 months")
        if r["band_width"] > 1.2:
            why.append("wide P10–P90 band")
        if r["backtest_wape"] > 0.6:
            why.append(f"hard to forecast (backtest WAPE {r['backtest_wape']:.0%})")
        if (sku, reg) in sw:
            why.append("customer switched in history")
        if str(r["current_since"]) >= "2025-11" and r["current_cycle"] != "no orders":
            why.append(f"ordering cycle changed recently ({r['current_cycle']})")
        if r["adjustment"]:
            why.append("event / lifecycle adjustment: " + r["adjustment"])
        reasons.append("; ".join(why))
    e["reasons"] = reasons
    e = e[e["reasons"] != ""].reset_index()
    e["revenue_at_stake"] = (e["p90"] - e["p10"]) * e["price"].fillna(e["rev"] / e["fc12"].replace(0, np.nan))
    return e.sort_values("revenue_at_stake", ascending=False)


e = apply_filters(exceptions(), sel)
st.metric("Series flagged", f"{len(e):,}")
kinds = ["lost account", "new series", "forecast", "wide", "hard to forecast", "customer switched", "cycle changed",
         "adjustment"]
pick = st.multiselect("Reason contains", kinds)
if pick:
    e = e[e["reasons"].apply(lambda s: any(k in s for k in pick))]
st.dataframe(e[["sku", "region", "family", "customer", "channel", "hist12", "fc12", "change", "p10", "p90",
                "backtest_wape", "current_cycle", "revenue_at_stake", "reasons"]]
             .style.format({"hist12": "{:,.0f}", "fc12": "{:,.0f}", "change": "{:+.0%}", "p10": "{:,.0f}",
                            "p90": "{:,.0f}", "backtest_wape": "{:.0%}", "revenue_at_stake": "${:,.0f}"}),
             hide_index=True, use_container_width=True, height=600)
st.download_button("Download exceptions (CSV)", e.to_csv(index=False), "planner_exceptions.csv")
