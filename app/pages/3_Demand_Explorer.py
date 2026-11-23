import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Demand explorer", layout="wide")
from common import (FAMILY_COLORS, apply_filters, csv, filter_caption, history, master, page_header,  # noqa
                    sidebar_filters, timeline)

page_header("Demand explorer", "Drill down family → SKU → region → customer. Use the sidebar filters; the "
            "breakdown table below lets you pick the next level.")
sel = sidebar_filters()
st.caption(filter_caption(sel))
h = apply_filters(history(), sel)
if h.empty:
    st.warning("No data for this filter selection.")
    st.stop()

level = next((lv for lv in ["family", "sku", "region", "customer"] if not sel.get(lv)), "customer")
st.plotly_chart(timeline(h, None, "units", "Shipped vs unconstrained demand"), use_container_width=True)

st.subheader(f"Breakdown by {level}")
b = h.groupby(level, observed=True).agg(demand=("demand", "sum"), shipped=("shipped", "sum"),
                                        revenue=("revenue_demand", "sum"),
                                        zero_months=("demand", lambda s: int((s == 0).sum())),
                                        months=("demand", "size")).reset_index()
b = b[b["months"] > 0].sort_values("demand", ascending=False)
b["share"] = b["demand"] / b["demand"].sum()
st.dataframe(b.style.format({"demand": "{:,.0f}", "shipped": "{:,.0f}", "revenue": "${:,.0f}", "share": "{:.1%}"}),
             hide_index=True, use_container_width=True, height=260)
st.caption("Tip: add a value from this table to the matching sidebar filter to drill one level down.")

a, c = st.columns(2)
s = h.assign(year=h["date"].str[:4], month=h["Month"]).groupby(["year", "month"], observed=True)["demand"].sum()
heat = s.unstack()
fig = px.imshow(heat / heat.mean(axis=1).values[:, None], aspect="auto", color_continuous_scale="RdBu",
                zmin=0.4, zmax=1.6, labels=dict(color="vs year avg"), title="Seasonality: month vs yearly average")
fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
a.plotly_chart(fig, use_container_width=True)
z = h.assign(zero=(h["demand"] == 0)).groupby("date")["zero"].mean().reset_index()
fig2 = px.bar(z, x="date", y="zero", title="Share of series-months with zero orders (ordering cycles)")
fig2.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis_tickformat=".0%")
c.plotly_chart(fig2, use_container_width=True)

st.subheader("Ordering cycles and lifecycle")
cyc = csv("order_cycles.csv")
keys = h[["sku", "region"]].drop_duplicates().astype(str)
cy = cyc.merge(keys, on=["sku", "region"])
x, y = st.columns(2)
fig3 = px.bar(cy["current_cycle"].value_counts().reset_index(), x="current_cycle", y="count",
              title="Current ordering cycle of the selected series")
fig3.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
x.plotly_chart(fig3, use_container_width=True)
m = master()
mm = m[m["SKU_ID"].isin(keys["sku"])][["SKU_ID", "Lifecycle_Status", "Powertrain_Application", "Launch_Date",
                                        "EOL_Date", "Replaces_SKU"]]
y.dataframe(mm.sort_values("Lifecycle_Status"), hide_index=True, use_container_width=True, height=320)
if len(keys) <= 12:
    st.subheader("Series detail")
    seg = csv("order_cycle_segments.csv").merge(keys, on=["sku", "region"])
    st.dataframe(seg, hide_index=True, use_container_width=True)
