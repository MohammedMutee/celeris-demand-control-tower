import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Lifecycle", layout="wide")
from common import csv, forecast, history, master, page_header, sidebar_filters  # noqa: E402

page_header("Lifecycle: launches, EOLs and the EV shift", "New products take over from the SKU they replace. "
            "The handover curve is learned from 6 past transitions and applied to the 3 launches in the forecast window.")
sel = sidebar_filters()
m = master()
lc = m[(m["Lifecycle_Status"] != "Active") | m["Replaces_SKU"].notna()].copy()
lc["start"] = pd.to_datetime(lc["Launch_Date"].where(lc["Launch_Date"] > "2018-01", "2018-01"))
lc["end"] = pd.to_datetime(lc["EOL_Date"].fillna("2028-05"))
fig = px.timeline(lc, x_start="start", x_end="end", y="SKU_ID", color="Lifecycle_Status",
                  hover_data=["Replaces_SKU", "Powertrain_Application", "Launch_Date", "EOL_Date"],
                  title="Lifecycle of SKUs with launches / EOLs / replacements")
fig.add_vline(x=pd.Timestamp("2026-11-01"), line_dash="dot")
fig.update_layout(height=480, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Handover curve (learned from history)")
cur = csv("handover_curve.csv", index_col=0)
f2 = go.Figure()
f2.add_trace(go.Scatter(x=cur.index, y=cur["new"], name="Successor (share of predecessor's rate)"))
f2.add_trace(go.Scatter(x=cur.index, y=cur["old"], name="Predecessor during overlap"))
f2.update_layout(height=320, xaxis_title="Months since successor launch", yaxis_tickformat=".0%",
                 margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(f2, use_container_width=True)

st.subheader("Predecessor → successor: history and forecast")
pair = st.selectbox("Transition", [f"{r.Replaces_SKU} → {r.SKU_ID}" for r in m[m["Replaces_SKU"].notna()].itertuples()])
old, new = [x.strip() for x in pair.split("→")]
h, f = history(), forecast()
rows = []
for sku, role in [(old, "predecessor"), (new, "successor")]:
    hh = h[h["sku"] == sku].groupby("date")["demand"].sum()
    ff = f[f["sku"] == sku].groupby("date")[["p50", "p10", "p90"]].sum()
    rows.append(pd.DataFrame({"date": hh.index, "units": hh.values, "series": f"{sku} history"}))
    rows.append(pd.DataFrame({"date": ff.index, "units": ff["p50"].values, "series": f"{sku} forecast"}))
x = pd.concat(rows)
f3 = px.line(x, x="date", y="units", color="series", title=pair)
f3.add_vline(x="2026-10", line_dash="dot")
f3.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(f3, use_container_width=True)
st.dataframe(csv("lifecycle_audit.csv"), hide_index=True, use_container_width=True)

st.subheader("The shift to electric vehicles")
ev = csv("ev_shift.csv")
f4 = px.area(ev, x="year", y=["share_ICE", "share_EV", "share_Universal"], title="Demand mix by powertrain")
f4.update_layout(height=340, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(f4, use_container_width=True)
pt = m.set_index("SKU_ID")["Powertrain_Application"]
fm = f.assign(pt=f["sku"].astype(str).map(pt), year=f["date"].str[:4]).groupby(["year", "pt"])["p50"].sum().unstack()
st.caption("Forecast mix by powertrain")
st.dataframe((fm.div(fm.sum(1), axis=0)).style.format("{:.1%}"), use_container_width=True)
st.dataframe(csv("ev_elasticity.csv"), hide_index=True)
