import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Anomaly monitor", layout="wide")
from common import OUT, apply_filters, csv, filter_caption, history, page_header, sidebar_filters  # noqa: E402

page_header("Anomaly monitor", "Every flagged point with its type, reason and action. 'error' = a data mistake "
            "we fixed; 'real_*' = genuine demand behaviour we modelled, corrected (stockouts) or excluded.")
sel = sidebar_filters()
st.caption(filter_caption(sel))


@st.cache_data(show_spinner=False)
def register():
    return pd.read_csv(OUT / "anomaly_register.csv", low_memory=False)


reg = apply_filters(register(), sel)
cls = st.multiselect("Classification", sorted(reg["classification"].unique()),
                     default=[c for c in reg["classification"].unique() if c != "real_supply"])
reg = reg[reg["classification"].isin(cls)] if cls else reg
c = st.columns(4)
c[0].metric("Flags in view", f"{len(reg):,}")
c[1].metric("Data errors fixed", f"{(reg['classification'] == 'error').sum():,}")
c[2].metric("Real events", f"{reg['classification'].str.startswith('real').sum():,}")
c[3].metric("Series affected", f"{reg[['sku', 'region']].drop_duplicates().shape[0]:,}")

h = apply_filters(history(), sel)
ts = h.groupby("date")[["shipped", "demand"]].sum().reset_index()
pts = reg.groupby(["date", "type"]).size().rename("n").reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts["date"], y=ts["demand"], name="Unconstrained demand", line=dict(color="#1f4e79")))
fig.add_trace(go.Scatter(x=ts["date"], y=ts["shipped"], name="Shipped", line=dict(color="#7f8c8d", width=1)))
for t, g in pts.groupby("type"):
    y = ts.set_index("date")["demand"].reindex(g["date"]).values
    fig.add_trace(go.Scatter(x=g["date"], y=y, mode="markers", name=t, marker=dict(size=4 + 10 * (g["n"] / pts["n"].max()) ** 0.5),
                             text=g["n"], hovertemplate="%{x}: %{text} flags<extra>" + t + "</extra>"))
fig.update_layout(height=440, title="Flags on the time series (marker size = number of flags)",
                  margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

a, b = st.columns(2)
cnt = reg.groupby(["classification", "type"]).size().rename("n").reset_index()
f2 = px.bar(cnt, x="n", y="type", color="classification", orientation="h", title="Flags by type", log_x=True)
f2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
a.plotly_chart(f2, use_container_width=True)
act = reg.groupby("action").size().rename("n").reset_index().sort_values("n")
f3 = px.bar(act, x="n", y="action", orientation="h", title="Actions taken")
f3.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
b.plotly_chart(f3, use_container_width=True)

st.subheader("Anomaly register")
st.dataframe(reg[["anomaly_id", "date", "sku", "region", "customer", "type", "classification", "raw_value",
                  "clean_value", "demand_value", "expected_value", "score", "linked_event_id", "reason", "action"]]
             .head(5000), hide_index=True, use_container_width=True, height=420)

st.subheader("Calendar events: measured effect")
ev = csv("event_effects.csv")
st.dataframe(ev[["Event_ID", "Start_Date", "Scope_Value", "Event_Type", "Description", "method", "effect_pct",
                 "z", "next_month_effect_pct", "verdict"]], hide_index=True, use_container_width=True, height=360)
x, y = st.columns(2)
x.markdown("**Holidays (regression, per full month of holiday)**")
x.dataframe(csv("holiday_effects.csv"), hide_index=True, use_container_width=True)
y.markdown("**Pre-buying before January price increases (Nov–Dec uplift vs normal years)**")
pb = csv("prebuy_effects.csv")
y.dataframe(pb.pivot(index="channel", columns="year", values="novdec_uplift_pct"), use_container_width=True)
st.markdown("**Customer switches and lost accounts**")
st.dataframe(apply_filters(csv("customer_changes.csv"), {k: v for k, v in sel.items() if k in ("sku", "region")}),
             hide_index=True, use_container_width=True)
