import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Stockouts & lost sales", layout="wide")
from common import (apply_filters, filter_caption, fmt_units, fmt_usd, history, js, page_header,  # noqa
                    sidebar_filters, timeline)

page_header("Stockouts & lost sales", "Shipments understate demand when supply is short. Listed stockouts use the "
            "requested quantity; unlisted chip-shortage months are estimated (validated at 11 % WAPE).")
sel = sidebar_filters()
st.caption(filter_caption(sel))
h = apply_filters(history(), sel)
sup = h[h["demand_source"].str.startswith("stockout")]
lost_rev = (h["lost_units_est"].fillna(0) * h["price"]).sum()
c = st.columns(4)
c[0].metric("Stockout months (listed + inferred)", f"{len(sup):,}")
c[1].metric("Estimated lost units", fmt_units(h["lost_units_est"].sum()))
c[2].metric("Estimated lost revenue", fmt_usd(lost_rev))
c[3].metric("Demand above shipments", f"{h['demand'].sum() / max(h['shipped'].sum(), 1) - 1:+.2%}")
st.plotly_chart(timeline(h, None, "units", "Shipped vs unconstrained demand"), use_container_width=True)
a, b = st.columns(2)
r = h.assign(reason=h["st_reason"].fillna(h["demand_source"].where(h["demand_source"].str.contains("chip"))))
r = r[r["lost_units_est"] > 0]
lr = r.groupby("reason")["lost_units_est"].sum().reset_index().sort_values("lost_units_est")
f1 = px.bar(lr, x="lost_units_est", y="reason", orientation="h", title="Lost units by reason")
f1.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
a.plotly_chart(f1, use_container_width=True)
lf = r.assign(rev=r["lost_units_est"] * r["price"]).groupby("family", observed=True)["rev"].sum().reset_index()
f2 = px.bar(lf.sort_values("rev"), x="rev", y="family", orientation="h", title="Lost revenue by family (USD)")
f2.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
b.plotly_chart(f2, use_container_width=True)
fr = h.groupby("date")[["shipped", "demand"]].sum()
fr["fill_rate"] = fr["shipped"] / fr["demand"]
f3 = px.line(fr.reset_index(), x="date", y="fill_rate", title="Fill rate (shipped / unconstrained demand)")
f3.update_layout(height=320, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(f3, use_container_width=True)
s = js("phase2_summary.json")["validation"]
st.markdown(f"**Method check:** the unlisted chip-shortage estimator was cross-validated on listed stockouts "
            f"(truth known): WAPE {s['chip_estimator_wape']['A_shipped_over_fillrate']:.1%} "
            f"vs {s['chip_estimator_wape']['B_expected_x_index']:.1%} for the alternative. Isolated unlisted "
            "stockouts were *not* imputed: the best detector reached only ~7.5 % precision.")
st.dataframe(sup[["date", "sku", "region", "customer", "shipped", "demand", "lost_units_est", "st_reason",
                  "demand_source"]].sort_values("lost_units_est", ascending=False).head(2000),
             hide_index=True, use_container_width=True)
