import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Executive overview", layout="wide")
from common import (FAMILY_COLORS, aggregate_forecast, apply_filters, filter_caption, fmt_units, fmt_usd,  # noqa
                    forecast, history, page_header, sidebar_filters, timeline)

page_header("Executive overview", "History flows into the forecast on one timeline. Forecast = unconstrained "
            "demand (what customers will request), P10–P90 band shaded.")
sel = sidebar_filters()
st.caption(filter_caption(sel))
h, f = apply_filters(history(), sel), apply_filters(forecast(), sel)
if h.empty and f.empty:
    st.warning("No data for this filter selection.")
    st.stop()

last12 = h[h["date"] > "2025-10"]
f12 = f[f["date"] <= "2027-10"]
tot = aggregate_forecast(f.assign(k=1), ["k"])
c = st.columns(5)
c[0].metric("Forecast units, 19 months (P50)", fmt_units(f["p50"].sum()),
            help=f"P10–P90: {fmt_units(tot['p10'].iloc[0])} – {fmt_units(tot['p90'].iloc[0])}")
c[1].metric("Forecast revenue, 19 months", fmt_usd(f["revenue_p50"].sum()))
yoy = f12["p50"].sum() / max(last12["demand"].sum(), 1) - 1
c[2].metric("Next 12 mo vs last 12 mo", f"{yoy:+.1%}", help="Forecast Nov-26..Oct-27 vs demand Nov-25..Oct-26")
c[3].metric("Last 12 mo demand", fmt_units(last12["demand"].sum()))
c[4].metric("Last 12 mo revenue", fmt_usd(last12["revenue_demand"].sum()))

t1, t2 = st.tabs(["Units", "Revenue"])
with t1:
    st.plotly_chart(timeline(h, f, "units", "Units: history → forecast"), use_container_width=True)
with t2:
    st.plotly_chart(timeline(h, f, "revenue", "Revenue (USD): history → forecast"), use_container_width=True)

st.subheader("Mix")
dim = st.radio("Split by", ["family", "region", "channel"], horizontal=True)
a, b = st.columns(2)
hy = h.assign(year=h["date"].str[:4]).groupby(["year", dim], observed=True)["demand"].sum().reset_index()
fy = f.assign(year=f["date"].str[:4] + " (fc)").groupby(["year", dim], observed=True)["p50"].sum().reset_index()
mix = __import__("pandas").concat([hy.rename(columns={"demand": "units"}), fy.rename(columns={"p50": "units"})])
fig = px.bar(mix, x="year", y="units", color=dim, title=f"Units by year and {dim}",
             color_discrete_map=FAMILY_COLORS if dim == "family" else None)
fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
a.plotly_chart(fig, use_container_width=True)
share = f.groupby(dim, observed=True)["revenue_p50"].sum().reset_index()
fig2 = px.pie(share, names=dim, values="revenue_p50", title=f"19-month forecast revenue by {dim}", hole=0.45,
              color=dim, color_discrete_map=FAMILY_COLORS if dim == "family" else None)
fig2.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
b.plotly_chart(fig2, use_container_width=True)

st.subheader("Headline 19-month forecast")
g = aggregate_forecast(f, [dim]).sort_values("p50", ascending=False)
hl = h[h["date"] > "2025-04"].groupby(dim, observed=True)["demand"].sum() * 19 / 18
hl.index = hl.index.astype(str)
g[dim] = g[dim].astype(str)
g["history_run_rate_19m"] = g[dim].map(hl).astype(float)
g["change_vs_run_rate"] = g["p50"] / g["history_run_rate_19m"] - 1
st.dataframe(g[[dim, "p10", "p50", "p90", "revenue", "history_run_rate_19m", "change_vs_run_rate"]]
             .style.format({"p10": "{:,.0f}", "p50": "{:,.0f}", "p90": "{:,.0f}", "revenue": "${:,.0f}",
                            "history_run_rate_19m": "{:,.0f}", "change_vs_run_rate": "{:+.1%}"}),
             use_container_width=True, hide_index=True)
