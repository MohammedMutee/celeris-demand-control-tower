import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Macro lead/lag", layout="wide")
from common import csv, page_header, sidebar_filters  # noqa: E402

page_header("Macro lead/lag", "Which economic indicator leads which product family, by how many months, and is the "
            "relationship stable? Correlations of YoY demand growth with YoY indicator change, pre-whitened.")
sel = sidebar_filters()
best = csv("macro_leadlag.csv")
full = csv("macro_ccf.csv")
lead_only = st.toggle("Leading relationships only (lag ≥ 1 month)", value=True)
src = full[full["lag"] >= 1] if lead_only else full
i = src.groupby(["family", "indicator"])["corr"].apply(lambda s: s.abs().idxmax())
b = src.loc[i.values]
if sel.get("family"):
    b = b[b["family"].isin(sel["family"])]
a, c = st.columns(2)
f1 = px.imshow(b.pivot(index="family", columns="indicator", values="corr"), color_continuous_scale="RdBu",
               zmin=-0.8, zmax=0.8, text_auto=".2f", aspect="auto", title="Correlation at the best lag")
f1.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10))
a.plotly_chart(f1, use_container_width=True)
f2 = px.imshow(b.pivot(index="family", columns="indicator", values="lag"), color_continuous_scale="Viridis",
               text_auto=True, aspect="auto", title="Best lag (months the indicator leads)")
f2.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10))
c.plotly_chart(f2, use_container_width=True)

st.subheader("Cross-correlation by lag")
fam = st.selectbox("Family", sorted(full["family"].unique()))
ind = st.selectbox("Indicator", sorted(full["indicator"].unique()), index=3)
g = full[(full["family"] == fam) & (full["indicator"] == ind)]
f3 = px.bar(g, x="lag", y="corr", title=f"{ind} → {fam}")
thr = 2 / g["n"].mean() ** 0.5
f3.add_hline(y=thr, line_dash="dot")
f3.add_hline(y=-thr, line_dash="dot")
f3.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(f3, use_container_width=True)

st.subheader("Is the lag structure stable over time?")
roll = csv("macro_lag_stability.csv")
r = roll[(roll["family"] == fam) & (roll["indicator"] == ind)]
f4 = px.line(r, x="window_end", y=["lag", "corr"], markers=True, title="Best lag and correlation in rolling 36-month windows")
f4.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(f4, use_container_width=True)
st.dataframe(best[["family", "indicator", "lag", "corr", "granger_p", "lag_post2022", "corr_post2022", "stable"]]
             .round(3), hide_index=True, use_container_width=True, height=300)
