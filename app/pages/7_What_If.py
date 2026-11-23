import plotly.express as px
import streamlit as st

st.set_page_config(page_title="What-if", layout="wide")
from common import fmt_units, page_header, parquet, sidebar_filters  # noqa: E402

page_header("What-if scenarios", "Each scenario shifts the macro path for the whole forecast window and re-runs the "
            "trained model. The difference shows how sensitive demand is to the economy.")
sel = sidebar_filters()
s = parquet("scenarios.parquet")
if sel.get("family"):
    s = s[s["family"].isin(sel["family"])]
sc = st.selectbox("Scenario", s["scenario"].unique())
if sc.startswith("EV"):
    st.info("Caveat: the EV adoption index rises almost monotonically through history, so the model partly reads "
            "it as a general growth trend. Treat this scenario's *total* effect with caution; the reliable EV "
            "effect is the mix shift from ICE to EV SKUs shown on the Lifecycle page (ICE share elasticity −1.2, "
            "EV +4.3).")
x = s[s["scenario"] == sc]
tot = x.groupby("date")[["p50", "base_p50"]].sum().reset_index()
d = tot["p50"].sum() - tot["base_p50"].sum()
c = st.columns(3)
c[0].metric("Baseline 19-month units", fmt_units(tot["base_p50"].sum()))
c[1].metric(f"Scenario: {sc}", fmt_units(tot["p50"].sum()), delta=f"{d / tot['base_p50'].sum():+.2%}")
c[2].metric("Change in units", f"{d:+,.0f}")
f = px.line(tot, x="date", y=["base_p50", "p50"], title="Baseline vs scenario (P50 units)")
f.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(f, use_container_width=True)
fam = x.groupby("family")[["p50", "base_p50"]].sum()
fam["change"] = fam["p50"] / fam["base_p50"] - 1
f2 = px.bar(fam.reset_index(), x="family", y="change", title="Impact by family")
f2.update_layout(height=360, yaxis_tickformat=".2%", margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(f2, use_container_width=True)
allsc = s.groupby("scenario")[["p50", "base_p50"]].sum()
allsc["change"] = allsc["p50"] / allsc["base_p50"] - 1
st.dataframe(allsc.style.format({"p50": "{:,.0f}", "base_p50": "{:,.0f}", "change": "{:+.2%}"}),
             use_container_width=True)
