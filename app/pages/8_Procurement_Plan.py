import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Procurement plan", layout="wide")
from common import apply_filters, filter_caption, fmt_units, fmt_usd, forecast, page_header, sidebar_filters  # noqa

page_header("Procurement plan from P90", "Plan supply to the P90 forecast so demand is covered about 9 months in "
            "10. Orders are booked 2–3 months before the requested delivery month, so each line gets a "
            "release month.")
sel = sidebar_filters()
st.caption(filter_caption(sel))
f = apply_filters(forecast(), sel)
c1, c2 = st.columns(2)
lead = c1.slider("Procurement lead time (months)", 1, 6, 3)
level = c2.select_slider("Service level to plan for", ["P50", "P90"], value="P90")
col = "p90" if level == "P90" else "p50"
f = f.assign(plan_units=f[col].round(), safety_stock=(f["p90"] - f["p50"]).clip(lower=0),
             release_month=(pd.PeriodIndex(f["date"], freq="M") - lead).astype(str))
f["plan_value_usd"] = f["plan_units"] * f["price"]
k = st.columns(4)
k[0].metric("Planned units", fmt_units(f["plan_units"].sum()))
k[1].metric("Of which safety stock (P90 − P50)", fmt_units(f["safety_stock"].sum()))
k[2].metric("Plan value", fmt_usd(f["plan_value_usd"].sum()))
k[3].metric("Inventory cost of the buffer", fmt_usd((f["safety_stock"] * f["price"]).sum()),
            help="Value of units bought above P50")
m = f.groupby("release_month")[["plan_units", "safety_stock"]].sum().reset_index()
fig = px.bar(m, x="release_month", y=["plan_units", "safety_stock"], barmode="overlay",
             title="Units to release per month (by procurement release month)")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)
tab = (f.groupby(["sku", "family", "release_month", "date"], observed=True)
       .agg(plan_units=("plan_units", "sum"), p50=("p50", "sum"), p90=("p90", "sum"),
            safety_stock=("safety_stock", "sum"), plan_value_usd=("plan_value_usd", "sum")).reset_index())
tab = tab[tab["plan_units"] > 0].rename(columns={"date": "delivery_month"})
st.dataframe(tab.head(5000), hide_index=True, use_container_width=True, height=420)
st.download_button("Download procurement plan (CSV)", tab.to_csv(index=False), "procurement_plan.csv")
