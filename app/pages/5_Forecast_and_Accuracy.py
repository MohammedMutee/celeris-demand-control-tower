import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Forecast & accuracy", layout="wide")
from common import (aggregate_forecast, apply_filters, csv, filter_caption, forecast, history, js,  # noqa: E402
                    page_header, parquet, sidebar_filters, timeline)

page_header("Forecast & accuracy", "The forecast continues the history; accuracy comes from a rolling-origin "
            "backtest (7 forecast origins, each forecasting up to 19 months ahead, never seeing the future).")
sel = sidebar_filters()
st.caption(filter_caption(sel))
h, f = apply_filters(history(), sel), apply_filters(forecast(), sel)
st.plotly_chart(timeline(h, f, "units", "Forecast vs history", height=460), use_container_width=True)

ms = js("model_selection.json")
cov = js("quantile_coverage.json")
c = st.columns(4)
c[0].metric("Ensemble backtest WAPE", f"{ms['backtest_mean_wape']:.1%}")
c[1].metric("Seasonal-naive baseline WAPE", f"{ms['ladder']['snaive']:.1%}")
c[2].metric("P10–P90 coverage (held-out folds)", f"{cov['coverage_p10_p90']:.1%}", help="Target 80 %")
c[3].metric("Structural share in ensemble", f"{ms['structural_weight']:.0%}")

bt = apply_filters(parquet("backtest_long.parquet"), sel)
models = [m for m in ["snaive", "ma12", "croston", "sba", "tsb", "structural"] +
          [e["name"] for e in ms["ensemble"]] + ["ensemble"] if m in bt]


def wb(g, m):
    a = g["actual"].sum()
    return pd.Series({"WAPE": (g[m] - g["actual"]).abs().sum() / a, "Bias": (g[m].sum() - a) / a})


st.subheader("Model ladder (selected data)")
lad = pd.DataFrame({m: wb(bt, m) for m in models}).T.reset_index(names="model")
fig = px.bar(lad, x="model", y="WAPE", text_auto=".1%", title="Backtest WAPE by model (lower is better)")
fig.update_layout(height=360, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

dim = st.radio("Accuracy by", ["family", "region", "channel", "h"], horizontal=True,
               format_func=lambda x: "horizon (months ahead)" if x == "h" else x)
t = bt.groupby(dim).apply(lambda g: pd.concat([wb(g, "ensemble"), wb(g, "snaive").add_prefix("naive ")]),
                          include_groups=False).reset_index()
a, b = st.columns(2)
f1 = px.bar(t, x=dim, y=["WAPE", "naive WAPE"], barmode="group", title=f"WAPE by {dim}: ensemble vs naive")
f1.update_layout(height=380, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
a.plotly_chart(f1, use_container_width=True)
f2 = px.bar(t, x=dim, y="Bias", title=f"Ensemble bias by {dim} (+ = over-forecast)")
f2.update_layout(height=380, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
b.plotly_chart(f2, use_container_width=True)

st.subheader("Accuracy across the hierarchy (bottom-up forecasts add up exactly)")
hi = csv("backtest_hierarchy.csv")
hp = hi[hi["model"].isin(["snaive", "structural", "ensemble"])].pivot(index="level", columns="model", values="wape")
order = ["total", "business_unit", "family", "region", "family_region", "sku", "sku_region"]
st.dataframe(hp.reindex(order).style.format("{:.1%}"), use_container_width=True)

st.subheader("Backtest: forecast vs actual by origin")
o = st.selectbox("Forecast origin", sorted(bt["origin"].unique()))
g = bt[bt["origin"] == o].groupby("date")[["actual", "ensemble", "p10", "p90", "snaive"]].sum().reset_index()
f3 = px.line(g, x="date", y=["actual", "ensemble", "snaive"], title=f"Origin {o}: actual vs forecasts")
f3.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(f3, use_container_width=True)

st.subheader("Why does the model forecast this? (top drivers)")
imp = csv("feature_importance.csv").head(20)
f4 = px.bar(imp.iloc[::-1], x="gain_share", y="feature", orientation="h", title="Feature importance (gain share)")
f4.update_layout(height=480, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(f4, use_container_width=True)
if sel.get("sku") and sel.get("region") and len(sel["sku"]) == 1 and len(sel["region"]) == 1:
    import lightgbm as lgb

    from common import OUT
    px_ = parquet("forecast_features.parquet")
    rows = px_[(px_["sku"] == sel["sku"][0]) & (px_["region"] == sel["region"][0])]
    if len(rows):
        mth = st.selectbox("Month", rows["date"].tolist())
        x = rows[rows["date"] == mth]
        contrib = None
        for e in ms["ensemble"]:
            mdl = lgb.Booster(model_file=str(OUT / "models" / f"{e['name']}.txt"))
            feats = mdl.feature_name()
            xx = x[feats].copy()
            for c in ["family", "region", "channel", "customer", "powertrain", "size_class"]:
                if c in xx:
                    xx[c] = xx[c].astype("category")
            cvals = pd.Series(mdl.predict(xx, pred_contrib=True)[0][:-1], index=feats) * e["weight"]
            contrib = cvals if contrib is None else contrib.add(cvals, fill_value=0)
        top = contrib.abs().sort_values(ascending=False).head(12).index
        ex = pd.DataFrame({"feature": top, "contribution": contrib[top].values,
                           "value": [x[c].iloc[0] for c in top]})
        f5 = px.bar(ex.iloc[::-1], x="contribution", y="feature", orientation="h", hover_data=["value"],
                    title=f"{sel['sku'][0]} · {sel['region'][0]} · {mth}: what pushes the forecast up / down")
        f5.update_layout(height=440, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f5, use_container_width=True)
        st.caption("SHAP contributions (TreeSHAP, LightGBM) to the scaled monthly-rate prediction, "
                   "weighted by ensemble weight. Positive = raises the forecast.")
else:
    st.caption("Select exactly one SKU and one region to see a per-month explanation.")
