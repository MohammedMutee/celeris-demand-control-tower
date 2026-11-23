import streamlit as st

st.set_page_config(page_title="Data download", layout="wide")
from common import OUT, apply_filters, csv, filter_caption, forecast, history, page_header, sidebar_filters  # noqa

page_header("Data download", "Everything the dashboard shows, filtered by the sidebar selection.")
sel = sidebar_filters()
st.caption(filter_caption(sel))
h = apply_filters(history(), sel)
f = apply_filters(forecast(), sel)
c = st.columns(2)
hc = ["date", "sku", "region", "family", "bu", "channel", "customer", "shipped", "demand", "demand_source", "price",
      "revenue_demand", "lost_units_est", "order_expected", "train_weight"]
c[0].download_button(f"Clean history ({len(h):,} rows)", h[hc].to_csv(index=False), "history_clean.csv")
c[1].download_button(f"Forecast P10/P50/P90 ({len(f):,} rows)", f.to_csv(index=False), "forecast.csv")
c[0].download_button("Submission file", (OUT / "submission.csv").read_bytes(), "submission.csv")
c[1].download_button("Data-quality report (JSON)", (OUT / "dq_report.json").read_bytes(), "dq_report.json")
c[0].download_button("Anomaly register", (OUT / "anomaly_register.csv").read_bytes(), "anomaly_register.csv")
c[1].download_button("Backtest metrics", csv("backtest_metrics.csv").to_csv(index=False), "backtest_metrics.csv")
st.dataframe(f.head(200), hide_index=True, use_container_width=True)
