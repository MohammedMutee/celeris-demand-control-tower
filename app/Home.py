"""Celeris Planning Control Tower - entry page.   Run:  make dashboard"""
import streamlit as st

st.set_page_config(page_title="Celeris Control Tower", page_icon="📈", layout="wide")

from common import fmt_units, fmt_usd, forecast, history, js, page_header, sidebar_filters  # noqa: E402

page_header("Celeris Planning Control Tower",
            "Demand history (2019-01 → 2026-10, cleaned and corrected for stockouts) and the 19-month forecast "
            "(2026-11 → 2028-05) in one app. Filters in the sidebar apply to every page.")
sidebar_filters()

h, f = history(), forecast()
sel = js("model_selection.json")
c1, c2, c3, c4 = st.columns(4)
c1.metric("History, unconstrained demand", fmt_units(h["demand"].sum()), help="2019-01 → 2026-10")
c2.metric("19-month forecast (P50)", fmt_units(f["p50"].sum()))
c3.metric("Forecast revenue (P50)", fmt_usd(f["revenue_p50"].sum()))
c4.metric("Backtest WAPE (ensemble)", f"{sel['backtest_mean_wape']:.1%}",
          help="Mean over 7 rolling origins, 19-month horizon")

st.markdown("""
### Pages
**Training-data dashboard (what happened)**
- **Executive overview**: units and revenue from history into the forecast, mix, headline numbers
- **Data health**: every data-quality rule, rows affected, before vs after
- **Demand explorer**: family → SKU → region → customer, seasonality, zero months, ordering cycles, lifecycle
- **Anomaly monitor**: flagged points with type, reason and action
- **Stockouts & lost sales**: shipped vs requested, lost units and revenue

**Prediction dashboard (what will happen)**
- **Forecast & accuracy**: forecast vs history, backtest WAPE / bias by family, region, horizon
- **Macro lead/lag**: which indicator leads which family, and is it stable?
- **What-if**: macro scenarios re-scored by the model
- **Procurement plan**: order quantities from P90
- **Lifecycle**: launches, EOLs, handover curves, EV shift
- **Planner exceptions**: series that need a human look
- **Data download**

### How the forecast works
`demand = monthly rate × months covered by the order × order expected this month`. Each customer's
ordering cycle (monthly, every 2nd month, quarterly, …) is detected from history and predicts the zero months.
A global LightGBM model forecasts the monthly rate 1–19 months ahead, blended with a structural
level × seasonality model. Lifecycle rules, forecast-window events and calibrated P10/P90 bands are layered on top.
""")
