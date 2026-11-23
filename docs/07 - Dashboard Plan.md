# 07 - Dashboard Plan (Workstream 5)

Back: [[00 - START HERE]] · Previous: [[06 - Modeling & Validation Plan]]

## Tool choice
**Streamlit + Plotly** (recommended):
- Python end to end, reads pipeline Parquet directly, fast to build, easy local setup (`streamlit run app/Home.py`).
- Multipage app with **shared filters** in `st.session_state` (sidebar: family, SKU, region, channel, customer), persisted across pages.
- Deploys free to Streamlit Community Cloud (Expert: "deployed"). Needs small precomputed outputs (Parquet, <100 MB).
- Alternative considered: Dash (more control, more code) or Evidence/React (more polish, more work). We'll stick with Streamlit unless we hit a wall.

## Design rules
- **One continuous timeline** wherever history and forecast both appear: history solid, forecast dashed, a P10–P90 band shaded, a vertical "forecast starts" line at 2026-10/11, plus the unconstrained-demand line vs the shipped line in history.
- Same colour per family everywhere. Consistent number formats (units, $ revenue).
- Every page has a one-line "what this tells a planner" caption.
- Reads only `outputs/*.parquet|csv|json`, nothing typed by hand.
- Fast: pre-aggregate in the pipeline (`outputs/dash/…`), and use `st.cache_data`.

## Required views (Core)
| # | View | Part | Must show | Our design |
|---|---|---|---|---|
| 1 | **Executive overview** | Both | Units + revenue trend history → forecast on one timeline; mix by family / region / channel; headline 19-month forecast | KPI tiles (19-mo units, revenue, YoY %, backtest WAPE), timeline chart with band, stacked mix bars/treemap, top movers |
| 2 | **Data health** | Training | Issues by type and file, rows affected, before vs after | From `dq_report.json`: issue bar chart by file × type, before/after table (nulls, distinct labels, duplicates, revenue mismatches), drill-down to `dq_issues.csv` sample rows |
| 3 | **Demand explorer** | Training | Drill-down family → SKU → region → customer; seasonality, zero months, lifecycle | Linked drill (treemap/selectors), series chart (shipped vs unconstrained), seasonal profile heatmap (month × year), zero-month calendar strip, lifecycle markers (launch/EOL/replacement), intermittency class badge, ordering cycle |
| 4 | **Anomaly monitor** | Training | Flagged points on the time series, with type, reason, action | Series chart with markers coloured by type, register table filtered to the selection, event overlays, counts by type/action |
| 5 | **Forecast & accuracy** | Prediction | Forecast vs history; backtest WAPE and bias by family, region, horizon | Timeline with P10/P50/P90, WAPE/bias heatmaps (family × horizon, region × horizon), model comparison ladder (baselines vs ML vs ensemble), coverage of the P10–P90 interval |

## Optional views (we build all 7)
| View | Content |
|---|---|
| **Macro lead/lag heatmap** | family × indicator, best lag + correlation; click a cell to see the CCF and the rolling-stability plot |
| **What-if** | Slider/inputs: PMI −3, EV index +10%, price increase %, Orion shutdown on/off. Re-scores the model on modified covariates (precomputed scenario grid, or live scoring of a lightweight model) and shows the delta vs baseline |
| **Procurement plan from P90** | Per SKU × month: P90 demand, suggested order quantity, safety stock = P90 − P50, revenue at risk; downloadable |
| **Lifecycle** | Gantt of launches/EOLs, predecessor → successor handover curves (history + forecast), EV vs ICE mix over time |
| **Stockouts & lost sales** | Shipped vs requested, lost units and revenue by reason / family / region, listed vs inferred stockouts, fill-rate trend |
| **Exception list for planners** | Auto-flagged series: forecast jumps vs history, wide P10–P90, high backtest error, customer switch ahead, EOL/launch ahead, event-affected months. Sorted by revenue impact |
| **Data download** | Filtered CSV/Excel download of clean history, forecasts, anomaly register, DQ report |

## Possible extras (originality)
- "Explain this forecast": SHAP waterfall for a selected series-month (why is Dec 2026 high? → pre-buy + seasonality).
- A methodology page linked from every view (short, visual).
- Natural-language summary per filter selection (template-generated from numbers, not an LLM, so it's always correct).

## Deliverable checks
- [ ] `README` with setup (venv/uv, `make pipeline`, `make dashboard`)
- [ ] App starts with no errors from a fresh clone after running the pipeline
- [ ] All 5 filters work on every page where relevant
- [ ] History + forecast join on one timeline everywhere both appear
- [ ] Deployed URL (Streamlit Cloud) + screenshots in the repo
