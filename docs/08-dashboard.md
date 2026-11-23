# Planning Control Tower (Dashboard)

Date: 2026-11-23 · Code: `app/` (Streamlit + Plotly) · Run: `make dashboard` → http://localhost:8501

## Requirements → what we built
| Requirement | Status |
|---|---|
| One app, two parts (training / prediction) | ✅ multipage app; Home explains both parts |
| Shared filters: family, SKU, region, channel, customer | ✅ sidebar on every page, **persist across pages** (session-state copy, so Streamlit doesn't reset them on page change); SKU list narrows to the chosen family |
| Reads pipeline outputs only | ✅ `outputs/` + `data/clean/`; no typed numbers |
| History + forecast on one continuous timeline, forecast clearly marked | ✅ shipped (grey), unconstrained demand (blue), P50 dashed orange, P10–P90 band, dotted line + "Forecast →" at 2026-10 |
| Unconstrained demand shown in history | ✅ shipped vs demand on every timeline |
| P10/P50/P90 | ✅ with **calibrated aggregation** of intervals for any filter |

## Pages (5 required + all 7 optional)
| # | Page | Highlights |
|---|---|---|
| 1 | **Executive overview** (both) | KPI tiles (19-mo units with P10–P90, revenue, next-12 vs last-12), units/revenue timeline, mix by family/region/channel per year incl. forecast years, headline table vs run rate |
| 2 | **Data health** (training) | 43 rules by file × type, rows affected (log scale), treemap, rule catalogue, before/after (rows, nulls, distinct labels), **drill into any rule's records**, master-data conflicts, stockout-file reconciliation 96.9% → 99.98% |
| 3 | **Demand explorer** (training) | Drill family → SKU → region → customer via a breakdown table + filters; seasonality heatmap (month × year), zero-month share, current ordering cycles, lifecycle table, cycle segments for small selections |
| 4 | **Anomaly monitor** (training) | Flags plotted on the series (marker size = count), by type / classification / action, the full register (type, reason, action), event effects with verdicts, holiday regressions, pre-buy table, customer switches / lost accounts |
| 5 | **Forecast & accuracy** (prediction) | Forecast vs history, model ladder, WAPE + bias by family / region / channel / horizon vs naive, hierarchy accuracy table, backtest replay per origin, feature importance, **per-month SHAP explanation for a selected series** |
| 6 | Macro lead/lag heatmap | Correlation + best-lag heatmaps (lead-only toggle), CCF by lag with significance band, rolling-window stability |
| 7 | What-if | 5 macro scenarios re-scored by the model; total + by family; EV caveat |
| 8 | Procurement plan from P90 | Lead-time slider, P50/P90 toggle, release-month plan, safety stock and its cost, CSV download |
| 9 | Lifecycle | Gantt of launches / EOLs / replacements, handover curve, predecessor → successor history + forecast, EV/ICE mix (history + forecast) |
| 10 | Stockouts & lost sales | Lost units / revenue by reason and family, fill rate, method validation |
| 11 | Exception list for planners | Lost accounts, new series, big change vs history, wide band, hard-to-forecast, customer switch, recent cycle change, event/lifecycle adjustments; ranked by revenue at stake; CSV |
| 12 | Data download | Clean history, forecast, submission, DQ report, register, metrics (all filtered) |

## Quality checks
- `tests/test_dashboard.py` runs **every page headless (Streamlit AppTest)**, with and without filters → 26/26 pass. These tests already caught one real bug (a categorical-dtype division on the overview page).

## Deployment
Live on Streamlit Community Cloud: https://celeris-demand-control-tower.streamlit.app/ (reads the committed `outputs/` and `data/clean/`).
