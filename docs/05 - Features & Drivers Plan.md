# 05 - Features & Drivers Plan (Workstream 3)

Back: [[00 - START HERE]] · Previous: [[04 - Anomaly & Unconstrained Demand Plan]]

## Forecasting setup determines feature design
We forecast up to **19 months ahead** from a single origin (2026-10). With a **direct multi-horizon** global model (see [[06 - Modeling & Validation Plan]]), every feature must be either:
- **known at the origin** (lags/rolling stats computed at the origin month), or
- **known in the future** (calendar, events, lifecycle, price schedule, test-row customer, given macro scenario path).

That rule prevents leakage and matches how the real forecast is produced. A recursive design would need different features; we decide in Phase 4 after a quick comparison.

## Feature groups
### Calendar (Core)
month, quarter, month sin/cos, year, months since 2019-01, working-days proxy, holiday fractions per region (LNY, Ramadan, NA July, EU August).

### Lags & rolling stats (Core), computed at the origin
- demand at origin lag 0..12, same month last year (lag 12 − h alignment), lag 24
- rolling mean/median/std/min/max over 3, 6, 12, 24 months; EWMA with several alphas
- trend slope over the last 6/12/24 months; YoY ratio
- **target encodings** computed on training windows only: series level, SKU level, family × region × month seasonality index

### Intermittency (Core)
- **ADI** (average inter-demand interval) and **CV²** → Syntetos-Boylan class: smooth / erratic / intermittent / lumpy
- zero-share in the last 12/24 months, months since last order, mean non-zero size
- **ordering cycle period + phase** (from WS2): does the target month fall on an expected order month? This is likely one of the strongest features for periodic orderers

### Hierarchy & static
family, BU, SKU, region, channel, customer, size class, powertrain (ICE/EV/Universal), SKU age, series age (MEA is young).

### Lifecycle (Advanced / Expert)
months since launch, months until EOL (signed), `is_in_eol_rundown`, `is_successor`, predecessor's volume at handover, replacement chain position, Lifecycle_Status.

### Events (Advanced)
- Expanded event table → features: `holiday_frac_{type}`, `price_increase_pct_next_jan` (with months-to-increase, giving the pre-buy signal), `surcharge_active`, `customer_shutdown` (Orion NA 2027-07→08), `capacity_step` (Helios India ≥2025-09), `market_age_mea`, `covid_lockdown`, `chip_constraint` (training only), `export_logistics`.
- Customer change flag: does the test customer differ from the last observed customer?

### Price (Advanced)
canonical price per SKU × year, price change % this year, relative price within family. The future price schedule is known from the events (+4.5% 2027, +4.0% 2028).

### Macro (Advanced): the lead/lag study
**Question: which indicator leads which family, and at what lag?**
1. Aggregate unconstrained demand to family × month and deseasonalise (STL), then YoY-difference or detrend both sides to avoid spurious correlation.
2. Cross-correlate each of the 8 indicators against each of the 7 families for lags 0–12. Pre-whiten (fit an AR to the indicator, apply the filter to both) to get honest CCFs.
3. Confirm with **Granger causality** tests and a distributed-lag regression per family.
4. Result: a **7 × 8 heatmap of best lag and its correlation**. That's the "macro lead/lag heatmap" optional view, and it picks which `indicator_lagK` features go into the model.

Expected hypotheses to test (not to assume): New_Vehicle_Orders leads Powertrain/Chassis by 2–4 months; Vehicle_Inventory_Days moves inversely; PMI leads industrial-heavy families (Hydraulics, Precision Machined); EV_Adoption pushes EV SKUs up and ICE down; Raw_Material_Cost relates to price/pre-buy behaviour; Fuel_Price relates to aftermarket.

**Macro in the forecast window:** the file supplies 2026-11 → 2028-05 values (a scenario path). Use them as known-future covariates, which makes the **what-if view (PMI −3)** a real model re-run, not a cosmetic change.

### Expert: is the lag structure stable?
- Rolling-window CCF (e.g. 36-month windows stepped monthly) → plot best-lag and correlation over time per family-indicator pair.
- Chow / structural-break tests on the distributed-lag coefficients around COVID and the chip shortage.
- Conclude whether we should trust long-history macro relationships or down-weight pre-2022 data.

### Expert: the EV shift
- Demand share of ICE vs EV vs Universal over time, by family and region.
- Elasticity of ICE and EV SKU demand to EV_Adoption_Index; does the forecast-window EV path imply ICE decline? Feed `ev_index × powertrain` interactions to the model.
- The 3 planned launches are EV/Universal successors of ICE SKUs, which fits the story.

## Outputs
- `data/features/train_matrix.parquet` (origin × series × horizon rows), `predict_matrix.parquet`
- `outputs/macro_leadlag.csv` (family × indicator × lag → corr, p-value, Granger p)
- `outputs/macro_lag_stability.csv`
- `outputs/ev_shift.csv`
- `outputs/feature_importance.csv` (after modelling, SHAP)
