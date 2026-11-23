# 01 - Problem Understanding

Back: [[00 - START HERE]]

## Business context
- **Company:** Celeris Automobili Industrial, a tier-1 supplier of automotive and industrial components.
- **Scale:** ~600 SKUs, 7 product families, 5 business units, 6 regions, 4 sales channels, OEM / distributor customers.
- **Why forecasts matter:** monthly SKU × region forecasts drive **procurement, production scheduling and inventory**. Planners need a point forecast *and* a range (P90 drives safety stock and procurement).
- **History covers:** COVID (2020), chip shortage (2021-03 → 2022-09, Electrical Assemblies), the inflation spike (price increases of 5% in 2022, a 6% surcharge, 9% in 2023), the EV shift, and the opening of the Middle East & Africa region (2022-01).

## Deliverables
1. **Submission CSV:** `Date, SKU_ID, Region, Units_Sold_Forecast`, one row per `test.csv` row (58,219), non-negative integers, no blanks.
2. **Cleaning pipeline:** tested, runs with one command, produces a machine-readable data-quality report.
3. **Anomaly register:** each flagged point with type, reason, and action (fix / model / exclude).
4. **Drivers analysis:** which macro indicator leads which family, at what lag, and whether that holds over time. The EV shift.
5. **Forecast model(s):** 2 baselines + better models, rolling-origin backtest with a 19-month horizon, P10/P50/P90, coherent across the hierarchy.
6. **Dashboard app:** one app, two parts (training / prediction), shared filters, runs locally with documented setup, reads only pipeline outputs.
7. **Presentation:** methodology, key decisions, business impact (implied by the jury criteria).

## Business rules (verbatim intent) and how we enforce them
| # | Rule | Enforcement |
|---|------|-------------|
| 1 | `Date` is authoritative; if Year/Month disagree, trust Date | Parse Date in every format, rebuild Year/Month from it, log mismatches (268 seen) |
| 2 | Repeated `Record_ID` = loaded twice; keep the latest `Load_Timestamp` | Dedupe by Record_ID, keep max timestamp (3,682 repeats) |
| 3 | `Units_Sold = 0` means no order; **a missing row is not a zero** | Never fill missing months with 0 by default. Classify every missing month as *not ordered* (ordering cycle), *data loss* (e.g. LatAm ERP migration), or *outside lifecycle* |
| 4 | Price per SKU, USD, the same in every region, changes on 1 Jan, except event surcharges | Build a canonical price table per SKU × year; use it to detect cents / ×10 / currency errors and fix revenue |
| 5 | No sales after EOL_Date or before a planned Launch_Date | Hard post-processing clamp on forecasts; flag violations in train as master-data conflicts |
| 6 | Customers order on their own cycle (monthly / bi-monthly / quarterly…), and the cycle can change | Detect the cycle per series, detect changes, use the cycle phase as a feature. This is a big lever on accuracy |
| 7 | Forecast window has no supply limits: forecast **unconstrained demand** | Train on demand corrected for stockouts (Requested_Units, plus inferred corrections for months the stockout file misses) |

## Tier checklist (target: everything)
### 1. Cleaning
- [ ] Core: profile all files, standardise labels/dates, dedupe, one clean row per Date-SKU-Region
- [ ] Advanced: reconcile units × price = revenue, fix unit and currency errors, missing rows vs zeros, clean indicators
- [ ] Expert: tested one-command pipeline, machine-readable DQ report

### 2. Anomalies
- [ ] Core: flag outliers, separate errors from real events, anomaly register
- [ ] Advanced: structural breaks, pre-buying before price rises, holidays, stockouts, estimate unconstrained demand
- [ ] Expert: ordering-cycle changes, customer switches, lost accounts, master data vs transactions

### 3. Features & drivers
- [ ] Core: calendar features, lags, rolling stats, intermittency class
- [ ] Advanced: which indicator leads which family and at what lag, event + lifecycle features
- [ ] Expert: is the lag structure stable over time? the EV shift

### 4. Forecasting
- [ ] Core: 2 baselines + 1 better model, time-based backtest
- [ ] Advanced: intermittent-demand methods, global ML model, rolling-origin backtest with 19-month horizon
- [ ] Expert: new launches via Replaces_SKU, EOL, events in the forecast window, P10/P50/P90, hierarchy-coherent forecasts

### 5. Dashboard
- [ ] Core: 5 required views
- [ ] Advanced: +2 optional views
- [ ] Expert: +4 optional views, deployed or polished. **We'll do all 7 optional views and deploy.**

## What "unconstrained demand" means here
`train.Units_Sold` = shipments. In a stockout month, shipments < demand, and **backorders get shipped the following month**, which inflates that month. So a stockout creates two distortions:
- month *t*: under-shipment → true demand = `Requested_Units`
- month *t+1*: over-shipment from `Carried_In_Units` → true demand ≈ shipped − carried-in (if t+1 is not itself listed)

The stockout file is **known to be incomplete**, so we also have to *infer* corrections for unlisted stockout months (see [[04 - Anomaly & Unconstrained Demand Plan]]).

## Unknowns
- **Scoring metric for the submission is not stated.** WAPE is the metric named for the dashboard backtest, so we optimise and report WAPE + bias and also track RMSE/MAE. See [[10 - Risks & Open Questions]].
