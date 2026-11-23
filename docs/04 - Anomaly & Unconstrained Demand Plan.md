# 04 - Anomaly & Unconstrained Demand Plan (Workstream 2)

Back: [[00 - START HERE]] · Previous: [[03 - Cleaning Plan]]

## The key judgement: error vs real event
| Class | Examples | Action |
|---|---|---|
| **Data error** | cents prices, ×1000 units, duplicates, sign flips, currency, bad dates, ERP-migration gaps | **Fix** (WS1) and log |
| **Real, recurring** | holidays, seasonality, pre-buy before price rises, ordering cycles | **Model** (features) |
| **Real, one-off** | bulk orders, strike, COVID, Red Sea | **Exclude or down-weight** in training, and explain |
| **Real, supply-side** | stockouts, chip shortage | **Correct to unconstrained demand** |
| **Real, structural** | MEA launch, Helios capacity step-up, EV shift, customer switch, lost account, lifecycle | **Model the new level** and don't treat it as an outlier |

## Anomaly register (`outputs/anomaly_register.csv`)
Columns: `anomaly_id, date, sku, region, customer, family, type, detector, raw_value, expected_value, score, classification (error/real), reason, linked_event_id, action (fix/model/exclude/correct), corrected_value`.
That's the backbone of the **Anomaly monitor** view.

## Detectors
1. **Point outliers:** robust z-score on log1p(units) vs a rolling median/MAD per series, using seasonal-adjusted residuals (STL where the series is long and smooth enough).
2. **Scale-cluster test:** ratio to the series median ≈ 10ⁿ → error (fixed in WS1). Otherwise it's a candidate real spike.
3. **Event join:** every flagged point is checked against the events calendar (scope match on Global / Region / Family / Channel / Customer). If it's explained by an event, it's real.
4. **Pre-buy detector:** for each January price increase, compare Nov+Dec vs baseline and Jan+Feb vs baseline. Expect a large effect for 2023 (9%, announced 2022-11-15) and maybe 2022 (5%). Estimate the **pull-forward elasticity vs price-increase size**, then use it for 2027-01 (+4.5%) and 2028-01 (+4.0%).
5. **Holiday effects:** estimate per region: LNY on APAC (Jan/Feb shift by date), Ramadan on MEA (moves ~11 days a year), NA July, EU August. Keep the fraction of the month covered as the feature.
6. **Structural breaks:** per family × region aggregates, then PELT / binary segmentation (ruptures) on level and trend. Expected: COVID 2020, chip 2021-22, MEA start 2022-01, Helios India 2025-09, EV/ICE trend changes.
7. **Ordering-cycle detection (Expert):** per series, look at the zero/non-zero pattern: autocorrelation of the order indicator, typical gap between orders (1/2/3/6 months), and the phase. Use a rolling window to **detect cycle changes** (e.g. monthly → quarterly). On a cycle change, also check that volume per order scales (quarterly orders ≈ 3× monthly).
8. **Customer switches / lost accounts (Expert):** per SKU-region, track Customer_Name over time. A switch = the customer changes and demand continues (maybe with a level change). A lost account = the series goes to permanent zero / no rows before EOL. Also compare with **test.csv customers**, which reveal future switches.
9. **Master vs transactions (Expert):** sales outside launch/EOL, family/BU disagreements, descriptions that don't match, discontinued SKUs still selling, planned-launch SKUs with history, Replaces_SKU overlap anomalies.
10. **Noise-event verification:** test whether rebrand, CRM upgrade, Hannover Messe, sales-director change and the Pune conference show any effect. Report "checked, no effect" or "effect found". Showing we checked is itself the insight.

## Unconstrained demand estimation (the training target)
`demand_t` replaces `Units_Sold` as the training target.

**A. Listed stockout months (5,830):**
- `demand_t = Requested_Units`
- Month t+1: if `Carried_In_Units` were shipped in t+1, then `demand_{t+1} = shipped_{t+1} − carried_in_{t+1}` (unless t+1 is itself listed, in which case its Requested_Units wins).
- Check chain consistency and log it.

**B. Unlisted stockout months (the file is incomplete):**
- Candidates: Electrical Assemblies 2021-03 → 2022-09 (semiconductor), plus series with listed stockouts in neighbouring months, plus "dip followed by a catch-up spike" patterns.
- Detection features: a negative residual vs the expected level, followed by a positive residual of similar size (backorder catch-up), sitting in a known constraint window or adjacent to listed months.
- Estimate: expected demand from a model **trained on non-constrained months** (seasonal level from the same series plus a family-level trend), and calibrate the method using the listed stockouts. We know the true answer there, so we can measure our imputation error. **That validation is a good point to present.**
- Flag the imputed values with `is_imputed_demand` and a confidence level.

**C. Data-loss gaps (LatAm ERP 2023-03→05 etc.):** impute for continuity of lags, but **exclude from loss** (sample weight 0).

**D. One-off real events (COVID months, strike, bulk orders):** keep the raw value for display, give it low/zero training weight, and add event features so the model learns the pattern without being dragged by it.

## Outputs
- `data/clean/demand.parquet`: per Date-SKU-Region: `shipped`, `demand_unconstrained`, `lost_units`, `is_stockout_listed`, `is_stockout_inferred`, `is_data_loss`, `train_weight`, `cycle_period`, `cycle_phase`, `customer`, `event_flags…`
- `outputs/anomaly_register.csv`, `outputs/structural_breaks.csv`, `outputs/order_cycles.csv`, `outputs/customer_changes.csv`, `outputs/master_conflicts.csv`, `outputs/prebuy_effects.csv`
- Charts and tables for the "Stockouts & lost sales" optional view.
