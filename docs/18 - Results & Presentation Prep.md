# 18 - Results & Presentation Prep

Back: [[00 - START HERE]] · Jury criteria: [[11 - Stand Out Strategy]]

## One-slide summary
> We turned a messy ERP extract into trusted **unconstrained demand**. We discovered that **customers order on deterministic cycles** and built a forecast around that structure. A global LightGBM ensemble cuts error by **23% vs seasonal naive** (WAPE 28.3% vs 36.8% at SKU × region; **6.0% at company level**), with **calibrated P10–P90 bands (84% coverage)** and a planner-ready control tower.

## Headline numbers
| | |
|---|---|
| Data errors found & fixed | 43 rules, e.g. 614 cents prices, 498 currency, 385 packs-of-10, 491 sign flips, 358 ×10/×100 units, 3,682 duplicate loads, 942 missing regions, 30 typo years |
| Independent validation of cleaning | Shipments match the stockout file: 96.9% raw → **99.98%** cleaned |
| Ordering cycles | Order / no-order predicted with **99.3%** holdout accuracy; 49 cycle changes, 16 lost accounts |
| Lost sales from stockouts | **≈119k units ≈ $119M** ($93M in Electrical Assemblies, chip shortage) |
| Chip-shortage demand estimator | 11.4% WAPE (cross-validated on listed stockouts) |
| Forecast accuracy (7-fold rolling origin, 19-month horizon) | Ensemble **28.3%** vs naive 36.8%; best on every fold |
| Interval calibration | 84% coverage on held-out folds |
| Forecast | 2.80M units, **$2.52B**, next 12 months **+4.6%** |
| Tests | **108** automated tests incl. every dashboard page |

## Storyline (≈10 min)
1. **The planner's problem** (1 min): monthly SKU × region forecasts drive procurement. Shipments ≠ demand, missing ≠ zero, and a P90 is needed for safety stock.
2. **Trust the data** (2 min): one ugly series before → after (cents price, ×100 units, sign flip, stockout dip + catch-up). The stockout-file cross-check (99.98%).
3. **Error vs real event** (2 min): register counts; events tested, not assumed (strike −77%, COVID, NA July −44% + August catch-up, Helios +24% permanent, rebrand / CRM / trade fair = no effect); pre-buy only before the announced 9% increase.
4. **The key insight** (1.5 min): ordering cycles → `demand = rate × months covered × order expected`. Zeros become predictable; Croston-type methods fail (50%) because they ignore it.
5. **Model & validation** (2 min): ladder chart, 7 folds, macro ablation (helps 1 pp), tuning (only 0.2 pp → noise-bound), ensemble, hierarchy table, coverage.
6. **Forecast & what to do** (1.5 min): timeline with band; launches via the learned handover curve; Orion shutdown; procurement plan from P90; exception list.

## Likely jury questions (with answers)
- **Why not deep learning (N-BEATS / TFT)?** M5 showed global GBMs dominate on this kind of retail/industrial panel. Our tuning showed the model is noise-bound (0.17 pp from 14 trials), so more capacity won't help. A structural insight (the cycles) mattered more than model class.
- **How do you know the stockout correction is right?** Listed stockouts hold the truth (Requested_Units). We cross-validated the estimator on them: 11.4% WAPE, vs 33% for the alternative.
- **Why not impute every suspicious dip as a stockout?** We measured it: the best detector reached ~7.5% precision. Imputing would add error, so we flagged and down-weighted instead.
- **Isn't the forecast biased low?** By 2–4%, by design: WAPE is minimised by the median. A correction factor made WAPE worse on the backtest (tested).
- **Leakage?** Every backtest fold rebuilds cycles, seasonality and features from data ≤ origin. Rows exist only where the series exists (mirrors test.csv).
- **How do launches work without history?** Handover curve from 6 past Replaces_SKU transitions; leave-one-out validated; wide bands.
- **Do forecasts add up?** Yes, bottom-up. The table shows accuracy at every level (6.0% total → 28.3% SKU × region).
- **Macro?** PMI leads all families by 1 month (Granger p < 0.0001); vehicle inventory leads negatively by 1–2 months; lags are only partly stable. Macro features improve WAPE by ~1 pp, and the what-if re-runs the model.
- **Weakest parts?** MEA (young region, 36% WAPE), Electrical (chip history), launches (6 examples only), and the EV what-if total (index ≈ trend).

## Remaining ideas (if time allows)
- Visual polish pass in the browser; deploy to Streamlit Cloud.
- Seed-averaging the ensemble members; separate hurdle model for the 0.7% cycle mis-predictions.
- Formal MinT reconciliation comparison (bottom-up already beats naive at all levels).
