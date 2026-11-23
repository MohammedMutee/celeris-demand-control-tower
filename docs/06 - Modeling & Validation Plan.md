# 06 - Modeling & Validation Plan (Workstream 4)

Back: [[00 - START HERE]] · Previous: [[05 - Features & Drivers Plan]]

## Validation first: the backtest harness
Build this **before** any fancy model, so every idea gets a number.

### Rolling-origin backtest, 19-month horizon
- Last train month T = 2026-10. For a full 19-month horizon the origin must be ≤ 2025-03.
- **Origins:** 2023-03, 2023-09, 2024-03, 2024-09, 2025-03 (5 folds, each with a full 19-month horizon), plus 2025-09 and 2026-03 as **partial-horizon** folds (the most recent regime, horizons 1–13 and 1–7).
- Each fold: train on data ≤ origin, predict origin+1 … origin+19, and **only for series-months that exist** (mimic test.csv: rows exist where the series is active).
- Target = **unconstrained demand**. Score on clean, non-data-loss months.

### Metrics
- **WAPE** = Σ|y − ŷ| / Σy (primary; matches the dashboard requirement)
- **Bias** = Σ(ŷ − y) / Σy
- MAE, RMSE, sMAPE (secondary); **pinball loss + coverage** for P10/P90
- Sliced by: **family, region, channel, horizon (1–19), intermittency class, lifecycle state**
- Output: `outputs/backtest_metrics.parquet` (fold × model × slice), which is the source for the "Forecast & accuracy" view

### Simulating the test row pattern
Test has rows only for active months. Check in Phase 4 whether test rows include the zero/no-order months (likely yes: zeros are rows). If zero months appear as rows, predicting **when** a periodic orderer orders matters a lot.

## Model ladder
| Tier | Model | Purpose |
|---|---|---|
| Baseline 1 | **Seasonal naive** (same month last year) | Required baseline |
| Baseline 2 | **Moving average** (last 12 mo mean) / naive last value | Required baseline |
| Stat | **ETS / Theta** per series (statsforecast) | Strong classic, especially at aggregate levels |
| Intermittent | **Croston, SBA, TSB, ADIDA / IMAPA** | Advanced requirement; for intermittent/lumpy classes |
| **Global ML** | **LightGBM** (and CatBoost to compare), direct multi-horizon, Tweedie or log1p-L2 objective, sample weights | Main workhorse |
| Optional | Global NN (N-HiTS / TFT via neuralforecast) | Only if time allows and it beats LGBM on backtest |
| **Ensemble** | Weighted blend per intermittency class × horizon bucket, weights learned on backtest | Usually the winning step |

### Global ML design details
- One row = (series, origin, horizon h, target month). Features as in [[05 - Features & Drivers Plan]].
- Training set = stack of many origins (e.g. every month from 2021-01 to the fold origin), which gives lots of data.
- Objective: try `tweedie` (handles zeros and skew), `l2 on log1p`, and `l1`; select on backtest WAPE.
- **Target normalisation:** divide by the series scale (e.g. 12-month mean at origin) so one model fits all volume levels; multiply back.
- Sample weights: 0 for data-loss months, low for one-off events; optionally weight recent origins higher.
- Optionally a separate model or feature for "order occurs this month?" × "size if ordered" (**hurdle model**) for periodic orderers; compare against the single-model approach.
- Hyperparameters: Optuna on the backtest folds (time-split only).

## Expert items
### New launches without history (PTC-029, PMP-020, THM-012)
- Learn the **handover curve** from the 6 historical Replaces_SKU transitions: successor volume as a % of predecessor pre-EOL level by month since launch; overlap behaviour; total-volume conservation ratio.
- Forecast successor = curve × predecessor's forecasted run-rate (per region, with the region mix taken from the predecessor). The predecessor ramps down to its EOL.
- Validate by back-testing the method on the 6 historical transitions (leave-one-out).
- Also handle the other 7 test series with no train history (new regions for existing SKUs?) via analogues: SKU-in-other-regions × region share, or family × size-class average.

### EOL
- Hard zero after EOL_Date (rule 5); test.csv likely already omits those rows, but clamp anyway.
- **Run-down** behaviour before EOL learned from the historical discontinued SKUs (ELA-014, HYD-020, PMP-017, PTC-025, PTC-026, THM-009): is there a last-time-buy spike or a decline?

### Events in the forecast window
- Orion Driveline NA 2027-07→08 shutdown, price-increase pre-buys (Nov–Dec 2026, Nov–Dec 2027), holidays, Helios India step-up persisting. Handled through features. If the model can't learn rare events, apply an **explicit, documented adjustment layer** estimated from history (e.g. the Nexon strike effect).

### P10 / P50 / P90
- Option A: LightGBM quantile objectives (α = 0.1, 0.5, 0.9).
- Option B: **conformal / empirical residual quantiles** from the backtest, by horizon × intermittency class (scaled residuals). Well-calibrated and simple to explain.
- Pick by **coverage on backtest** (target: 80% of actuals inside P10–P90) and pinball loss. Enforce P10 ≤ P50 ≤ P90 and non-negative.

### Hierarchy-coherent forecasts
- Hierarchy: Total → BU → Family → SKU → SKU×Region; also Total → Region → SKU×Region; Channel and Customer as attributes.
- Bottom-up is coherent by construction. Better: **MinT (shrink) reconciliation** with hierarchicalforecast, combining the stronger aggregate-level models (ETS on family totals) with the bottom level. Compare WAPE at every level, bottom-up vs MinT.
- Quantiles at aggregate levels: sum sample paths (or approximate with correlated residual bootstrapping), not by summing P90s. Mention this explicitly; it's a credibility point.

## Submission
- P50 (or the point forecast that minimises the expected metric; if WAPE/MAE, the median is optimal) → round to non-negative integer.
- Validate: 58,219 rows, keys equal to test.csv exactly, no NaN, ints ≥ 0, no forecasts outside lifecycle.
- Save as `outputs/submission.csv`; keep versioned copies with backtest scores in `outputs/submissions/`.

## Experiment tracking
`outputs/experiments.csv`: model, config hash, fold, WAPE, bias, timing. Every idea gets a row, so we can show the "what improved what" ladder in the presentation.
