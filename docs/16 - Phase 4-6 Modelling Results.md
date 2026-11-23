# 16 - Phase 4–6: Modelling, Validation & Final Forecast

Back: [[00 - START HERE]] · Architecture: [[14 - Model Architecture Decision]] · Features: [[15 - Phase 3 Features & Drivers]]
Date: 2026-11-23 · Code: `src/celeris/model/` · Stages: `make model` (≈33 min), `make forecast` (≈10 min)

## Validation design
- **Rolling-origin backtest, 7 origins:** 2023-03, 2023-09, 2024-03, 2024-09, 2025-03 (each with a full 19-month horizon) + 2025-09 and 2026-03 (partial horizons).
- **Leakage-safe:** for every origin, ordering cycles, the seasonal index and all features are rebuilt from data ≤ origin only.
- **Truth** = unconstrained demand. Data-loss rows are excluded; real one-off orders are *included*, since they're real demand and the model is charged for missing them.
- Metric: WAPE (primary) and bias, by fold / family / region / channel / horizon / hierarchy level.

## Model selection: how we got there
### 1. Model ladder (first 2 folds)
Croston / SBA / TSB ≈ 50% WAPE. They spread every order evenly and ignore the cycle → **the cycle-aware structure is essential.**

### 2. Objective × features × training-window sweep (3 folds, `outputs/experiments.csv`)
| Config | WAPE | Bias |
|---|---|---|
| L1, train from 2021 | **29.05%** | −2.0% |
| Tweedie | 29.3% | −2.1% |
| Huber | 29.4% | −0.3% |
| L2 | 29.5% | −0.1% |
| L1, from 2020-06 | 29.5% | −4.7% |
| L1 **without macro** | 30.2% | +1.0% |
| L2 **without macro** | 31.0% | +5.9% |
→ macro features help (0.7–1.5 pp). Starting in 2021 helps L1 (less COVID distortion).

### 3. Hyperparameter search (Optuna TPE, 14 trials, 2 folds, `outputs/tuning.json`)
Best (Tweedie, power 1.40, 255 leaves, min_data 100, feature_fraction 0.52, λ2 7.4) **28.61% vs 28.78% default**. Only 0.17 pp, so **capacity isn't the bottleneck; series noise is**. Good to say in the presentation: we tuned, and it showed the model was already near the data's limit.

### 4. Final 7-fold backtest + ensemble weights (`outputs/model_selection.json`)
| Model | Mean WAPE (7 folds) |
|---|---|
| MA12 | 50.3% |
| Croston / SBA / TSB | 50.3% / 50.4% / 50.2% |
| **Seasonal naive (baseline)** | **36.8%** |
| Structural (level × seasonality × cycle) | 30.7% |
| GBM Huber | 28.7% |
| GBM Tweedie (tuned) | 28.5% |
| GBM L1 | 28.4% |
| **Ensemble** = 0.9 × (½ Tweedie + ½ L1) + 0.1 × structural | **28.3%** |

**23% less error than seasonal naive.** The ensemble is best on **every one of the 7 folds** (29.9, 30.0, 27.8, 27.7, 29.0, 27.4, 26.2%). Huber got weight 0 in the grid search.

Bias check: the ensemble is 2–4% low everywhere. A multiplicative correction (×1.01…×1.04) *worsens* WAPE (28.28 → 28.36 → 28.46%), so the low bias is the expected median effect of optimising for WAPE. Left as is, and documented.

### Accuracy by slice (ensemble vs seasonal naive)
| Family | Ens. | Naive | | Region | Ens. | Naive |
|---|---|---|---|---|---|---|
| Brake & Safety | 26.9% | 35.5% | | North America | 25.6% | 33.6% |
| Powertrain | 28.0% | 36.4% | | Europe | 25.8% | 33.9% |
| Thermal | 28.1% | 36.3% | | Asia Pacific | 27.2% | 35.3% |
| Precision Machined | 28.3% | 37.2% | | India | 29.7% | 38.2% |
| Chassis | 28.8% | 36.6% | | Latin America | 31.0% | 40.6% |
| Hydraulic | 29.8% | 39.6% | | **MEA** (youngest region) | 36.2% | 46.8% |
| Electrical (chip history) | 31.3% | 40.2% | | | | |

**By horizon:** 25.7% at h=1 rising to ~29–31% at h=15–19. Degradation is gentle, which the direct multi-horizon design gives us.

### Hierarchy (bottom-up: forecasts add up exactly at every level)
| Level | Ensemble | Naive |
|---|---|---|
| Total company | **6.0%** | 7.9% |
| Business unit | 7.2% | 9.2% |
| Region | 6.6% | 8.5% |
| Family | 7.4% | 9.7% |
| Family × region | 8.8% | 11.3% |
| SKU | 17.6% | 23.1% |
| SKU × region (submission level) | 28.3% | 36.8% |
Coherence is by construction (bottom-up). A separate reconciliation step wasn't needed: the bottom-up totals already beat naive at every level by ~20–25%.

### Prediction intervals
- P10/P90 = P50 × empirical quantiles of actual/forecast, by horizon bucket × cycle class × volume tercile.
- **Calibrated on folds 1–4, validated on folds 5–7: 84.0% coverage** (target 80%; 10.9% below P10, 5.2% above P90). Slightly conservative → safe for procurement.
- Aggregating intervals in the dashboard: half-width = (1−λ)·√Σd² + λ·Σd, with λ calibrated so family × month aggregates hit 80% coverage in the backtest. (Summing P90s would overstate risk, and √Σd² alone would understate it.)

## Expert layers in the final forecast
| Layer | What | Evidence |
|---|---|---|
| **Launches** (PTC-029, PMP-020, THM-012) | Successor = handover curve × predecessor's per-region base. Curve: 10% → 33% → 52% → 61% → 88% → 98% plateau by month 7 | Learned from 6 historical transitions; leave-one-out WAPE 19–68% (tiny sample → wide band P10 = 0.5×, P90 = 1.6×) |
| **EOL** (PTC-005, PMP-003, THM-001) | Run-down to ~50% during the overlap, **0 after EOL** (156 test rows) | Rule 5, tested |
| **Lost accounts** | 16 series → 0 (304 rows) | No orders since 2025-04…2026-01 |
| **Orion NA retooling** 2027-07→08 | ×0.23 in Jul–Aug, ×1.16 in Sep (62 rows/month) | Analogue: Nexon strike −77% / +16% rebound |
| **Pre-buy** | None applied | Only seen before the large *announced* 2023 increase |
| **Prices** | 2026 list price × 1.045 (2027) × 1.04 (2028) | Events calendar |

## Final forecast (`outputs/submission.csv`, `forecast.parquet`)
- 58,219 rows, integers ≥ 0, keys identical to test.csv. **15.2% zero rows**, which are exactly the months the ordering cycle says no order.
- **19-month total: 2.80M units (P10 1.68M – P90 4.18M row-sum; the coherent aggregate band is narrower), $2.52B revenue.**
- Next 12 months 1.79M vs last 12 months 1.71M (**+4.6%**).
- By family vs the last-18-month run rate: Brake +7%, Electrical +6%, Hydraulic +5%, Precision +4%, Chassis +3%, Thermal +2%, Powertrain +1%.

## Drivers and scenarios
- Top features (gain share, averaged over the ensemble): same month 2 years ago (8%), seasonal index (8%), volume level (6%), family, customer, YoY growth, series age, powertrain, months since launch, 24-month mean.
- What-if (model re-run on a shifted macro path): **PMI −3 → −2.6%**, PMI +3 → +2.7%, capacity utilisation −3 pts → −0.9%, vehicle inventory +10 days → −0.9%.
- **Caveat:** EV index +10 → +2.9% total. The EV index trends with time, so the model partly reads it as growth. Flagged on the page; the reliable EV effect is the mix shift.

## Engineering notes / lessons
- Two background wait loops hung because `pgrep -f pattern` matched its own command line. Wait on PIDs or log markers instead.
- SHAP for all 58k rows on a 255-leaf model was far too slow (>20 min). Now the models are saved and the dashboard explains the selected series on demand (instant).
- `CELERIS_REUSE_MODELS=1 make forecast` re-runs post-processing without retraining (~30 s).
