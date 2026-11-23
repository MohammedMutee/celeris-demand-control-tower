# 14 - Model Architecture Decision

Back: [[00 - START HERE]] · Plan: [[06 - Modeling & Validation Plan]] · Evidence: [[13 - Phase 2 Anomaly Findings]]
Date: 2026-11-23

## What the literature says (and why it applies here)
| Source | Finding | Relevance |
|---|---|---|
| **M5 competition** (Makridakis et al., 2022; Walmart, 42k series, intermittent) | Top solutions were **global LightGBM models** across all series, with Tweedie/L2 losses, many origins stacked, simple lag/rolling features and **ensembles** of several model variants. Per-series statistical models were clearly worse at the bottom level | ~3k SKU × region series with shared structure → a global GBM is the right workhorse |
| **M4** (Makridakis 2020) | Combinations / hybrids beat single methods; pure ML without a statistical backbone underperformed | Blend the GBM with a strong structural baseline |
| Intermittent demand (Croston 1972; Syntetos-Boylan SBA 2005; Teunter-Syntetos-Babai TSB 2011; ADIDA, Nikolopoulos 2011) | Separate *when* demand occurs from *how much* | We go further: *when* is deterministic here (ordering cycles, 99.3% holdout accuracy) |
| Direct vs recursive multi-step (Taieb et al. 2012) | Direct strategy avoids error compounding at long horizons and can use known-future covariates | 19-month horizon, and calendar, events, macro scenario and lifecycle are all known in advance → **direct, horizon as a feature** |
| Hierarchical forecasting (Hyndman 2011; Wickramasuriya et al. 2019, MinT) | Reconciliation gives coherence; MinT/WLS usually improves aggregate accuracy, bottom-level gains vary | Bottom-up (coherent by construction) vs WLS reconciliation, **decided on backtest** |
| Conformal / empirical-residual intervals (Stankevičiūtė 2021; quantile-calibration practice from M5 Uncertainty) | Calibrating residual quantiles on out-of-sample backtest errors by horizon gives honest coverage | P10/P90 from backtest residuals by horizon × series class, checked for ~80% coverage |

## Our architecture: "structural decomposition + global GBM"
The data has a structure most generic pipelines ignore:

```
demand(t) = monthly_rate(t) × months_covered(cycle) × order_expected(cycle, t)
```

- `order_expected`, `months_covered`: **known for every test row** from the detected ordering cycle (`test_cycles.parquet`)
- the hard, smooth part to forecast is `monthly_rate(t)`

So the model predicts the **monthly rate, scaled by the series level at the forecast origin**:
```
target = rate(t+h) / scale(origin)        scale = mean clean monthly rate, last 12 months
forecast = pred × scale × months_covered × order_expected
```
**Why:**
1. Removes the zeros (no zero-inflation problem) and the 3× order lumps of quarterly customers, so the GBM learns a smooth signal.
2. Scale-free target → one global model for series from 2 to 2,000 units/month.
3. Zero months in test are predicted exactly by the cycle, not guessed.

### Model ladder (each must earn its place on the rolling-origin backtest)
| # | Model | Role |
|---|---|---|
| B1 | Seasonal naive (same month last year) | Required baseline |
| B2 | Moving average 12 months | Required baseline |
| S1 | Croston / SBA / TSB (on raw demand) | Intermittent-demand methods (Advanced tier) |
| S2 | **Structural**: level × seasonal index × cycle | Strong, explainable statistical model |
| **M1** | **Global LightGBM, direct multi-horizon, on scaled rate** | Main model |
| E | Ensemble of M1 variants (+ S2), weights learned on backtest | Final point forecast if it wins |

### Post-processing layers (Expert)
1. Lifecycle: zero after EOL / before launch; lost accounts → 0; **launches via the handover curve learned from the 6 historical Replaces_SKU transitions**.
2. Forecast-window events not learnable from features (Orion NA shutdown) → explicit adjustment estimated from analogues.
3. P10/P50/P90 from calibrated backtest residual quantiles.
4. Hierarchy: bottom-up vs WLS reconciliation, chosen by backtest.

### Validation
Rolling origins **2023-03, 2023-09, 2024-03, 2024-09, 2025-03** (each has the full 19-month horizon inside history), plus partial folds 2025-09 and 2026-03. Metrics: WAPE (primary), bias, by family / region / horizon. Truth = unconstrained demand; data-loss rows excluded.
