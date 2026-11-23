# Features and Macro Drivers

Date: 2026-11-23 · Code: `src/celeris/features/macro_leadlag.py`, `src/celeris/model/panel.py` · Stage: `make data` (drivers)

## Macro lead/lag study
**Method.** Demand = YoY log growth of family unconstrained demand (MEA excluded: its ramp is not macro). Indicator = YoY change (Auto production is already YoY). Cross-correlation at lags 0–12 on **pre-whitened** series (AR(3) fitted to the indicator, applied to both; removes spurious correlation between smooth trends). Granger F-test at the best lag. Stability: best lag in rolling 36-month windows + a pre/post-2022 split.

Outputs: `outputs/macro_ccf.csv` (every family × indicator × lag), `macro_leadlag.csv` (best lag, Granger p, post-2022 lag, stable flag), `macro_lag_stability.csv`.

### Findings
| Indicator | Relationship | Evidence |
|---|---|---|
| **ISM Manufacturing PMI** | **Leads every family by 1 month** (r 0.26–0.38) | Granger p < 0.0001 for all 7 families |
| **Vehicle inventory days** | **Leads negatively by 1–2 months**: high dealer stock → lower orders next month (r −0.31 to −0.40) | Granger p 0.002–0.02 for Electrical, Powertrain, Precision |
| **Industrial capacity utilisation** | Moves **at the same time** (r 0.63–0.76). After 2022 the best lag is **2 months**, matching "orders booked 2–3 months before delivery" | Strongest single relationship |
| New vehicle orders | Same month (r ~0.45–0.52), lag unstable after 2022 | — |
| Raw material cost | r ~+0.4 at lag 12, stable | Mechanism unclear (likely a base effect from the 2021 cycle). Reported, not over-interpreted |
| EV adoption index | **No family-level effect**. It shifts the *mix* within families (EV vs ICE SKUs), see [Anomalies, Ordering Cycles and Unconstrained Demand](04-anomalies-and-unconstrained-demand.md) | Granger p 0.66–0.89 |

### Is the lag structure stable?
**Only partly.**
- In rolling 36-month windows the best lag moves by 2–4 months (lag std: PMI 2.1, util 2.6, orders 3.8).
- Of 56 family × indicator pairs, 48 are significant over the full period, but only **16 are stable** (same sign, lag within ±2, significant both before and after 2022).
- The 2020 shock drives much of the full-period correlation.

**Consequence:** macro enters the model as candidate features (value at the target month, change vs origin, PMI and inventory days lagged 1). The GBM decides how much to use, and the backtest ablation decides whether they stay.

**Ablation result (3 folds):** macro features **help**. L1: 29.5% with vs 30.2% without. L2: 29.5% with vs 31.0% without. They also make the **what-if view a genuine model re-run**.

## Feature set (leakage-safe, direct multi-horizon)
One row = (series, origin o, horizon h). Everything is computed from data ≤ o; cycles and seasonality are **re-detected per backtest fold** on truncated history.

| Group | Features |
|---|---|
| Level & dynamics (÷ scale) | `l1..l3` (last 3 deseasonalised rates), `m3`, `m6`, `m24`, `yoy_growth`, `slope12`, `cv12`, `log_scale`, `nobs12` |
| Seasonality | `si_t`, `si_o` (family × region × month index), `ly_rate` / `ly2_rate` (same month 1–2 years before), `month` |
| Ordering cycle | `k_t` (months covered by the order), cycle-derived zero share, months since the cycle changed |
| Intermittency | `zero_share12`, `cv12` (with zero share this is the ADI/CV² information), `oneoff_recent` |
| Calendar events | `lny_frac` (APAC), `ramadan_frac` (MEA), `price_pct_year` |
| Lifecycle | `months_since_launch`, `months_to_eol`, `age`, `mea_age` |
| Macro (known scenario path) | 8 indicators at the target month (`mx_*`), change vs origin (`md_*`), PMI and inventory days at t−1 |
| Static categoricals | family, region, channel, customer, powertrain, size class |
| Horizon | `h` (1–19) |

**Target:** `y = rate(t) / scale(o)`. **Weight:** `scale × months_covered × train_weight`, so an L1 loss directly minimises WAPE in units. COVID-period targets get weight 0.3.
