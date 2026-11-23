# Celeris Demand Forecasting & Planning Control Tower

Forecast of unconstrained monthly demand for every SKU × region in `test.csv` (2026-11 → 2028-05), built on a
cleaned, stockout-corrected history, plus a Streamlit control tower for planners and managers.

**Live dashboard:** https://celeris-demand-control-tower.streamlit.app/  
**Documentation:** [`docs/`](docs/README.md)

## Repository contents
| Path | What |
|---|---|
| `outputs/submission.csv` | **The submission**: 58,219 rows, `Date, SKU_ID, Region, Units_Sold_Forecast` |
| `src/celeris/` | Pipeline: cleaning, anomalies, drivers, models, backtest, forecast |
| `app/` | Streamlit control tower (12 pages + home) |
| `outputs/`, `data/clean/` | Pipeline outputs the dashboard reads (committed so the app runs without re-running the pipeline) |
| `docs/` | Write-up: problem understanding, data profile, every cleaning rule, anomaly findings, model architecture, drivers, results, dashboard. Start at [`docs/README.md`](docs/README.md) |
| `tests/` | 108 tests (normalisers, cycle detection, output invariants, submission validity, every dashboard page) |

## Instructions to run

**Option 1 – live dashboard (no setup):** open https://celeris-demand-control-tower.streamlit.app/ . Sidebar filters (family, SKU, region, channel, customer) apply across all pages.

**Option 2 – dashboard locally (~2 min):**
```bash
git clone https://github.com/MohammedMutee/celeris-demand-control-tower.git
cd celeris-demand-control-tower
python3 -m venv .venv && source .venv/bin/activate      # Python 3.12
pip install -r requirements.txt
streamlit run app/Home.py                               # http://localhost:8501
```
The forecast is `outputs/submission.csv` (58,219 rows).

**Option 3 – reproduce everything from the raw data:** see Setup and Run below.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-pipeline.txt     # dashboard only: requirements.txt
```
The raw competition files are not redistributed here: copy them into `student_data/`
(`train.csv`, `test.csv`, `macro_indicators.csv`, `sku_master.csv`, `events_calendar.csv`,
`inventory_stockouts.csv`, `sample_submission.csv`). They are only needed to re-run the pipeline;
the dashboard runs from the committed outputs.

## Run
```bash
make pipeline     # everything below, in order (~45 min, dominated by the 7-fold backtest)
make data         # clean -> anomalies -> drivers               (~1 min)
make model        # rolling-origin backtest, ensemble weights, P10/P90 calibration   (~35 min)
make forecast     # final training on all history, submission, scenarios           (~5 min)
make test         # pytest: normalisers, cycles, invariants, submission validity
make dashboard    # control tower at http://localhost:8501  (or: streamlit run app/Home.py)
```

## Pipeline stages
| Stage | Code | Main outputs |
|---|---|---|
| `clean` | `src/celeris/clean/` | `data/clean/sales_clean.parquet` (one row per Date-SKU-Region), `outputs/dq_report.json`, `dq_issues.csv` |
| `anomalies` | `src/celeris/anomalies/` | `data/clean/demand.parquet` (unconstrained demand + train weights), `anomaly_register.csv`, ordering cycles, event effects |
| `drivers` | `src/celeris/features/macro_leadlag.py` | `macro_ccf.csv`, `macro_leadlag.csv`, `macro_lag_stability.csv` |
| `select` | `src/celeris/model/select.py` | `backtest_long.parquet`, `backtest_metrics.csv`, `backtest_hierarchy.csv`, `quantile_table.csv`, `model_selection.json` |
| `forecast` | `src/celeris/model/forecast.py` | **`submission.csv`**, `forecast.parquet` (P10/P50/P90 + revenue), `scenarios.parquet`, `feature_importance.csv` |

## Method in one paragraph
Customers order on deterministic cycles (monthly, every 2nd month, quarterly, …), so
`demand = monthly rate × months covered by the order × order expected this month`. Cycles are detected per
series (99.3 % holdout accuracy on order / no-order). A global LightGBM ensemble (Tweedie + L1; Huber was tested and got weight 0) forecasts the
monthly rate 1–19 months ahead with a direct multi-horizon design, blended with a structural
level × seasonality model. Lifecycle rules (EOL, launches via a handover curve learned from past replacements,
lost accounts), forecast-window events and calibrated P10/P90 bands are applied on top. Everything is validated
with a rolling-origin backtest (7 origins, 19-month horizon).

Model and tuning experiments: `src/celeris/model/experiment.py`, `tune.py` → `outputs/experiments.csv`, `tuning.json`.

## Results (rolling-origin backtest, 7 origins, up to 19 months ahead)
| Model | WAPE |
|---|---|
| Seasonal naive | 36.8 % |
| Croston / SBA / TSB | ~50 % |
| Structural (level x seasonality x ordering cycle) | 30.7 % |
| **Ensemble (LightGBM Tweedie + L1, + structural)** | **28.3 %** |

Company total 6.0 %, family 7.4 %, SKU x region 28.3 %. P10-P90 coverage 84 % on held-out folds.
19-month forecast: 2.80M units, $2.52B revenue.
