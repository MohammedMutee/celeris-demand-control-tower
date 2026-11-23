# 08 - Repo Architecture & Pipeline

Back: [[00 - START HERE]]

## Layout (in `~/CODE/MKS Hackathon`)
```
MKS Hackathon/
├── student_data/                # raw inputs, read-only, never modified
├── src/celeris/
│   ├── config.py                # paths, constants, canonical label maps, backtest origins
│   ├── io.py                    # loaders / writers (parquet)
│   ├── dq.py                    # DQ rule registry + report builder
│   ├── clean/
│   │   ├── sales.py             # train/test cleaning (rules DQ-0xx)
│   │   ├── macro.py             # DQ-10x
│   │   ├── master.py            # sku_master checks
│   │   ├── events.py            # event expansion to month × scope
│   │   └── stockouts.py         # stockout reconciliation
│   ├── anomalies/
│   │   ├── detect.py            # outliers, scale clusters, event join
│   │   ├── demand.py            # unconstrained demand (listed + inferred stockouts)
│   │   ├── cycles.py            # ordering-cycle detection & changes
│   │   ├── customers.py         # switches / lost accounts
│   │   └── breaks.py            # structural breaks, pre-buy effects
│   ├── features/
│   │   ├── build.py             # direct multi-horizon matrix
│   │   ├── macro_leadlag.py     # CCF, Granger, rolling stability
│   │   └── ev.py
│   ├── models/
│   │   ├── baselines.py         # seasonal naive, MA
│   │   ├── stats.py             # ETS/Theta/Croston/SBA/TSB/ADIDA
│   │   ├── gbm.py               # LightGBM/CatBoost global
│   │   ├── lifecycle.py         # launches via Replaces_SKU, EOL run-down
│   │   ├── quantiles.py         # P10/P90
│   │   ├── reconcile.py         # bottom-up / MinT
│   │   └── ensemble.py
│   ├── backtest.py              # rolling-origin harness + metrics
│   ├── scenarios.py             # what-if re-scoring
│   └── pipeline.py              # CLI entry: run all stages
├── app/                         # Streamlit multipage dashboard
│   ├── Home.py
│   ├── pages/01_Executive.py … 12_Download.py
│   └── components/filters.py, charts.py
├── tests/                       # pytest: unit + invariants + regression
├── notebooks/                   # exploration only; nothing the pipeline depends on
├── data/                        # intermediate (gitignored): clean/, features/
├── outputs/                     # pipeline products: submission, dq_report.json, registers, metrics, dash/
├── Makefile
├── pyproject.toml               # deps pinned
└── README.md
```

## One command
```bash
make all        # = make pipeline test
make pipeline   # python -m celeris.pipeline --stages all
make test       # pytest -q
make dashboard  # streamlit run app/Home.py
```
`pipeline.py` stages: `clean → anomalies → demand → features → backtest → train → forecast → reconcile → quantiles → scenarios → dash_aggregates → validate_submission`. Each stage is cacheable (skips if inputs unchanged) and can run alone: `--stages backtest`.

## Dependencies (to install in Phase 0)
pandas, numpy, pyarrow, scikit-learn, **lightgbm**, catboost, statsforecast, hierarchicalforecast, statsmodels, ruptures, optuna, shap, rapidfuzz, plotly, streamlit, pytest. (Optional: neuralforecast + torch.)
Use a project venv (`uv venv` or `python -m venv .venv`).

## Reproducibility
- Fixed seeds, pinned versions, `run_id` stamped in every output.
- Raw data is never modified. All outputs are regenerated from the raw data with one command.
- `git init` the project so we can track progress (commit only when you say so).

## Memory budget
~2–3 GB free at profiling time. Use categoricals, float32, and Parquet. The direct multi-horizon matrix can be large (3k series × ~60 origins × 19 h ≈ 3.5M rows), so subsample origins or build per-horizon-bucket models if needed.
