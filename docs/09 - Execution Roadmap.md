# 09 - Execution Roadmap

Back: [[00 - START HERE]]

No hard deadline was given beyond "1-week project", so this is **phase-gated**: each phase has a definition of done (DoD), and we don't move on until it's met. A submission file exists from Phase 4 onward, so there's always something to hand in.

## Phase 0: Setup
- venv + dependencies, `git init`, repo skeleton per [[08 - Repo Architecture & Pipeline]], Makefile, README stub.
- **DoD:** `make test` runs (empty), `python -m celeris.pipeline --help` works.

## Phase 1: Cleaning (WS1, Core → Expert)
- All DQ rules from [[03 - Cleaning Plan]], with logging.
- Deep profiling of open questions: the remaining duplicate keys after dedupe, the price ratio clusters (cents/×10/currency/surcharge), negatives, unit-scale outliers, region NaN recovery, macro fixes.
- Stockout reconciliation vs train.
- **DoD:** `sales_clean.parquet` has a unique Date-SKU-Region; all invariant tests pass; `dq_report.json` + `dq_issues.csv` exist; every rule has rows-affected counts.

## Phase 2: Anomalies & unconstrained demand (WS2)
- Detectors, event join, pre-buy, holidays, structural breaks, ordering cycles, customer switches, master conflicts, noise-event checks.
- Unconstrained demand: listed + inferred stockouts, with the method validated against the listed ones.
- **DoD:** `anomaly_register.csv` with type/reason/action for every flag; `demand.parquet` with weights and flags; a short write-up of the top 10 findings in `12 - Build Log`.

## Phase 3: Features & drivers (WS3)
- Feature builder (direct multi-horizon, leakage-safe).
- Macro lead/lag study + stability + EV shift analysis.
- **DoD:** feature matrices built; `macro_leadlag.csv`; a heatmap reviewed and the conclusions written down.

## Phase 4: Baselines + backtest harness (WS4 Core)
- Rolling-origin harness, metrics by slice, seasonal naive + MA baselines.
- **First valid submission** (baseline) written and validated.
- **DoD:** `backtest_metrics.parquet` for the baselines; `submission.csv` passes the validator.

## Phase 5: Better models (WS4 Advanced)
- Stats + intermittent methods (statsforecast), global LightGBM (+ CatBoost), objective / target-scaling experiments, Optuna tuning, ensemble.
- **DoD:** a clear WAPE improvement over the baselines at every horizon bucket; experiment ladder logged; SHAP importance saved.

## Phase 6: Expert forecasting
- Launches via Replaces_SKU (validated on the 6 historical transitions), EOL run-down, forecast-window events (Orion, pre-buy, Helios), P10/P50/P90 with calibrated coverage, MinT reconciliation, what-if scenario re-scoring.
- **DoD:** final submission; coverage ≈ 80% on backtest; coherent at all hierarchy levels; scenario outputs exist.

## Phase 7: Dashboard (WS5)
- 5 required views → 7 optional views → polish (SHAP explain, methodology page).
- **DoD:** every checklist item in [[07 - Dashboard Plan]] ticked; runs from a fresh clone.

## Phase 8: Hardening & storytelling
- Test coverage of the pipeline, a fresh-clone dry run, Streamlit Cloud deploy, README with screenshots.
- `13 - Results` (final metrics) and `14 - Presentation & Q&A Prep` (deck outline, likely jury questions, the 5 decisions we're proudest of).
- **DoD:** all 5 workstreams at Expert; presentation rehearsed.

## Dependency graph
```
P0 → P1 → P2 → P3 → P4 → P5 → P6 → P8
                 ↘          ↘
                  P7 (training views can start after P2; prediction views after P4)
```
Dashboard work on the training-data views can run in parallel with modelling once P2 is done.
