# MKS Hackathon: Celeris Demand Forecasting & Planning Control Tower

> Planning hub. Read this first, then follow the links in order.
> Created: 2026-11-22 · Project dir: `~/CODE/MKS Hackathon` · Data: `student_data/`

## The task in one paragraph
Celeris (fictional tier-1 auto supplier) has ~600 SKUs × 6 regions of monthly sales from 2019-01 to 2026-10, exported raw from their ERP and messy. We must:
1. **clean** the data,
2. **find and explain anomalies** (separate *errors* from *real events*),
3. **forecast unconstrained demand** (`Units_Sold`) for every row of `test.csv`: 58,219 rows, 2026-11 → 2028-05, a 19-month horizon,
4. build a **dashboard app** (training part + prediction part, shared filters, one continuous timeline).

Each workstream has Core / Advanced / Expert tiers. **Our target is Expert on all five.**

## Notes in this folder
| # | Note | What it covers |
|---|------|----------------|
| 01 | [[01 - Problem Understanding]] | Requirements, business rules, deliverables, tier checklist |
| 02 | [[02 - Data Findings (initial profile)]] | Verified facts from profiling every file |
| 03 | [[03 - Cleaning Plan]] | Workstream 1: every fix, in order |
| 04 | [[04 - Anomaly & Unconstrained Demand Plan]] | Workstream 2: anomaly register, stockout correction |
| 05 | [[05 - Features & Drivers Plan]] | Workstream 3: features, macro lead/lag, EV shift |
| 06 | [[06 - Modeling & Validation Plan]] | Workstream 4: baselines, global ML, backtest, P10/P50/P90, reconciliation |
| 07 | [[07 - Dashboard Plan]] | Workstream 5: 5 required + all 7 optional views |
| 08 | [[08 - Repo Architecture & Pipeline]] | Code layout, one-command pipeline, tests, outputs |
| 09 | [[09 - Execution Roadmap]] | Phases, order of work, definition of done |
| 10 | [[10 - Risks & Open Questions]] | What could go wrong, what we still need to decide |
| 11 | [[11 - Stand Out Strategy]] | Mapping our work to the 7 jury criteria |
| 12 | [[12 - Phase 0-1 Build Log & Cleaning Findings]] | What was built, every DQ rule with evidence |
| 13 | [[13 - Phase 2 Anomaly Findings]] | Anomaly register, unconstrained demand, cycles, customers |
| 14 | [[14 - Model Architecture Decision]] | Literature + why structural decomposition + global GBM |
| 15 | [[15 - Phase 3 Features & Drivers]] | Macro lead/lag, stability, feature set |
| 16 | [[16 - Phase 4-6 Modelling Results]] | Backtest, model selection, tuning, ensemble, intervals, final forecast |
| 17 | [[17 - Phase 7 Dashboard]] | Control tower pages and checks |
| 18 | [[18 - Results & Presentation Prep]] | Headline numbers, storyline, jury Q&A |



## Status board
- [x] Read problem statement + evaluation criteria
- [x] Initial data profile (see [[02 - Data Findings (initial profile)]])
- [x] Full plan written
- [x] Phase 0: environment + repo skeleton
- [x] Phase 1: cleaning pipeline + DQ report → [[12 - Phase 0-1 Build Log & Cleaning Findings]]
- [x] Phase 2: anomalies + unconstrained demand → [[13 - Phase 2 Anomaly Findings]]
- [x] Phase 3: features + drivers analysis → [[15 - Phase 3 Features & Drivers]]
- [x] Phase 4: baselines + backtest harness → [[16 - Phase 4-6 Modelling Results]]
- [x] Phase 5: global ML model + intermittent methods + ensemble (WAPE 28.3% vs naive 36.8%)
- [x] Phase 6: lifecycle / events / quantiles / hierarchy (84% interval coverage)
- [x] Phase 7: dashboard (5 required + 7 optional) → [[17 - Phase 7 Dashboard]]
- [ ] Phase 8: polish, tests, deploy, write-up, presentation → [[18 - Results & Presentation Prep]] (108 tests ✅; visual review + deploy pending)

## Guiding principles
1. **Every number in the dashboard comes from pipeline outputs.** No hand-typed figures.
2. **Every cleaning or anomaly decision is logged** with a rule ID, rows affected, and a reason. That log is the DQ report and the anomaly register.
3. **Validate the way we'll be scored.** Rolling-origin backtest, 19-month horizon, WAPE + bias, measured against *unconstrained* demand.
4. **Error vs. real event** is the central judgement call. Fix errors. Model or exclude real events, and explain which and why.
5. **Explain everything.** Judges score insight and communication as much as accuracy.
