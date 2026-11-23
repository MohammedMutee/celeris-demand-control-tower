# Documentation

Start here, then read in order.

| # | Document | What it covers |
|---|---|---|
| 1 | [Problem Understanding](01-problem-understanding.md) | Business context, deliverables, the 7 business rules and how each is enforced, what "unconstrained demand" means |
| 2 | [Data Profile](02-data-profile.md) | What was found in every raw file before cleaning |
| 3 | [Data Cleaning](03-data-cleaning.md) | Every data-quality rule with its evidence, rows affected and the independent validation (96.9 % → 99.98 %) |
| 4 | [Anomalies, Ordering Cycles and Unconstrained Demand](04-anomalies-and-unconstrained-demand.md) | Error vs real event, ordering cycles, stockout correction, measured effect of every calendar event, pre-buying, customer switches, EV shift |
| 5 | [Model Architecture](05-model-architecture.md) | Literature review and why a structural decomposition + global LightGBM |
| 6 | [Features and Macro Drivers](06-features-and-drivers.md) | Macro lead/lag study and its stability, the leakage-safe feature set |
| 7 | [Modelling, Validation and Final Forecast](07-modelling-results.md) | Rolling-origin backtest, model selection, tuning, ensemble, intervals, hierarchy, lifecycle and event layers, scenarios |
| 8 | [Planning Control Tower](08-dashboard.md) | Dashboard pages and how they meet the requirements |

Live dashboard: https://celeris-demand-control-tower.streamlit.app/
