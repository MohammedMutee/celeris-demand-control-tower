# 10 - Risks & Open Questions

Back: [[00 - START HERE]]

## Open questions (for the user / organisers)
1. **What's the submission scoring metric?** Not stated. We assume WAPE (named for the backtest). If it's RMSE, the optimal point forecast is the mean, not the median, and the choice changes. Ask the organisers if possible.
2. **Team size / deadline?** Individual or a team of 2–3? Final submission date? Affects how ambitious Phase 6–7 can be.
3. **Submission format for the code/dashboard:** GitHub repo? Video? Live demo? Slides?
4. **Deploy target:** is Streamlit Community Cloud acceptable (makes the repo public), or local-only?

## Data / modelling risks
| Risk | Impact | Mitigation |
|---|---|---|
| Misclassifying real spikes as errors (or the reverse) | Distorted target and lost marks on the "error vs event" grading | Revenue triangle check + event join + power-of-10 test; document every borderline case in the register |
| Unlisted stockouts not detected | Forecast biased low for constrained series (especially Electrical Assemblies) | Validate the inference method on the listed stockouts; report the imputation error |
| Backorder catch-up months mistaken for demand spikes | Biased high | Subtract carried-in units in t+1 |
| Leakage in the multi-horizon features | Backtest looks great, test is bad | Strict "known at origin" rule; unit test that features at origin o use only data ≤ o |
| Macro forecast-window values are a scenario | Over-reliance on macro may hurt | Use macro only where lead/lag is significant and stable; ablation on the backtest |
| Regime shift after 2022 (post-COVID, post-chip) | Old data misleads | Recency weighting; test training windows starting 2021 vs 2019 |
| Periodic orderers (zeros in the test rows) | Large WAPE if order timing is wrong | Cycle phase features; hurdle model; compare against "spread evenly" forecasts. With WAPE, an even spread can sometimes beat a wrong-phase guess, so test both |
| New launches with no history | Large error on 3 SKUs | Handover curve learned from the 6 historical transitions; validated leave-one-out |
| Low free RAM (~2–3 GB) | OOM on feature matrices | float32, categoricals, origin subsampling, chunked builds |
| Dashboard scope creep | Late polish | Required 5 views first; optional in priority order: exceptions, procurement, stockouts, lifecycle, lead/lag, what-if, download |

## Decisions log (fill in as we go)
| Date | Decision | Why |
|---|---|---|
| 2026-11-22 | Target = unconstrained demand, not shipments | Rule 7 |
| 2026-11-22 | Streamlit + Plotly for the dashboard | Python end to end, fast, deployable |
| 2026-11-22 | Direct multi-horizon global GBM as the main model (to be confirmed by backtest) | 19-month horizon; avoids recursive error build-up; uses known-future covariates |
| 2026-11-22 | Fall back to Load_Timestamp − 1 month for out-of-range dates | Load month = Date + 1 for 99.99% of rows |
| 2026-11-22 | Missing months are data loss (imputed, weight 0), never zero | Scattered randomly; ordering cycles appear as explicit zero rows |
| 2026-11-22 | India 2021-H2 = packs-of-10 UoM error, not price error | units 0.1×, price 10×, revenue correct; confirmed by stockout file |
| 2026-11-22 | Chip-shortage unlisted months estimated as shipped / fill-rate | 11.4% WAPE in CV vs 33% for the level model |
| 2026-11-22 | Do NOT impute isolated unlisted stockouts | Best detector precision ~7.5%; flag + weight 0.5 instead |
| 2026-11-22 | One-off bulk orders excluded from training (weight 0) | Isolated, ≥2.5× expected, revenue consistent → real but not repeatable |
| 2026-11-22 | No pre-buy in the forecast window | Pre-buy only seen before the large *announced* 2023 increase |
| 2026-11-23 | Structural decomposition + global LightGBM (direct multi-horizon) | Literature (M5, Taieb 2012) + cycles found in Phase 2 |
| 2026-11-23 | Macro features kept | Ablation: −0.7 to −1.5 pp WAPE |
| 2026-11-23 | Ensemble 0.9 × (½ Tweedie + ½ L1) + 0.1 × structural | Best mean WAPE over 7 folds (28.3%), best on every fold |
| 2026-11-23 | No bias correction | ×1.01–1.04 worsens WAPE on the backtest |
| 2026-11-23 | Bottom-up hierarchy | Coherent by construction; beats naive at every level by 20–25% |
| 2026-11-23 | Orion NA shutdown ×0.23 Jul–Aug 2027, ×1.16 Sep | Nexon strike analogue |
| 2026-11-23 | SHAP on demand in the dashboard, not precomputed | Bulk TreeSHAP on 255-leaf trees took >20 min |
