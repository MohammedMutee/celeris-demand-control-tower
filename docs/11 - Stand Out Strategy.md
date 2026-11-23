# 11 - Stand Out Strategy

Back: [[00 - START HERE]]

The jury scores 7 criteria: technical depth, practical insight, creativity and communication. How we win each one:

| # | Criterion | What the jury wants | Our edge |
|---|---|---|---|
| 01 | **Problem Understanding** | Clear grasp of the problem, a well-structured approach | Lead with the *planner's* problem: unconstrained demand for procurement, a P90 for safety stock. Show the business-rule → enforcement table ([[01 - Problem Understanding]]). Explicitly explain shipments ≠ demand, and missing ≠ zero |
| 02 | **Data Preparation** | Cleaning, preprocessing, feature engineering quality | Rule-ID'd, tested, one-command pipeline; machine-readable DQ report; the **revenue-triangle** method for deciding *which* field is wrong; stockout file cross-checked against train; ordering-cycle features |
| 03 | **Model Performance & Validation** | Effective ML approach, performance, validation method | Rolling-origin 19-month backtest that mimics the real forecast; a model ladder (naive → stats → intermittent → global GBM → ensemble) with WAPE at each step; metrics sliced by horizon/family/region; **calibrated** P10/P90 coverage; reconciliation compared bottom-up vs MinT |
| 04 | **Quality of Insights** | Relevance, accuracy, practical value | Quantified findings: pre-buy elasticity vs price-increase size; lost sales $ from stockouts; macro lead times per family (and whether they're stable); EV cannibalisation of ICE; Helios step-up; Orion shutdown impact; "checked, no effect" for noise events |
| 05 | **Dashboard & Visualization** | Clarity, usability, effectiveness | All 5 required + all 7 optional views, shared filters, one continuous timeline with bands, an exception list sorted by revenue impact, deployed |
| 06 | **Originality & Innovation** | Unique approach, creativity | (a) learning the **launch handover curve** from past Replaces_SKU transitions; (b) **validating stockout imputation** on the listed stockouts; (c) a what-if that **actually re-runs the model**; (d) SHAP "why this forecast" per series-month; (e) a planner-ready P90 procurement plan |
| 07 | **Solution Presentation** | Explain the method, key decisions, business impact | Story arc: messy ERP → trusted data → true demand → drivers → forecast with uncertainty → decisions. Put **business impact in $**: lost-sales recovery, inventory saved by using P50/P90 instead of naive, accuracy gain vs baseline |

## Signature moments to put in the presentation
1. "Before vs after" of one ugly series (cents price, ×1000 units, duplicate, stockout dip + catch-up) → a clean, unconstrained series.
2. The anomaly register counts: X errors fixed, Y real events modelled, Z excluded, each with a reason.
3. The macro lead/lag heatmap, plus "is it stable?"
4. Backtest ladder chart: WAPE baseline → final.
5. Launch handover: PTC-005 → PTC-029, forecast with a band.
6. Procurement plan: "If you order to P90, you cover demand 9 months in 10, at a cost of $X extra inventory."

## Things that lose marks (avoid)
- Filling missing rows with 0 (breaks rule 3).
- Training on shipments (breaks rule 7).
- Random K-fold CV on time series.
- Hard-coded numbers in the dashboard.
- Forecasts after EOL / before launch.
- Summing P90s up the hierarchy as if they were coherent.
- Assuming every event matters without testing.
