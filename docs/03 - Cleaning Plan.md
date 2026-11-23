# 03 - Cleaning Plan (Workstream 1)

Back: [[00 - START HERE]] · Facts: [[02 - Data Findings (initial profile)]]

## Principle
Every transformation is a **named rule** (`DQ-xxx`) that logs: file, column, rule, rows affected, before → after sample, action. The log becomes `outputs/dq_report.json` (machine-readable) and feeds the **Data health** view.

## Order of operations (train.csv)
Order matters: normalise keys before deduping, dedupe before price reconciliation.

| Step | Rule | What | Notes |
|---|---|---|---|
| 1 | DQ-001 | Load everything as `str` | Never let pandas guess types on dirty data |
| 2 | DQ-010 | **Parse Date** in all 7 formats → `YYYY-MM` period | Unparseable → quarantine table |
| 3 | DQ-011 | Rebuild Year/Month from Date | Log the 268 mismatches |
| 4 | DQ-020 | **Normalise SKU_ID** to `AAA-NNN` (strip, upper, `_`→`-`, add the dash, zero-pad) | Validate against sku_master; unmatched → quarantine |
| 5 | DQ-021 | Region mapping dict (~31 → 6) | Case/space-insensitive, plus aliases (APAC, EU, LATAM, MEA, ME&A, N. America…) |
| 6 | DQ-022 | Region **NaN imputation** (952) | Infer from the (SKU, Customer, Channel) combination, which is near-unique per region; fall back to the Record_ID neighbourhood / Load_Timestamp batch. Log confidence |
| 7 | DQ-023 | Channel mapping (17 → 4) | |
| 8 | DQ-024 | Customer canonicalisation (220 → ~12) | Lowercase, collapse whitespace, strip `Inc.` etc., then a fuzzy match to the canonical list |
| 9 | DQ-025 | Product_Family / BU / Description from **sku_master** | Fill NaN; log disagreements as master-vs-transaction conflicts (Expert WS2) |
| 10 | DQ-030 | **Dedupe by Record_ID**, keep the latest Load_Timestamp | 3,682 |
| 11 | DQ-031 | Remaining duplicate **Date-SKU-Region** keys | Inspect first: same record with different IDs → drop; genuine split lines (different customers/channels) → sum. Decide on evidence |
| 12 | DQ-040 | Units_Sold parsing: remove `,`, `-` → NaN | |
| 13 | DQ-041 | **Negative units** (497) | Check if `abs(units)` × price ≈ |revenue| and the level fits the series → sign error, flip it. Otherwise flag as return/credit and treat as an anomaly |
| 14 | DQ-042 | **Units NaN** (3,362) | Recover from `Revenue / canonical price` when revenue exists; else mark missing (interpolate later only for the training target, never as 0) |
| 15 | DQ-050 | **Canonical price table**: per SKU × year, robust mode of price across regions | Rule 4: price is the same in all regions and changes on 1 Jan, plus the 2022-07→12 6% surcharge |
| 16 | DQ-051 | **Price errors**: ratio to canonical ≈100 → cents; ≈10 → decimal shift; constant ≈0.85–0.9 or ≈94/106 → currency, so convert | Detect clusters statistically; never hard-code a guess. Verify currency hypotheses by checking the ratio is constant over many rows |
| 17 | DQ-052 | Price 0 / NaN → canonical price | |
| 18 | DQ-053 | **Reconcile revenue**: decide which of units / price / revenue is wrong using the two that agree | Triangle logic: if units×canonical_price ≈ revenue → price was wrong; if revenue/price ≈ a plausible unit count and units is ×1000 → units were wrong |
| 19 | DQ-054 | **Unit-scale errors** (×1000 / ×100 / ×12 boxes→units?) | Ratio to the series median clusters at powers of 10 → error. Non-round, one-off spikes with consistent revenue → **real one-off order** (goes to the anomaly register, not fixed) |
| 20 | DQ-060 | **Missing rows vs zeros**: build the full Date × SKU × Region grid within each series' active window | Label each gap `not_ordered` (fits ordering cycle) / `data_loss` (LatAm ERP 2023-03→05, other gaps) / `pre_launch` / `post_eol` / `pre_market` (MEA before 2022-01). **Only `not_ordered` becomes 0** |
| 21 | DQ-070 | Rule 5 check: sales after EOL / before launch | Flag as master-data conflicts; decide case by case (wrong master date vs wrong transaction) |
| 22 | DQ-080 | Customer per series over time | Feeds the customer-switch / lost-account analysis in WS2 |

Output: `data/clean/sales_clean.parquet`, **one row per Date-SKU-Region** (plus Customer, Channel, Units_clean, Price_canon, Revenue_clean, flags), and `sales_lines_clean.parquet` at line level for drill-down.

## test.csv
- Same label normalisation (described as "clean keys", so verify anyway).
- Check test customers vs latest train customers for each series. **A different customer in test = a known future account switch.** That's a feature.

## macro_indicators.csv
- DQ-100 interpolate the 6 single NaNs (linear in time).
- DQ-101 PMI 4.79 → 47.9 (detected as > 5σ jump and a ×10 fix lands in the normal range).
- DQ-102 Auto_Production_YoY: detect the fraction vs percent switch and rescale to percent.
- DQ-103 general rolling-z outlier scan on every column, with a decimal-shift test.

## sku_master.csv
- Validate Replaces_SKU links (both exist, launch of new ≤ EOL of old).
- Check Lifecycle_Status vs dates (Discontinued must have an EOL ≤ 2026-10, etc.).
- Check `Size_Class` vs `SKU_Description` consistency.

## events_calendar.csv
- Normalise dates (`YYYY-MM` and `YYYY-MM-DD`) to month spans with a fraction of month covered (holidays are partial months).
- Parse `Scope_Value` like `Customer | Region`.
- Expand into a month × scope event table, so features can join on it.

## inventory_stockouts.csv
- Normalise SKU / Region the same way.
- Check identity: `Lost = Requested + Carried_In − Shipped − Backordered`; `Fill_Rate = Shipped/(Req+CarriedIn)`. Log violations.
- Check `Shipped_Units` == clean train Units_Sold for the same key. **Mismatches show which side has an error**, a useful second source of truth.
- Check chain consistency: backordered(t) == carried_in(t+1).

## Data-quality report (Expert)
`outputs/dq_report.json`:
```json
{"run_id": "...", "generated_at": "...",
 "files": {"train.csv": {"rows_in": 248248, "rows_out": ..., "issues": [
   {"rule": "DQ-030", "type": "duplicate_record", "column": "Record_ID",
    "rows_affected": 3682, "action": "keep_latest_load_timestamp", "examples": [...]}]}},
 "before_after": {"null_counts": {...}, "distinct_labels": {...}}}
```
Plus `outputs/dq_issues.csv` (one row per affected record, with rule IDs) for drill-down in the dashboard.

## Tests (pytest)
- Unit: each parser and mapper (date formats, SKU regex, region aliases, price-ratio classifier).
- Invariants on output: unique Date-SKU-Region, no NaN in keys, units ≥ 0, `units × price ≈ revenue` everywhere, all SKUs in master, no sales outside lifecycle (or flagged), all labels in canonical sets.
- Regression: row counts and key totals pinned in a snapshot.
