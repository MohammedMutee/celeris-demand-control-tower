# Data Cleaning: Rules, Evidence and Validation

Date: 2026-11-22

## Summary
**Result:** the cleaning stage (`make data`) runs in ~12 s; 43 DQ rules triggered and logged; output invariants are covered by `tests/test_clean_outputs.py`.

### Outputs
| File | What |
|---|---|
| `data/clean/sales_clean.parquet` | **245,522 rows, one per Date-SKU-Region**, 2019-01 → 2026-10, with `units`, canonical `price`, `revenue`, `row_status`, `units_missing`, `dq_rules` (rule ids applied per row) and the raw `units_in/price_in/rev_in` kept for before/after |
| `data/clean/test_clean.parquet` | 58,219 rows, normalised keys (test was already clean) |
| `data/clean/price_table.parquet` | Canonical price per SKU × price period (year, plus a separate 2022-H2 surcharge period) |
| `data/clean/stockouts_clean.parquet` | Stockouts plus recomputed fill rate, backorder chain, and train reconciliation columns |
| `data/clean/macro_clean.parquet` | 113 months, no gaps |
| `data/clean/events_clean.parquet`, `event_months.parquet` | Events expanded to month × scope with the fraction of the month covered |
| `outputs/dq_report.json` | **Machine-readable DQ report**: per rule: file, type, column, rows affected, action, note, examples. Plus before/after profiles per file |
| `outputs/dq_issues.csv` | One row per affected record (rule, before, after) for dashboard drill-down |
| `outputs/master_conflicts.csv` | Master-data vs transaction conflicts |

### What we found, and the fix (with evidence)
Each was **diagnosed with evidence before fixing**. Nothing is a blind rule.

| Rule | Issue | Rows | Evidence → decision |
|---|---|---|---|
| DQ-010 | 7 date formats | 9,967 | Parsed all of them |
| DQ-011 | **Typo years (2062-01 = 2026-01)** | 30 | Year/Month and Load_Timestamp both say 2026. Load month = Date + 1 for 99.99% of rows, so it's a reliable fallback |
| DQ-012 | Year/Month ≠ Date | 246 | Load_Timestamp agrees with Date in **100%** of them, so rule 1 (trust Date) is confirmed by the data |
| DQ-020 | SKU format variants (`THM008`, `HYD_006`, ` PMP-55`) | 3,741 | 2,659 raw IDs → 601, all in master |
| DQ-021/023 | Region (31→6) / channel (17→4) variants | 5,992 / 3,745 | Alias maps (APAC, EU, LATAM, ME&A, `T1 Distributor`…) |
| DQ-024 | Customer typos (220→12) | 2,490 | Fuzzy match (rapidfuzz ≥ 80) to the 12 names that appear in test |
| DQ-025/026 | Family / description missing | 1,252 / 996 | Filled from sku_master |
| DQ-030 | Exact duplicate reloads | 2,946 | Dropped |
| DQ-031 | **Restated records** (same Record_ID, later load, *different units*) | 736 | Rule 2: kept latest. These are corrections, not just copies |
| DQ-032 | Duplicate Date-SKU-Region after dedupe | **0** | The 3,687 key duplicates were all Record_ID reloads |
| DQ-022 | **Region missing** | 942 | Recovered **all 942 with a unique match**: the region missing for that SKU in that month ∩ the region of that (SKU, customer, channel) combo |
| DQ-040 | `9,600`-style units | 44 | Thousands separator |
| DQ-041 | **Negative units** | 491 | Sign error, not returns: \|units\| ≈ series median (p10 0.57, p90 1.6), revenue negative too, evenly spread 2019–2026 → abs() |
| DQ-050 | **Price in cents (×100)** | 614 | All regions/years; ratio exactly 100 (106 / 94.3 near the 2022 surcharge). Revenue matched the canonical price, so only the price field was wrong |
| DQ-051 | **Unit-of-measure: India 2021-07→12 booked in packs of 10** | 385 | Price ×10 **and** units ×0.1 of normal level, revenue correct. Not a price error. Fix units ×10, price ÷10 |
| DQ-062 | ↳ refine with exact shipped qty | 36 | Raw value was `round(units/10)`, so ×10 is ±5. Where the stockout file has `Shipped_Units`, we use it |
| DQ-052 | **Currency: Europe 2022-01→03 at ×0.89** | 498 | Price *and* revenue both ×0.89 while units are correct, so local currency was keyed as USD |
| DQ-053 | Price 0 on rows with units > 0 | 295 | Canonical price |
| DQ-055a/b | **Units ×10 / ×100** | 284 / 74 | revenue/(units×price) = 0.1 / 0.01 **and** units are 10× / 110× the series level, so the units are wrong and revenue is right |
| DQ-056 | **Revenue ×10** | 425 | q = 10 but units at normal level, so revenue is wrong |
| DQ-057/058 | Small units/revenue disagreements | 7 / 481 | Keep whichever of units or revenue/price is an integer and closer to the series level. Mostly revenue noise, plus 295 rows where price 0 → revenue 0 |
| DQ-042 | Units missing, revenue present | 2,545 | units = revenue / price (integer check) |
| DQ-043 | Units missing, revenue 0 | 619 | No-order row → 0 |
| DQ-044 | **Units unrecoverable** | 380 | Kept as **missing**, not zero. 384 of these are **Latin America 2023-03→05 = the ERP migration event (EV-033)**: 128 rows/month lost |
| DQ-060 | **Missing months inside a series** | 930 | Scattered (~0.3 per series, all months/regions), so data loss, not ordering cycles. Added as rows with units = NaN (rule 3) |
| DQ-061 | ↳ restored from the stockout file | 26 | Shipped_Units known for those months |
| DQ-102 | Macro: Auto_Production_YoY is a **fraction in 2019** (-0.015) and a percent afterwards | 12 | ×100 |
| DQ-101 | Macro: **PMI 4.79 (2023-05)** | 1 | Decimal shift → 47.9 |
| DQ-104 | Macro: **2022-06 appears twice** (different PMI / orders) | 1 | Kept the row closest to the neighbour mean |
| DQ-100 | Macro single NaNs | 5 | Linear interpolation |
| DQ-110 | **Master: PMP-004 BU = "Power & Drive Systems"** | 1 | Every other PMP SKU and all transactions say Advanced Manufacturing (family → BU is 1:1), so the master is wrong |
| DQ-111 | **Master: BSS-018 sells from 2025-02, Launch_Date 2025-06** | 1 | Real sales in all regions at normal levels → master launch date too late; sales kept |
| DQ-131 | Stockouts: Fill_Rate_Pct only approximate | 1,725 >2pp off | Recomputed |
| DQ-120 | Events: open-ended (Helios India capacity) | 1 | Extended to end of forecast window |

### Validation that the fixes are right
The stockout file's `Shipped_Units` is a **second, independent source** for Units_Sold:
- raw train matches it in **5,652 / 5,830** rows (96.9%)
- **cleaned train matches in 5,829 / 5,830 (99.98%)**

Every mismatch in the raw data was one of our diagnosed error types (sign flips, ×100 units, India packs). That's strong evidence the cleaning is correct, not just plausible.

### Other observations
- **Price 0 is the convention on no-order rows** (35k rows with units 0 have price 0). Not an error; normalised to the canonical price.
- The **2022-H2 6% surcharge is real** (event EV-026) and is modelled as its own price period.
- **52 rows at ~10× normal with consistent revenue are real bulk orders**, not errors. They go to the [anomaly register](04-anomalies-and-unconstrained-demand.md).
- Raw "gap months" looked like 14.7k before date fixing. The 2062-typo rows stretched series spans to 2062. After fixing, only 930 real gaps remain. Order of operations matters.
- Ordering cycles show up as **explicit zero rows** (e.g. a series with a zero every 3rd month), while missing rows are random data loss. This is how we separate "not ordered" from "data loss".
- **Test contains 156 rows after EOL** (PTC-005, PMP-003, THM-001) → must be forecast 0 (rule 5).
- Test series with < 19 rows are exactly the 3 Planned Launch SKUs (10 series).
- One stockout row still disagrees with train after cleaning. Negligible; noted.
