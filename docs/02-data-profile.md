# Data Profile (raw files)

## File sizes
| File | Rows | Notes |
|---|---|---|
| train.csv | 248,248 | 15 cols, 2019-01 → 2026-10, 48.8 MB |
| test.csv | 58,219 | 3,068 SKU-region series, 2026-11 → 2028-05, no duplicate keys |
| macro_indicators.csv | 114 | 2019-01 → 2028-05 (covers the forecast window) |
| sku_master.csv | 601 | 0 duplicate SKUs |
| events_calendar.csv | 68 | 10 events fall inside the forecast window |
| inventory_stockouts.csv | 5,830 | 2019-01 → 2026-10; 74% are "Semiconductor allocation" |

## train.csv: issues confirmed
### Duplicates
- **3,682** repeated `Record_ID` (of which 2,946 are exact duplicate rows; the rest differ, so keep the latest `Load_Timestamp`).
- **3,687** duplicate `Date-SKU-Region` keys (still to be checked after Record_ID dedupe: there may be split lines that need summing rather than dropping).

### Dates
- 243,228 in `YYYY-MM`; ~5,000 in other formats: `2019-01-01`, `01/2019`, `2019/01`, `Jan 2019`, `Jan-2019`, `01-2019`.
- **268** rows where Year/Month disagree with Date (rule 1: trust Date).

### Labels
- **Sales_Channel:** 4 true values plus 13 variants (`tier-1 distributor`, `T1 Distributor`, `Direct-OEM`, `OEM Direct`, `After-market`, `Export ` with a trailing space…).
- **Region:** 6 true values plus ~25 variants (`APAC`, `EU`, `LATAM`, `N. America`, `ME&A`, `MEA`, `INDIA`, trailing spaces…), and **952 NaN** (recoverable from SKU + customer + channel combos, since one customer per SKU-region).
- **Product_Family:** 1,252 NaN (recover from SKU prefix / sku_master).
- **SKU_Description:** 996 NaN (recover from sku_master).
- **SKU_ID:** 2,659 raw unique values vs ~600 real ones. Variants: `THM008`, `PMP-55`, `HYD_006`, ` HYD-041` (leading space), `PTC004`. A regex normalisation to `AAA-NNN` with zero padding fixes these.
- **Customer_Name:** 220 raw values vs ~12 real customers. Variants: `Vantara Auto Inc.`, `Quantum  Drive Systems` (double space), `HELIOS AUTO PARTS`.
- Business_Unit looks clean (5 values, no NaN), but must be checked against family (each family belongs to exactly one BU).

### Units_Sold (target)
- **3,362 NaN**. Text values: `9,600`, `1,060` (thousands separators), `-`.
- **497 negatives** (returns or credit notes? sign errors? To decide: probably take the absolute value if |x| matches typical levels, otherwise flag).
- **35,805 zeros** (rule 3: real "no order" months).
- Median 36, max 10,700. **347 rows > 20× SKU median, 176 rows > 500×**, which points to a unit-scale error (×1000, e.g. "units vs thousands") or one-off bulk orders. Need to separate them.

### Price / revenue
- Price ratio to the SKU-year median shows clean clusters:
  - **×100** (569 rows): price entered in cents
  - **×10** (390 rows): decimal shift
  - **~×106 / ×94 / ×103**: probably a currency mix-up (e.g. INR or JPY instead of USD). To verify with exchange-rate-like constant ratios
  - **~×0.89 / ×0.86 / ×0.84**: probably EUR or GBP entered as USD, *or* the 2022-H2 6% surcharge, *or* annual price steps, to separate carefully
  - **ratio 0** (~36k): price is 0 or the comparison is broken. Check whether these are the zero-unit rows (price might be blank or 0 when no order)
  - **inf** (175): median price is 0 for that SKU-year
- **1,575 rows** where `Units × Price ≠ Revenue` (>1%). **383 Revenue NaN.**

### Series structure
- Test: 3,068 series. 58,219 rows ≈ 3,068 × 19 − 73, so a few series are cut by EOL / launch dates.
- **3 test SKUs not seen in train** (the Planned Launch SKUs: PTC-029, PMP-020, THM-012). **10 test series have no train history.**
- Zero-share per series: median 0%, 90th percentile 53%, so most series are smooth and a minority are intermittent or periodic ordering.

## macro_indicators.csv
- 6 single NaNs across different columns (2019-07, 2020-09, 2021-02, 2023-08, 2024-11). Interpolate.
- **PMI = 4.79 in 2023-05**: decimal shift (should be ~47.9).
- **Auto_Production_YoY_Pct** appears as a fraction (-0.015) in early rows and as a percent (11.6, -45.4) later, a unit inconsistency. Check where the switch happens and rescale.
- Values exist through 2028-05, so macro features are usable in the forecast window (they are given scenarios, not true leading info).

## sku_master.csv
- Lifecycle: 589 Active, 6 Discontinued, 3 EOL Announced, 3 Planned Launch.
- Powertrain: 440 Universal, 93 ICE, 68 EV. This is the EV-shift lever.
- **Replacement chains (9):**

| New SKU | Replaces | Launch | Status |
|---|---|---|---|
| ELA-015 | ELA-014 (EOL 2023-06) | 2023-03 | Active |
| HYD-021 | HYD-020 (EOL 2024-09) | 2024-06 | Active |
| PMP-018 | PMP-017 (EOL 2022-03) | 2022-01 | Active |
| PTC-027 | PTC-025 (EOL 2021-06) | 2021-03 | Active |
| PTC-028 | PTC-026 (EOL 2024-03) | 2023-12 | Active |
| THM-010 | THM-009 (EOL 2022-12) | 2022-09 | Active |
| **PTC-029** | PTC-005 (EOL **2027-05**) | **2027-03** | Planned Launch |
| **PMP-020** | PMP-003 (EOL **2027-08**) | **2027-06** | Planned Launch |
| **THM-012** | THM-001 (EOL **2027-12**) | **2027-10** | Planned Launch |

  The 6 historical transitions are **training data for how a cannibalising handover ramps**: overlap months, ramp-up curve, total-volume transfer ratio. We apply that learned curve to the 3 future transitions.
- Need to cross-check master vs transactions: sales before launch / after EOL, family/BU mismatches, description mismatches.

## events_calendar.csv: what matters
**History (model or exclude):**
- Annual price increases each January: 2.0, 2.0, 2.5, **5.0**, **9.0 (announced 2022-11-15)**, 4.5, 3.7, 3.7%. Expect **pre-buying in Nov–Dec before the big increases** and a dip in January.
- 2022-07 → 2022-12: temporary 6% raw-material surcharge (explains some price ratios ≈1.06).
- COVID: APAC 2020-02→03, Europe 2020-03→05, India/LatAm/NA 2020-04→06.
- **Chip shortage: Electrical Assemblies 2021-03 → 2022-09** (supply-constrained: shipments < demand).
- MEA market entry 2022-01 (structural start; MEA has fewer rows: 28k vs ~42k).
- **LatAm ERP migration 2023-03→05: "some transactions were not migrated"**. Missing rows here are data loss, not zero demand.
- Nexon Motor Group Europe strike 2023-10→11 (real demand dip).
- Red Sea: Export channel 2024-01→03 delays (timing shift).
- **Helios Auto Parts India: permanent capacity increase from 2025-09** (level shift up that must carry into the forecast).
- Holidays: Lunar New Year (APAC), Ramadan (MEA), NA July retooling, EU August shutdown.
- Noise events that probably don't affect demand: rebrand 2019-10, APAC sales director 2023-07, Hannover Messe 2024-04, CRM upgrade 2025-02 (maybe a data artefact?), Pune dealer conference 2026-09. **We must check each one, not assume.**

**Forecast window (must be reflected in forecasts):**
- Price increases 2027-01 (+4.5%) and 2028-01 (+4.0%), so possible pre-buy in Nov–Dec 2026/2027
- **Orion Driveline NA retooling shutdown 2027-07→08**
- Holidays: LNY 2027-02 / 2028-01–02, Ramadan 2027-02–03 / 2028-01–02, NA July 2027, EU Aug 2027
- 3 product launches + 3 EOLs (from the SKU master)

## inventory_stockouts.csv
- 5,830 rows. Reasons: Semiconductor allocation 4,314, Logistics delay 391, Supplier quality hold 381, Capacity constraint 377, Tooling breakdown 367.
- Gives `Requested_Units` (true demand), `Carried_In_Units`, `Shipped_Units` (should equal train Units_Sold: **a reconciliation check**), `Backordered_Units`, `Lost_Units`.
- "Known to be incomplete": we must detect unlisted stockout months (e.g. other Electrical Assemblies series during 2021-03→2022-09).
