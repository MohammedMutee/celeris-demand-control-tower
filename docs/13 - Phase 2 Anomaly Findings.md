# 13 - Phase 2: Anomalies, Ordering Cycles & Unconstrained Demand

Back: [[00 - START HERE]] · Plan: [[04 - Anomaly & Unconstrained Demand Plan]] · Previous: [[12 - Phase 0-1 Build Log & Cleaning Findings]]
Date: 2026-11-22 · Status: **done** · `make pipeline` (clean + anomalies) ≈ 40 s · **74/74 tests pass**

## Code
`src/celeris/anomalies/`
| Module | Does |
|---|---|
| `cycles.py` | Ordering-cycle detection + cycle-change segmentation (penalised binary segmentation, O(1) costs) |
| `demand.py` | Seasonal index, expected-demand model, **unconstrained demand** (5 steps), validation |
| `events.py` | Effect of every calendar event (DiD / YoY / step / ramp / data-loss), holiday regressions, pre-buy |
| `structure.py` | Customer switches, lost accounts, structural breaks (PELT), EV shift |
| `detect.py` | Outlier detection + **anomaly register** (errors *and* real events, with reason + action) |
| `run.py` | Stage runner → `make anomalies` |

## Code

`src/celeris/anomalies/`

| Module         | Description                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `cycles.py`    | Ordering-cycle detection and cycle-change segmentation using penalised binary segmentation with O(1) cost calculations.   |
| `demand.py`    | Seasonal index, expected-demand model, **unconstrained demand** (5-step process), and validation.                         |
| `events.py`    | Impact analysis of every calendar event (DiD, YoY, step, ramp, and data loss), holiday regressions, and pre-buy effects.  |
| `structure.py` | Customer switches, lost accounts, structural-break detection (PELT), and EV demand shifts.                                |
| `detect.py`    | Outlier detection and **anomaly register** covering both errors and genuine events, with reasons and recommended actions. |
| `run.py`       | Pipeline stage runner, exposed through `make anomalies`.                                                                  |

## Outputs
| File | Content |
|---|---|
| `data/clean/demand.parquet` | **Training target**: per Date-SKU-Region: `shipped`, **`demand`** (unconstrained), `demand_source`, `expected`, `level`, `si`, cycle fields, flags, **`train_weight`** |
| `outputs/anomaly_register.csv` | **17,624 entries**: type, classification, reason, action, linked event |
| `outputs/order_cycles.csv`, `order_cycle_segments.csv` | Current cycle per series + full cycle history |
| `data/clean/test_cycles.parquet` | **Expected order / no-order for every test row** (a forecasting feature) |
| `outputs/event_effects.csv` | Measured effect + verdict for all 68 events |
| `outputs/holiday_effects.csv`, `prebuy_effects.csv` | Holiday regressions; pre-buy by channel × year |
| `outputs/customer_changes.csv` | 23 customer switches + 16 lost accounts |
| `outputs/structural_breaks.csv`, `ev_shift.csv`, `ev_elasticity.csv` | Breaks; EV/ICE mix |
| `outputs/phase2_summary.json` | Machine-readable summary + validation metrics |

---

## Finding 1: Ordering cycles are deterministic (the biggest forecasting lever)
Zero months are **not random**: each customer orders on a fixed pattern.
| Current cycle | Series |
|---|---|
| monthly | 1,541 |
| monthly with 1 skipped month/yr | 757 |
| every 2nd month | 296 |
| quarterly | 242 |
| 2 of every 3 months | 229 |
| no orders (lost account) | 16 |

Zero months are **not random**: each customer orders on a fixed pattern.

| Current cycle                   | Series |
| ------------------------------- | -----: |
| monthly                         |  1,541 |
| monthly with 1 skipped month/yr |    757 |
| every 2nd month                 |    296 |
| quarterly                       |    242 |
| 2 of every 3 months             |    229 |
| no orders (lost account)        |     16 |

- **Validation:** fit on data ≤ 2025-10, predict order/no-order for the next 12 months → **99.3% accuracy** (vs 85% for "always orders").
- **49 cycle changes** detected (mostly 2022–2025), e.g. BSS-014 Europe quarterly → monthly from 2022-06.
- **Order size scales with the cycle, the monthly rate is conserved.** Across cycle changes the rate ratio is ~1.0–1.3 (just growth), while order size jumps ~3× for monthly → quarterly. So: **units = monthly rate × months covered × order_expected**.
- Bug fixed along the way: a 9-month window can't contain an *annual* skip month, so the first segmenter invented fake "cycle changes" in 2026-02. The penalised segmentation fixed it. A unit test pins it.
- Missing months are treated as **unknown**, never as zero (rule 3), and that's tested too.

## Finding 2: Unconstrained demand (training target, rule 7)
| Step | Rows | Method |
|---|---|---|
| UD-1 Listed stockout | 5,830 | demand = `Requested_Units` (listed requested ≈ 1.01× normal level; shipped ≈ 0.56×) |
| UD-2 Backorder catch-up | 1,048 | Month after a backorder ships 1.13× normal; minus the carried-in units → 0.88×. demand = shipped − backorder |
| UD-3 **Unlisted chip months** | 1,925 | Electrical Assemblies 2021-03→2022-09: only ~69% of series-months are listed, but **unlisted ones are equally constrained** (0.67× vs 0.65× normal). Estimated as shipped / fill-rate(month, region) |
| UD-4 Possible unlisted stockouts | 949 flagged | **Not imputed**, weight 0.5 (see below) |
| UD-5 Data loss | 1,309 | Imputed from the expected model, **train_weight 0** |

**Validation of the chip estimator** (5-fold by SKU on listed rows, where the truth is known):
| Estimator | WAPE |
|---|---|
| **A: shipped ÷ fill-rate(month, region)** | **11.4%** ← used |
| B: expected level × family demand index | 33.4% |
| geometric mean | 17.6% |

**Why we don't impute isolated unlisted stockouts:** shipment noise is large (normal-month shipped/expected: p5 = 0.51, p95 = 1.76). A threshold sweep on the listed stockouts gives at best ~**7.5% precision** (recall 13% at a 0.5% false-flag rate). Imputing would add more error than it removes, so those months are flagged and down-weighted instead. *Knowing when not to correct is part of the grade.*

**Totals:** shipped 11.72M units → unconstrained demand 11.89M. **Estimated lost sales: 119k units ≈ $119M**, of which **$93M is Electrical Assemblies** (semiconductor). Lost units by reason: semiconductor 55k listed + 24k inferred; logistics 11k; capacity 10k; supplier quality 10k; tooling 10k.

`train_weight`: 1.0 for 240,067 rows; 0.5 for 3,820 (inferred / possible stockouts); 0 for 1,635 (data loss + one-off orders).

## Finding 3: Which events matter (all 68 tested, none assumed)
Measured with methods matched to the event type: difference-in-differences vs a control group on ratio-to-trend (short events), YoY (COVID: every region was hit, so there's no clean control), a 12-month before/after step (permanent changes), level-DiD (long supply events), and a rows-lost count (system events).

| Event | Effect | Verdict |
|---|---|---|
| NA July retooling (every year) | **−41 to −44%**, then **+25% catch-up in August** | real, recurring |
| EU August shutdown (every year) | **−50 to −56%**, no catch-up | real, recurring |
| Lunar New Year (APAC, moving) | **−71% per full month of holiday** (t = −6.5) | real |
| Ramadan (MEA, moving) | **−28% per full month** (t = −6.0) | real |
| COVID 2020 | APAC −30%, Europe −57%, NA −54%, LatAm −50%, **India −69%** (−83% in Apr) | real one-off → down-weight |
| **Chip shortage (ELA)** | shipments down to **−36%** (Sep 2021), avg −11.5%, **+24% rebound** after | supply → corrected |
| MEA market entry 2022-01 | ramps **×7.5 in the first 12 months** | structural |
| Nexon strike, Europe, 2023-10→11 | **−77%** (z = −11), +16% next month | real one-off |
| **Helios India capacity, 2025-09 →** | **+24% permanent step** (z = 3.6) | must carry into the forecast |
| Red Sea, Export, 2024-01→03 | −19%, −16%, then **+19%** | timing shift, not lost demand |
| LatAm ERP migration 2023-03→05 | **384 rows lost vs 5 normal** | data error → imputed, weight 0 |
| CRM upgrade 2025-02 | 9 rows lost vs 14 normal | **no effect** |
| Rebrand, APAC sales director, Hannover Messe, Pune dealer conference, 9% price-increase announcement month | all within ±5%, \|z\| < 1 | **no effect** |

### Pre-buying before price increases
| Channel | 2022 (5%, not announced) | **2023 (9%, announced 15 Nov)** | 2024 (4.5%) |
|---|---|---|---|
| Tier-1 Distributor | Nov–Dec −7% | **+24%, Jan–Feb −13%** | −2% |
| Aftermarket | −4% | **+22%, −14%** | −1% |
| Direct OEM | −8% | +3% | +4% |
| Export | −10% | +4% | +3% |

**Only distributors and aftermarket pre-buy, and only before a large announced increase.** The forecast-window increases (+4.5% in 2027, +4.0% in 2028, no announcement) look like 2024 → **expect no pre-buy.**

## Finding 4: Accounts, master data, structure
- **23 customer switches** inside series (2022–2025). Level change after a switch varies (median +36%, range −48% to +115%), so the model must use post-switch history.
- **16 lost accounts:** series that stopped ordering in 2025-04→2026-01 (e.g. PTC-003 Europe, Solaris, was 114 units/month) **but still have rows in test** → forecast ~0 unless orders resume.
- Test customers = last train customer for every series (no switches announced in the forecast window).
- 25 train series are absent from test, all of them discontinued SKUs → consistent with rule 5.
- Master conflicts (Phase 1): PMP-004 BU wrong in master; BSS-018 sells 4 months before its master Launch_Date.
- Structural breaks (PELT): dominated by COVID (2020-04 drop, 2020-07/10 recovery) and the MEA ramp. Smaller steps: ELA +18% (2024-02), Hydraulics +23% (2024-06), Export +21% (2023-01).

## Finding 5: The EV shift
| Year | EV share | ICE share | Universal | EV index |
|---|---|---|---|---|
| 2019 | 0.0% | 24.1% | 75.9% | 42.7 |
| 2022 | 3.1% | 18.4% | 78.5% | 68.9 |
| 2024 | 9.2% | 13.9% | 76.9% | 88.7 |
| 2026 | 9.9% | 13.0% | 77.2% | 91.1 |
- Share elasticity to the EV index: **EV +4.3, ICE −1.2, Universal 0**. The EV index flattens in 2024–26 and so does the EV share. Phase 3 should check the EV index path in the forecast window to decide whether ICE keeps declining.

## Anomaly register: composition (17,624 entries)
| Classification | n | Action |
|---|---|---|
| `error` (fixed in WS1: currency, cents, ×10/×100 units, sign, UoM, date typos, missing region/rows) | 5,086 | fixed / imputed |
| `real_supply` (listed, inferred chip, catch-up) | ~8,800 | corrected to demand |
| `real_event` (in event windows, off-cycle / missed orders) | ~1,780 | model via features |
| `real_supply_uncertain` (possible unlisted stockout) | 949 | down-weight |
| `real_unexplained` dips | ~656 | down-weight |
| `real_one_off` **bulk orders** (isolated, ≥2.5× expected, median 5.3×, revenue consistent) | ~326 | exclude from training, keep in history view |
| `real_level_shift` | ~19 | model |

## Implications for Phase 3–6 (forecasting)
1. **Use `test_cycles.parquet`:** forecast = monthly rate × months covered × order_expected. Zero months in test are predictable.
2. Target = `demand`, sample weight = `train_weight`.
3. Features: holiday fractions (LNY −71%/month, Ramadan −28%), NA July / EU August (+ NA August catch-up), Helios step, MEA age, EV × powertrain.
4. **Forecast-window events:** Orion NA retooling 2027-07→08. Use the Nexon strike (−77%, +16% rebound) and NA retooling (−44%, +25%) as analogues. **No pre-buy expected.**
5. Lost accounts → 0. EOL rows (156) → 0. Launches via the Replaces_SKU handover curve.
6. COVID 2020 and the chip period: keep as history but consider down-weighting / starting the training window in 2021+ (to test in the backtest).
