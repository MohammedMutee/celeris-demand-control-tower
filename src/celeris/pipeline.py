"""One-command pipeline:  python -m celeris.pipeline [--stages clean,anomalies]"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

from celeris import config as C
from celeris.dq import DQLog


def stage_clean(run_id: str):
    from celeris.clean.reference import (check_master, clean_events, clean_macro, clean_stockouts,
                                         reconcile_stockouts)
    from celeris.clean.sales import clean_test, clean_train

    C.CLEAN.mkdir(parents=True, exist_ok=True)
    C.OUT.mkdir(parents=True, exist_ok=True)
    log = DQLog()
    master = pd.read_csv(C.RAW / "sku_master.csv", dtype=str)
    stock = clean_stockouts(pd.read_csv(C.RAW / "inventory_stockouts.csv"), log)
    train, prices = clean_train(pd.read_csv(C.RAW / "train.csv", dtype=str), master, stock, log)
    stock = reconcile_stockouts(stock, train, log)
    test = clean_test(pd.read_csv(C.RAW / "test.csv", dtype=str), master, log)
    macro = clean_macro(pd.read_csv(C.RAW / "macro_indicators.csv"), log)
    events, event_months = clean_events(pd.read_csv(C.RAW / "events_calendar.csv", dtype=str), log)
    conflicts = check_master(master, train, log)

    train.to_parquet(C.CLEAN / "sales_clean.parquet", index=False)
    test.to_parquet(C.CLEAN / "test_clean.parquet", index=False)
    prices.to_parquet(C.CLEAN / "price_table.parquet", index=False)
    stock.to_parquet(C.CLEAN / "stockouts_clean.parquet", index=False)
    macro.to_parquet(C.CLEAN / "macro_clean.parquet", index=False)
    events.to_parquet(C.CLEAN / "events_clean.parquet", index=False)
    event_months.to_parquet(C.CLEAN / "event_months.parquet", index=False)
    master.to_parquet(C.CLEAN / "sku_master.parquet", index=False)
    conflicts.to_csv(C.OUT / "master_conflicts.csv", index=False)
    rep = log.write(C.OUT / "dq_report.json", C.OUT / "dq_issues.csv", run_id)
    print(f"  clean: {len(train):,} train rows, {len(test):,} test rows, "
          f"{rep['n_rules_triggered']} rules triggered")


def stage_anomalies(run_id: str):
    from celeris.anomalies.run import run_anomalies
    run_anomalies(run_id)


def stage_drivers(run_id: str):
    import contextlib
    import io
    import warnings
    from celeris.features.macro_leadlag import run_leadlag
    warnings.filterwarnings("ignore")
    d = pd.read_parquet(C.CLEAN / "demand.parquet")
    m = pd.read_parquet(C.CLEAN / "macro_clean.parquet")
    with contextlib.redirect_stdout(io.StringIO()):
        full, best, roll = run_leadlag(d, m)
    full.to_csv(C.OUT / "macro_ccf.csv", index=False)
    best.to_csv(C.OUT / "macro_leadlag.csv", index=False)
    roll.to_csv(C.OUT / "macro_lag_stability.csv", index=False)
    print(f"  drivers: {int(best['significant'].sum())}/{len(best)} family-indicator pairs significant, "
          f"{int(best['stable'].sum())} stable")


def stage_select(run_id: str):
    """Full 7-fold backtest of the finalist configs, ensemble weights, quantile calibration."""
    from celeris.model.select import run_selection
    spec = json.loads((C.ROOT / "model_config.json").read_text())
    run_selection(spec["finalists"], spec["train_start"])


def stage_forecast(run_id: str):
    from celeris.model.forecast import run_forecast
    run_forecast(run_id)


STAGES = {"clean": stage_clean, "anomalies": stage_anomalies, "drivers": stage_drivers,
          "select": stage_select, "forecast": stage_forecast}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Celeris demand-forecasting pipeline")
    ap.add_argument("--stages", default="all", help=f"comma list of {list(STAGES)} or 'all'")
    a = ap.parse_args(argv)
    stages = list(STAGES) if a.stages == "all" else a.stages.split(",")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    for s in stages:
        t = time.time()
        print(f"[{s}] running ...")
        STAGES[s](run_id)
        print(f"[{s}] done in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
