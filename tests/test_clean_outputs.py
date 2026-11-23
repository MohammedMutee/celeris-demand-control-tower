"""Invariant tests on pipeline outputs. Run the clean stage first (make pipeline)."""
import json

import numpy as np
import pandas as pd
import pytest

from celeris import config as C

pytestmark = pytest.mark.skipif(not (C.CLEAN / "sales_clean.parquet").exists(), reason="run pipeline first")


@pytest.fixture(scope="module")
def sales():
    return pd.read_parquet(C.CLEAN / "sales_clean.parquet")


@pytest.fixture(scope="module")
def master():
    return pd.read_csv(C.RAW / "sku_master.csv")


def test_one_row_per_key(sales):
    assert not sales.duplicated(["date", "sku", "region"]).any()


def test_no_null_keys_or_labels(sales):
    cols = ["date", "sku", "region", "family", "bu", "customer", "channel", "price"]
    assert not sales[cols].isna().any().any()


def test_canonical_labels(sales):
    assert set(sales.region) <= set(C.REGIONS)
    assert set(sales.channel) <= set(C.CHANNELS)
    assert set(sales.customer) <= set(C.CUSTOMERS)


def test_skus_in_master(sales, master):
    assert set(sales.sku) <= set(master.SKU_ID)


def test_date_range(sales):
    assert sales.date.min() >= C.FIRST_MONTH and sales.date.max() <= C.LAST_TRAIN_MONTH


def test_units_valid(sales):
    u = sales.units.dropna()
    assert (u >= 0).all() and (u % 1 == 0).all()


def test_revenue_reconciles(sales):
    ok = sales.units.notna()
    assert np.allclose(sales.revenue[ok], sales.units[ok] * sales.price[ok])


def test_price_same_across_regions(sales):
    n = sales.groupby(["sku", "date"]).price.nunique()
    assert (n == 1).all()


def test_family_bu_one_to_one(sales):
    assert (sales.groupby("family").bu.nunique() == 1).all()


def test_missing_rows_not_zero(sales):
    """Rule 3: rows we added for missing months must not be silently zero."""
    added = sales[sales.row_status == "missing_row"]
    assert added.units.isna().all()


def test_no_unexplained_scale_outliers(sales):
    med = sales[sales.units > 0].groupby(["sku", "region"]).units.transform("median")
    r = (sales.units / med).dropna()
    assert (r < 30).all()


def test_stockout_reconciliation():
    st = pd.read_parquet(C.CLEAN / "stockouts_clean.parquet")
    have = st.train_units_clean.notna()
    assert (st.train_units_clean[have] == st.Shipped_Units[have]).mean() > 0.999


def test_macro_clean():
    m = pd.read_parquet(C.CLEAN / "macro_clean.parquet")
    assert m.date.is_unique and not m.isna().any().any()
    assert m.ISM_Manufacturing_PMI.between(30, 70).all()
    assert len(m) == 113 and m.date.min() == "2019-01" and m.date.max() == "2028-05"


def test_test_keys_match_raw():
    t = pd.read_parquet(C.CLEAN / "test_clean.parquet")
    raw = pd.read_csv(C.RAW / "test.csv")
    assert len(t) == len(raw) == 58219
    assert (t.sku.values == raw.SKU_ID.values).all() and (t.region.values == raw.Region.values).all()


def test_dq_report_machine_readable():
    r = json.loads((C.OUT / "dq_report.json").read_text())
    assert r["issues"] and all({"rule", "rows_affected", "action", "file"} <= set(i) for i in r["issues"])
