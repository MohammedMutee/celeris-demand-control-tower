"""Model-component tests + submission validity (no training here; fast)."""
import numpy as np
import pandas as pd
import pytest

from celeris import config as C
from celeris.model.lifecycle import fit_curve
from celeris.model.quantiles import apply_quantiles, coverage, fit_quantiles


class _FakeP:
    def __init__(self, D):
        self.D, self.S, self.T = D, D.shape[0], D.shape[1]


class _FakeF:
    def __init__(self, D, O):
        self.P, self.O = _FakeP(D), O


def test_croston_constant_series():
    from celeris.model.models import croston_family
    D = np.full((1, 40), 10.0)
    for m in ("croston", "tsb"):
        f = croston_family(_FakeF(D, 30), method=m)
        assert np.allclose(f[0, 31:40], 10.0)


def test_croston_intermittent_rate():
    from celeris.model.models import croston_family
    D = np.tile([0.0, 0.0, 30.0], 30)[None, :]           # 30 every 3rd month = 10 / month
    f = croston_family(_FakeF(D, 80), method="croston")
    assert abs(f[0, 81] - 10.0) < 1.0


def test_seasonal_naive_uses_last_available_year():
    from celeris.model.models import seasonal_naive
    D = np.arange(60, dtype=float)[None, :]
    f = seasonal_naive(_FakeF(D, 40))
    assert f[0, 41] == D[0, 29] and f[0, 55] == D[0, 31]     # h=15 -> t-24


def test_quantiles_monotone_and_calibrated():
    rng = np.random.default_rng(0)
    n = 6000
    df = pd.DataFrame({"hb": rng.choice(["1-3", "13-19"], n), "cyc": "monthly",
                       "pred": rng.uniform(5, 100, n)})
    df["actual"] = df["pred"] * rng.lognormal(0, 0.3, n)
    q, edges = fit_quantiles(df.iloc[:3000], "pred")
    out = apply_quantiles(df.iloc[3000:], "pred", q, edges)
    assert (out["p10"] <= out["pred"]).all() and (out["p90"] >= out["pred"]).all()
    assert 0.74 < coverage(out)["coverage_p10_p90"] < 0.86


def test_handover_curve_shape():
    tab = pd.DataFrame({"new": np.repeat(["A", "B"], 13), "m": np.tile(range(13), 2),
                        "new_ratio": np.tile(np.linspace(0.1, 1.0, 13), 2),
                        "old_ratio": np.tile([0.6, 0.5, 0.4] + [0.0] * 10, 2)})
    c = fit_curve(tab)
    assert c["new"].is_monotonic_increasing and c["new"].iloc[0] < 0.2


sub = pytest.mark.skipif(not (C.OUT / "submission.csv").exists(), reason="run forecast stage first")


@sub
def test_submission_valid():
    s = pd.read_csv(C.OUT / "submission.csv")
    t = pd.read_csv(C.RAW / "test.csv")
    assert list(s.columns) == ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
    assert len(s) == len(t) == 58219
    assert (s[["Date", "SKU_ID", "Region"]].values == t[["Date", "SKU_ID", "Region"]].values).all()
    assert s["Units_Sold_Forecast"].notna().all() and (s["Units_Sold_Forecast"] >= 0).all()
    assert pd.api.types.is_integer_dtype(s["Units_Sold_Forecast"])


@sub
def test_rule5_no_sales_outside_lifecycle():
    s = pd.read_csv(C.OUT / "submission.csv")
    m = pd.read_csv(C.RAW / "sku_master.csv").set_index("SKU_ID")
    eol = s["SKU_ID"].map(m["EOL_Date"])
    launch = s["SKU_ID"].map(m["Launch_Date"])
    assert (s.loc[eol.notna() & (s["Date"] > eol), "Units_Sold_Forecast"] == 0).all()
    assert (s.loc[s["Date"] < launch, "Units_Sold_Forecast"] == 0).all()


@sub
def test_forecast_quantiles_ordered():
    f = pd.read_parquet(C.OUT / "forecast.parquet")
    assert (f["p10"] <= f["p50"] + 1e-9).all() and (f["p50"] <= f["p90"] + 1e-9).all()
