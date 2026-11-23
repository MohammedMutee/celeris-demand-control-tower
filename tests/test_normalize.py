import pandas as pd
import pytest

from celeris.clean.normalize import (normalize_channel, normalize_customer, normalize_region, normalize_sku,
                                     parse_month, to_number)


@pytest.mark.parametrize("raw,exp", [
    ("2019-01", "2019-01"), ("2019-01-01", "2019-01"), ("01/2019", "2019-01"), ("2019/01", "2019-01"),
    ("Jan 2019", "2019-01"), ("Jan-2019", "2019-01"), ("01-2019", "2019-01"), ("Dec 2024", "2024-12"),
    ("garbage", None), ("2019-13", None), (None, None),
])
def test_parse_month(raw, exp):
    assert parse_month(raw) == exp


@pytest.mark.parametrize("raw,exp", [
    ("THM008", "THM-008"), ("PMP-55", "PMP-055"), ("HYD_006", "HYD-006"), (" HYD-041", "HYD-041"),
    ("bss-1", "BSS-001"), ("PTC004", "PTC-004"), ("XX-1", None),
])
def test_normalize_sku(raw, exp):
    assert normalize_sku(raw) == exp


@pytest.mark.parametrize("raw,exp", [
    ("APAC", "Asia Pacific"), ("Asia-Pacific", "Asia Pacific"), ("asia pacific", "Asia Pacific"),
    ("EU", "Europe"), ("EUROPE", "Europe"), ("Europe ", "Europe"), ("LATAM", "Latin America"),
    ("LatAm", "Latin America"), ("Latin-America", "Latin America"), ("N. America", "North America"),
    ("ME&A", "Middle East & Africa"), ("MEA", "Middle East & Africa"),
    ("Middle East and Africa", "Middle East & Africa"), ("INDIA", "India"), ("Mars", None),
])
def test_normalize_region(raw, exp):
    assert normalize_region(raw) == exp


@pytest.mark.parametrize("raw,exp", [
    ("tier-1 distributor", "Tier-1 Distributor"), ("T1 Distributor", "Tier-1 Distributor"),
    ("Tier 1 Distributor", "Tier-1 Distributor"), ("OEM Direct", "Direct OEM"), ("Direct-OEM", "Direct OEM"),
    ("After-market", "Aftermarket"), ("AfterMarket", "Aftermarket"), ("Export ", "Export"),
])
def test_normalize_channel(raw, exp):
    assert normalize_channel(raw) == exp


@pytest.mark.parametrize("raw,exp", [
    ("Vantara Auto Inc.", "Vantara Auto"), ("Quantum  Drive Systems", "Quantum Drive Systems"),
    ("HELIOS AUTO PARTS", "Helios Auto Parts"), ("Aex Tier One", "Apex Tier One"),
    ("CrestlineManufacturing", "Crestline Manufacturing"), ("Atlas Mideast omponents", "Atlas Mideast Components"),
])
def test_normalize_customer(raw, exp):
    assert normalize_customer(raw)[0] == exp


def test_to_number():
    s = to_number(pd.Series(["9,600", "-", "12", None, "1,060"]))
    assert s.tolist()[:1] == [9600] and pd.isna(s[1]) and s[2] == 12 and pd.isna(s[3]) and s[4] == 1060
