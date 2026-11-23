"""Pure normalisation functions (dates, SKU ids, labels). Unit-tested in tests/."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

from celeris.config import CHANNEL_ALIASES, CUSTOMERS, REGION_ALIASES

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_month(s: str) -> str | None:
    """Parse any of the observed date formats to 'YYYY-MM'. Returns None if unparseable."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/]\d{1,2})?", s)          # 2019-01, 2019/01, 2019-01-01
    if m:
        y, mo = int(m[1]), int(m[2])
    else:
        m = re.fullmatch(r"(\d{1,2})[-/](\d{4})", s)                       # 01/2019, 01-2019
        if m:
            y, mo = int(m[2]), int(m[1])
        else:
            m = re.fullmatch(r"([A-Za-z]{3})[A-Za-z]*[\s\-](\d{4})", s)    # Jan 2019, Jan-2019
            if not m or m[1].lower() not in _MONTHS:
                return None
            y, mo = int(m[2]), _MONTHS[m[1].lower()]
    if not (1 <= mo <= 12 and 2000 <= y <= 2100):
        return None
    return f"{y:04d}-{mo:02d}"


def normalize_sku(s: str) -> str | None:
    """' hyd_6' / 'THM008' / 'PMP-55' -> 'HYD-006' / 'THM-008' / 'PMP-055'."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    m = re.fullmatch(r"([A-Za-z]{3})[\s\-_]*(\d{1,3})", str(s).strip())
    if not m:
        return None
    return f"{m[1].upper()}-{int(m[2]):03d}"


def _key(s: str) -> str:
    s = str(s).lower().replace("&", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_region(s) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    k = _key(s)
    if k in REGION_ALIASES:
        return REGION_ALIASES[k]
    k2 = k.replace(" ", "")
    for a, v in REGION_ALIASES.items():
        if a.replace(" ", "") == k2:
            return v
    return None


def normalize_channel(s) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    return CHANNEL_ALIASES.get(_key(s))


_CUST_KEYS = {c: re.sub(r"[^a-z]", "", c.lower()) for c in CUSTOMERS}


def normalize_customer(s) -> tuple[str | None, float]:
    """Fuzzy-match a raw customer name to the 12 canonical names. Returns (name, score)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None, 0.0
    raw = re.sub(r"\binc\b\.?", "", str(s), flags=re.I)
    k = re.sub(r"[^a-z]", "", raw.lower())
    best = process.extractOne(k, _CUST_KEYS, scorer=fuzz.ratio)
    name, score = best[2], best[1]
    return (name, score) if score >= 80 else (None, score)


def map_unique(series: pd.Series, fn) -> pd.Series:
    """Apply ``fn`` once per distinct value (fast for low-cardinality columns)."""
    u = series.dropna().unique()
    lut = {v: fn(v) for v in u}
    return series.map(lut)


def to_number(s: pd.Series) -> pd.Series:
    """'9,600' -> 9600, '-' / '' -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip()
                         .replace({"-": None, "": None, "nan": None}), errors="coerce")
