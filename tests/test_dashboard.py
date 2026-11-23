"""Headless smoke test: every dashboard page runs without an exception, with and without filters."""
from pathlib import Path

import pytest

from celeris import config as C

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py"] + sorted(p.name for p in (APP / "pages").glob("*.py"))
ready = pytest.mark.skipif(not (C.OUT / "forecast.parquet").exists() or not (C.OUT / "scenarios.parquet").exists(),
                           reason="run the pipeline first")


def _run(page, state=None):
    import sys
    from streamlit.testing.v1 import AppTest
    sys.path.insert(0, str(APP))
    path = APP / page if page == "Home.py" else APP / "pages" / page
    at = AppTest.from_file(str(path), default_timeout=180)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    return at


@ready
@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = _run(page)
    assert not at.exception, f"{page}: {at.exception}"


@ready
@pytest.mark.parametrize("page", PAGES)
def test_page_runs_with_filters(page):
    at = _run(page, {"_f_family": ["Powertrain Components"], "_f_region": ["Europe"], "_f_sku": ["PTC-005"]})
    assert not at.exception, f"{page}: {at.exception}"
