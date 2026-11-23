"""Paths, constants and canonical label maps for the Celeris pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "student_data"
DATA = ROOT / "data"
CLEAN = DATA / "clean"
OUT = ROOT / "outputs"

FIRST_MONTH = "2019-01"
LAST_TRAIN_MONTH = "2026-10"
FIRST_TEST_MONTH = "2026-11"
LAST_TEST_MONTH = "2028-05"

REGIONS = ["North America", "Europe", "Asia Pacific", "India", "Latin America", "Middle East & Africa"]
CHANNELS = ["Direct OEM", "Tier-1 Distributor", "Aftermarket", "Export"]
CUSTOMERS = [
    "Apex Tier One", "Atlas Mideast Components", "Crestline Manufacturing", "Falcon Gulf Motors",
    "Helios Auto Parts", "Kestrel Mobility", "Meridian Assembly", "Nexon Motor Group",
    "Orion Driveline", "Quantum Drive Systems", "Solaris Automotive", "Vantara Auto",
]

# Keys are lower-cased, whitespace/punctuation-collapsed aliases.
REGION_ALIASES = {
    "north america": "North America", "n america": "North America", "na": "North America",
    "europe": "Europe", "eu": "Europe",
    "asia pacific": "Asia Pacific", "apac": "Asia Pacific",
    "india": "India",
    "latin america": "Latin America", "latam": "Latin America",
    "middle east africa": "Middle East & Africa", "middle east and africa": "Middle East & Africa",
    "mea": "Middle East & Africa", "me a": "Middle East & Africa",
}
CHANNEL_ALIASES = {
    "direct oem": "Direct OEM", "oem direct": "Direct OEM",
    "tier 1 distributor": "Tier-1 Distributor", "t1 distributor": "Tier-1 Distributor",
    "aftermarket": "Aftermarket", "after market": "Aftermarket",
    "export": "Export",
}

MEA_LAUNCH = "2022-01"
