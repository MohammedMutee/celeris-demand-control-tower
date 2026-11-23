"""Data-quality rule registry.

Every cleaning step calls ``DQLog.log(...)`` with a rule id, the affected row
indices and the action taken. The log is written as a machine-readable JSON
report plus a per-record CSV for drill-down in the dashboard.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd


@dataclass
class DQLog:
    issues: list[dict] = field(default_factory=list)
    records: list[pd.DataFrame] = field(default_factory=list)
    files: dict = field(default_factory=dict)

    def log(self, file: str, rule: str, issue_type: str, column: str, action: str,
            mask=None, df: pd.DataFrame | None = None, n: int | None = None,
            id_col: str | None = None, before=None, after=None, note: str = "") -> int:
        """Record one rule application. ``mask`` is a boolean Series aligned with ``df``."""
        examples = []
        if mask is not None:
            mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
            n = int(mask.sum())
            if n:
                rec = pd.DataFrame({
                    "file": file, "rule": rule, "issue_type": issue_type, "column": column,
                    "record_key": df.loc[mask, id_col].astype(str).values if id_col else df.index[mask].astype(str),
                    "before": (before[mask].astype(str).values if before is not None else ""),
                    "after": (after[mask].astype(str).values if after is not None else ""),
                    "action": action,
                })
                self.records.append(rec)
                examples = rec.head(5)[["record_key", "before", "after"]].to_dict("records")
        if not n:
            return 0
        self.issues.append({
            "file": file, "rule": rule, "issue_type": issue_type, "column": column,
            "rows_affected": int(n), "action": action, "note": note, "examples": examples,
        })
        return int(n)

    def file_summary(self, file: str, **kw):
        self.files.setdefault(file, {}).update(kw)

    def write(self, json_path, csv_path, run_id: str):
        report = {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_rules_triggered": len(self.issues),
            "files": self.files,
            "issues": self.issues,
        }
        json_path.write_text(json.dumps(report, indent=2, default=str))
        if self.records:
            pd.concat(self.records, ignore_index=True).to_csv(csv_path, index=False)
        return report


def profile(df: pd.DataFrame) -> dict:
    """Compact before/after profile of a frame."""
    return {
        "rows": int(len(df)),
        "nulls": {c: int(v) for c, v in df.isna().sum().items() if v},
        "distinct": {c: int(df[c].nunique()) for c in df.columns
                     if df[c].dtype == object and df[c].nunique() < 5000},
    }
