import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data health", layout="wide")
from common import OUT, csv, js, page_header, sidebar_filters  # noqa: E402

page_header("Data health", "Every cleaning rule from the pipeline's machine-readable data-quality report: what was "
            "wrong, in which file, how many rows, and what we did. Validation: the stockout file is an independent "
            "record of shipments.")
sidebar_filters()
r = js("dq_report.json")
iss = pd.DataFrame(r["issues"])

c = st.columns(4)
c[0].metric("Rules triggered", len(iss))
c[1].metric("Row-level fixes logged", f"{iss['rows_affected'].sum():,}")
rec = r["files"]["inventory_stockouts.csv"].get("reconciliation", {})
if rec:
    c[2].metric("Shipments match stockout file: raw", f"{rec['shipped_eq_raw_units'] / rec['rows']:.1%}")
    c[3].metric("…after cleaning", f"{rec['shipped_eq_clean_units'] / rec['rows']:.2%}",
                delta=f"+{(rec['shipped_eq_clean_units'] - rec['shipped_eq_raw_units'])} rows")

a, b = st.columns([3, 2])
fig = px.bar(iss.sort_values("rows_affected"), x="rows_affected", y="rule", color="file", orientation="h",
             hover_data=["issue_type", "column", "action"], title="Rows affected by rule", log_x=True)
fig.update_layout(height=700, margin=dict(l=10, r=10, t=40, b=10))
a.plotly_chart(fig, use_container_width=True)
bt = iss.groupby(["file", "issue_type"])["rows_affected"].sum().reset_index()
fig2 = px.treemap(bt, path=["file", "issue_type"], values="rows_affected", title="Issues by file and type")
fig2.update_layout(height=700, margin=dict(l=10, r=10, t=40, b=10))
b.plotly_chart(fig2, use_container_width=True)

st.subheader("Rule catalogue")
st.dataframe(iss[["file", "rule", "issue_type", "column", "rows_affected", "action", "note"]],
             use_container_width=True, hide_index=True, height=420)

st.subheader("Before vs after cleaning")
rows = []
for f, info in r["files"].items():
    if "before" in info:
        rows.append(dict(file=f, rows_before=info.get("rows_in"), rows_after=info.get("rows_out"),
                         nulls_before=sum(info["before"].get("nulls", {}).values()),
                         nulls_after=sum(info.get("after", {}).get("nulls", {}).values()) if "after" in info else None))
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
tr = r["files"]["train.csv"]
lab = pd.DataFrame({"before": tr["before"]["distinct"], "after": tr["after"]["distinct"]}).dropna()
lab.index.name = "column (distinct values)"
st.caption("Distinct values per column in train.csv: raw columns before, canonical columns after")
st.dataframe(lab, use_container_width=True)

st.subheader("Drill into a rule")
rule = st.selectbox("Rule", iss["rule"].tolist())
d = pd.read_csv(OUT / "dq_issues.csv", usecols=["rule", "record_key", "before", "after", "action"])
st.dataframe(d[d["rule"] == rule].head(500), hide_index=True, use_container_width=True)
mc = csv("master_conflicts.csv")
st.subheader("Master data vs transactions")
st.dataframe(mc, hide_index=True, use_container_width=True)
