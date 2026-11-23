PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: all pipeline data model forecast test dashboard
all: pipeline test
pipeline:            ## everything: clean -> anomalies -> drivers -> select (backtest) -> forecast
	$(PY) -m celeris.pipeline --stages all
data:                ## cleaning + anomalies + drivers only (fast, ~1 min)
	$(PY) -m celeris.pipeline --stages clean,anomalies,drivers
model:               ## backtest + model selection (slow, ~30-40 min)
	$(PY) -m celeris.pipeline --stages select
forecast:            ## final training + submission (~5 min)
	$(PY) -m celeris.pipeline --stages forecast
test:
	$(PY) -m pytest -q tests
dashboard:
	cd app && ../.venv/bin/python -m streamlit run Home.py
