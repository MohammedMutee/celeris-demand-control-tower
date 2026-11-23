"""Hyperparameter search (Optuna TPE) on time-based folds only. Objective: mean WAPE in units.

usage: python -m celeris.model.tune <n_trials>
"""
from __future__ import annotations

import json
import sys

import numpy as np
import optuna

from celeris import config as C
from celeris.model import models as M
from celeris.model.backtest import long_frame
from celeris.model.panel import Fold, Panel

ORIGINS = ["2024-03", "2025-03"]


def main(n_trials: int = 14):
    P = Panel()
    data = []
    for o in ORIGINS:
        F = Fold(P, o)
        fb = M.structural(F)
        fbm = np.where(np.isnan(fb), M.fallback_scale(F)[:, None] * F.SI * F.K * F.OE, fb)
        data.append((F, F.training_set(start="2021-01", step=2), F.prediction_set(), fbm))

    def objective(trial):
        params = dict(objective="tweedie", tweedie_variance_power=trial.suggest_float("tvp", 1.1, 1.6),
                      learning_rate=0.05,
                      num_leaves=trial.suggest_categorical("num_leaves", [31, 63, 127, 255]),
                      min_data_in_leaf=trial.suggest_categorical("min_data_in_leaf", [50, 100, 200, 500, 1000]),
                      feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
                      lambda_l2=trial.suggest_float("lambda_l2", 1e-2, 30, log=True))
        rounds = trial.suggest_categorical("rounds", [400, 700, 1000])
        ws = []
        for F, X, ps, fbm in data:
            m, feats = M.fit_gbm(X, params=params, rounds=rounds)
            df = long_frame(F.P, F.O, {"g": M.gbm_forecast(F, m, feats, ps)}, fbm)
            ws.append((df["g"] - df["actual"]).abs().sum() / df["actual"].sum())
        return float(np.mean(ws))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial(dict(tvp=1.3, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, lambda_l2=1.0,
                             rounds=700))                      # the default config as a reference point
    study.optimize(objective, n_trials=n_trials)
    res = dict(best_value=study.best_value, best_params=study.best_params,
               default_value=study.trials[0].value,
               trials=[dict(t.params, wape=t.value) for t in study.trials])
    (C.OUT / "tuning.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("best_value", "default_value", "best_params")}, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
