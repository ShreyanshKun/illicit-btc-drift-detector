"""Feature importance and per-prediction explanations.

All 165 features are anonymised: importance tells us WHICH columns matter, never WHAT
they mean in the real world.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance

from .config import SEED


def tree_importance(pipeline, cols):
    """Impurity (RF) / gain (XGB) importance of the final estimator in a pipeline."""
    imp = pipeline.named_steps["model"].feature_importances_
    return pd.Series(imp, index=cols).sort_values(ascending=False)


def perm_importance(pipeline, X, y, n_repeats=3):
    """Drop in PR-AUC when one feature is shuffled (model-agnostic, measured on held-out data)."""
    r = permutation_importance(pipeline, X, y, scoring="average_precision",
                               n_repeats=n_repeats, random_state=SEED, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def xgb_contributions(pipeline, X):
    """Per-row additive feature contributions (TreeSHAP values built into XGBoost).

    Returns (contribs DataFrame in log-odds, bias Series). Positive values push the
    prediction towards ILLICIT."""
    Xi = pipeline[:-1].transform(X)
    booster = pipeline.named_steps["model"].get_booster()
    c = booster.predict(xgb.DMatrix(Xi), pred_contribs=True)
    contrib = pd.DataFrame(c[:, :-1], columns=X.columns, index=X.index)
    return contrib, pd.Series(c[:, -1], index=X.index)
