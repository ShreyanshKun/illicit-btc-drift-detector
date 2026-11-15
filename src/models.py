"""Model factories and time-aware hyperparameter tuning.

Models (from simple to advanced):
  majority : trivial baseline (always predicts "licit")
  logreg   : linear, interpretable, class-weighted
  rf       : Random Forest, non-linear, class-weighted
  xgb      : gradient-boosted trees, class imbalance via scale_pos_weight
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import SEED
from .preprocessing import make_imputed_pipeline
from .splitting import expanding_window_folds

MODEL_NAMES = ["majority", "logreg", "rf", "xgb"]

# Small, explainable search grids (searched with expanding-window validation).
GRIDS = {
    "majority": [{}],
    "logreg": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "rf": [dict(max_depth=d, min_samples_leaf=l, max_features=f)
           for d, l, f in itertools.product((12, None), (1, 3), ("sqrt", 0.3))],
    "xgb": [dict(max_depth=d, learning_rate=lr, n_estimators=n)
            for d, lr, n in itertools.product((4, 6, 8), (0.05, 0.1), (200, 400))],
}

DEFAULT_PARAMS = {
    "majority": {},
    "logreg": {"C": 0.1},
    "rf": {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    "xgb": {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300},
}


def _clip(X):
    # Features are heavy-tailed; clip standardised values so outliers cannot dominate LR.
    return np.clip(X, -5, 5)


def make_model(name, params=None, scale_pos_weight=1.0, n_jobs=-1):
    params = dict(DEFAULT_PARAMS[name] if params is None else params)
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clip", FunctionTransformer(_clip)),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000,
                                         random_state=SEED, **params)),
        ])
    if name == "rf":
        return make_imputed_pipeline(RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=n_jobs,
            random_state=SEED, **params))
    if name == "xgb":
        return make_imputed_pipeline(XGBClassifier(
            tree_method="hist", scale_pos_weight=scale_pos_weight, subsample=0.8,
            colsample_bytree=0.8, eval_metric="aucpr", n_jobs=n_jobs,
            random_state=SEED, **params))
    raise ValueError(name)


def spw(y):
    """scale_pos_weight = (#licit / #illicit) computed on TRAINING labels only."""
    y = np.asarray(y)
    return float((y == 0).sum() / max(y.sum(), 1))


def fit_model(name, X, y, params=None):
    model = make_model(name, params, scale_pos_weight=spw(y))
    model.fit(X, y)
    return model


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def cv_score(name, df, cols, params=None):
    """Mean PR-AUC over expanding-window folds (train on past, validate on later steps)."""
    scores = []
    for tr, va in expanding_window_folds(df):
        m = fit_model(name, tr[cols], tr["label"], params)
        scores.append(average_precision_score(va["label"], predict_proba(m, va[cols])))
    return float(np.mean(scores)), scores


def tune(name, df, cols):
    """Grid search with expanding-window CV. `df` must contain train+validation steps only."""
    rows, best, best_score = [], None, -1.0
    for params in GRIDS[name]:
        mean, folds = cv_score(name, df, cols, params)
        rows.append({**{k: str(v) for k, v in params.items()}, "cv_pr_auc_mean": mean,
                     **{f"fold{i + 1}": s for i, s in enumerate(folds)}})
        if mean > best_score:
            best, best_score = params, mean
    res = pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False).reset_index(drop=True)
    return best, res
