"""Data-quality checks and leakage-safe preprocessing pipelines."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RAW_FEATURE_COLS


def quality_report(master, edges):
    """Return a dict of data-quality facts (missing, duplicates, dtypes, edges...)."""
    X = master[RAW_FEATURE_COLS]
    t = master.set_index("txId")["time_step"]
    same_step = (t.loc[edges["txId1"]].values == t.loc[edges["txId2"]].values).mean()
    desc = X.describe().T
    iqr = desc["75%"] - desc["25%"]
    # Share of values beyond 3*IQR outside the quartiles, per feature (heavy-tail indicator)
    lo, hi = desc["25%"] - 3 * iqr, desc["75%"] + 3 * iqr
    out_share = (X.lt(lo, axis=1) | X.gt(hi, axis=1)).mean()
    return {
        "n_transactions": len(master),
        "n_edges": len(edges),
        "n_features": len(RAW_FEATURE_COLS),
        "n_time_steps": int(master["time_step"].nunique()),
        "missing_values_total": int(master.isna().sum().sum()),
        "duplicate_txids": int(master["txId"].duplicated().sum()),
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "duplicate_edges": int(edges.duplicated().sum()),
        "self_loop_edges": int((edges["txId1"] == edges["txId2"]).sum()),
        "edges_within_same_time_step": float(same_step),
        "constant_features": [c for c in RAW_FEATURE_COLS if X[c].nunique() <= 1],
        "dtypes": {str(k): int(v) for k, v in master.dtypes.astype(str).value_counts().items()},
        "inf_values": int(np.isinf(X.to_numpy()).sum()),
        "label_counts": {int(k): int(v) for k, v in master["label"].value_counts().items()},
        "outlier_share_per_feature": out_share,
    }


def make_scaled_pipeline(estimator):
    """Median-impute + standardise + estimator.

    Calling .fit() on this pipeline learns medians/means/stds from the rows it is
    given (training data only), then applies them unchanged to validation/test rows.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def make_imputed_pipeline(estimator):
    """Median-impute + estimator (tree models do not need scaling)."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])
