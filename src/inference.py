"""Inference helpers shared by predict.py (CLI) and the Streamlit app.

Models are trained once by train.py and loaded from models/*.joblib; nothing is
retrained at prediction time.
"""
import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, RAW_FEATURE_COLS

DISCLAIMER = ("A high risk score is a statistical signal, not proof of criminal activity. "
              "Scores should support, not replace, human investigators.")


class InputValidationError(ValueError):
    """Raised when an uploaded table cannot be scored."""


def load_bundle(name, models_dir=MODELS_DIR):
    return joblib.load(models_dir / f"{name}.joblib")


def validate_features(df, features=RAW_FEATURE_COLS, max_rows=200_000):
    """Check that `df` has every required numeric feature column; return a clean copy.

    Missing values are allowed (the pipeline imputes with training medians);
    infinite values, non-numeric values and missing columns are rejected.
    """
    if df is None or len(df) == 0:
        raise InputValidationError("The table is empty.")
    if len(df) > max_rows:
        raise InputValidationError(f"Too many rows ({len(df):,}); the limit is {max_rows:,}.")
    missing = [c for c in features if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise InputValidationError(f"{len(missing)} required feature column(s) missing: {shown}")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    bad_cells = X.isna() & df[features].notna()
    if bad_cells.to_numpy().any():
        col = bad_cells.any().idxmax()
        raise InputValidationError(f"Non-numeric value found in column '{col}'.")
    if np.isinf(X.to_numpy(dtype="float64")).any():
        raise InputValidationError("Infinite values are not allowed.")
    return X.astype("float32")


def risk_band(p, threshold):
    """Human-readable band relative to the model's validation-tuned threshold."""
    p = np.asarray(p)
    return np.where(p >= max(threshold, 0.8), "High",
                    np.where(p >= threshold, "Elevated", np.where(p >= threshold / 2, "Low-moderate", "Low")))


def predict_frame(df, model_name="xgb", threshold=None, bundle=None):
    """Score every row. Returns a DataFrame with illicit_probability, predicted_class, risk_band."""
    bundle = bundle or load_bundle(model_name)
    X = validate_features(df, bundle["features"])
    thr = bundle["threshold"] if threshold is None else threshold
    proba = bundle["model"].predict_proba(X)[:, 1]
    out = pd.DataFrame({"illicit_probability": proba,
                        "predicted_class": np.where(proba >= thr, "illicit", "licit"),
                        "risk_band": risk_band(proba, thr)}, index=df.index)
    if "txId" in df.columns:
        out.insert(0, "txId", df["txId"].values)
    return out
