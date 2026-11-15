"""Metrics for the illicit (positive) class and per-time-step evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def best_f1_threshold(y_true, proba):
    """Threshold maximising F1 for the illicit class (to be chosen on VALIDATION data)."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.nanargmax(f1))])


def evaluate(y_true, proba, threshold=0.5):
    """Illicit-class precision/recall/F1/PR-AUC + ROC-AUC, accuracy and confusion counts."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    out = {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba) if y_true.sum() else np.nan,
        "roc_auc": roc_auc_score(y_true, proba) if 0 < y_true.sum() < len(y_true) else np.nan,
        "accuracy": accuracy_score(y_true, pred),
        "threshold": threshold,
    }
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


def per_step_metrics(steps, y_true, proba, threshold):
    """Illicit-class metrics for each time step (NaN where a step has no illicit rows)."""
    df = pd.DataFrame({"t": np.asarray(steps), "y": np.asarray(y_true),
                       "p": np.asarray(proba)})
    rows = []
    for t, g in df.groupby("t"):
        m = evaluate(g["y"], g["p"], threshold)
        m.update(time_step=int(t), n=len(g), n_illicit=int(g["y"].sum()))
        if g["y"].sum() == 0:
            m.update(recall=np.nan, f1=np.nan, pr_auc=np.nan)
        rows.append(m)
    return pd.DataFrame(rows).set_index("time_step")
