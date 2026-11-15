"""Concept-drift tools: feature-distribution drift and retraining strategies."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import SEED, RAW_FEATURE_COLS
from .evaluation import evaluate
from .models import fit_model, predict_proba


def ks_vs_reference(ref, other, cols):
    """Two-sample Kolmogorov-Smirnov statistic per feature (0 = identical, 1 = disjoint)."""
    return pd.Series({c: ks_2samp(ref[c].to_numpy(), other[c].to_numpy()).statistic
                      for c in cols})


def feature_drift_by_step(master, ref_steps=(1, 27), cols=RAW_FEATURE_COLS,
                          ref_sample=20000):
    """KS statistic of every feature, each time step vs the reference (training) period.

    Uses ALL transactions (labels are not needed to measure input drift)."""
    lo, hi = ref_steps
    ref = master[master["time_step"].between(lo, hi)]
    ref = ref.sample(min(ref_sample, len(ref)), random_state=SEED)
    out = {}
    for t, g in master.groupby("time_step"):
        out[t] = ks_vs_reference(ref, g, cols)
    return pd.DataFrame(out).T.rename_axis("time_step")


def retrain_experiment(df, cols, model_name, params, test_steps, window=None,
                       strategy="expanding", threshold=0.5):
    """Walk forward through test steps; before scoring step t, (re)train on past steps only.

    strategy: 'expanding' -> all labelled steps < t
              'sliding'   -> the last `window` labelled steps before t
    Assumes labels of step t-1 are available when scoring step t (stated limitation).
    Returns a DataFrame with txId, time_step, label, proba.
    """
    out = []
    for t in range(test_steps[0], test_steps[1] + 1):
        lo = 1 if strategy == "expanding" else max(1, t - window)
        tr = df[df["time_step"].between(lo, t - 1)]
        te = df[df["time_step"] == t]
        if te.empty:
            continue
        m = fit_model(model_name, tr[cols], tr["label"], params)
        res = te[["txId", "time_step", "label"]].copy()
        res["proba"] = predict_proba(m, te[cols])
        out.append(res)
    return pd.concat(out, ignore_index=True)


def summarise_strategy(res, threshold, split_step=43):
    """Pooled metrics for a retraining run, overall and before/after `split_step`."""
    rows = {}
    for tag, sel in (("all_35-49", res.time_step >= 0),
                     (f"35-{split_step - 1}", res.time_step < split_step),
                     (f"{split_step}-49", res.time_step >= split_step)):
        r = res[sel]
        rows[tag] = evaluate(r["label"], r["proba"], threshold)
    return pd.DataFrame(rows).T.astype(float)
