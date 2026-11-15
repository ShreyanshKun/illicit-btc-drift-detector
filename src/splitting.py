"""Chronological (leakage-free) splits and time-aware validation folds."""
import numpy as np

from .config import TRAIN_STEPS, VAL_STEPS, TEST_STEPS, SEED, UNKNOWN


def in_steps(df, steps):
    lo, hi = steps
    return df["time_step"].between(lo, hi)


def temporal_split(df):
    """df must contain only labelled rows. Returns train, val, test frames."""
    assert (df["label"] != UNKNOWN).all(), "unknown labels must be removed first"
    tr, va, te = (df[in_steps(df, s)] for s in (TRAIN_STEPS, VAL_STEPS, TEST_STEPS))
    # Strictly ordered in time, no overlap:
    assert tr.time_step.max() < va.time_step.min() <= va.time_step.max() < te.time_step.min()
    return tr, va, te


def random_split(df, seed=SEED, frac=(0.55, 0.15, 0.30)):
    """Random split of the same labelled rows (for the leakage demonstration only)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(df))
    n_tr, n_va = int(frac[0] * len(df)), int(frac[1] * len(df))
    return (df.iloc[perm[:n_tr]], df.iloc[perm[n_tr:n_tr + n_va]],
            df.iloc[perm[n_tr + n_va:]])


# Expanding-window folds used for tuning: (train steps, validation steps).
# The last fold equals the official train/validation split. Test steps never appear.
CV_FOLDS = [((1, 13), (14, 20)), ((1, 20), (21, 27)), ((1, 27), (28, 34))]


def expanding_window_folds(df):
    for tr_steps, va_steps in CV_FOLDS:
        yield df[in_steps(df, tr_steps)], df[in_steps(df, va_steps)]
