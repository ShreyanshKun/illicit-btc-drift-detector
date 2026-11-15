"""Unit tests for the critical data-processing, splitting, evaluation and prediction code."""
import numpy as np
import pandas as pd
import pytest

from src.config import RAW_FEATURE_COLS, LOCAL_COLS, AGG_COLS
from src.data_loader import build_master
from src.evaluation import best_f1_threshold, evaluate, per_step_metrics
from src.feature_engineering import add_graph_features
from src.inference import InputValidationError, predict_frame, validate_features
from src.models import fit_model, predict_proba, spw
from src.preprocessing import make_scaled_pipeline
from src.splitting import CV_FOLDS, expanding_window_folds, random_split, temporal_split
from sklearn.linear_model import LogisticRegression


def _toy(n=400, seed=0):
    rng = np.random.RandomState(seed)
    steps = np.repeat(np.arange(1, 50), n // 49 + 1)[:n]
    X = rng.randn(n, len(RAW_FEATURE_COLS)).astype("float32")
    y = (X[:, 0] + 0.3 * rng.randn(n) > 1).astype(int)
    df = pd.DataFrame(X, columns=RAW_FEATURE_COLS)
    df.insert(0, "txId", np.arange(n) + 1000)
    df.insert(1, "time_step", steps)
    df["label"] = y
    return df


# ---------------------------------------------------------------- data loading ----------
def test_build_master_recodes_labels_and_joins():
    names = ["txId", "time_step"] + LOCAL_COLS + AGG_COLS
    feats = pd.DataFrame(np.zeros((3, len(names)), dtype="float32"), columns=names)
    feats["txId"] = [1, 2, 3]
    feats["time_step"] = [1, 1, 2]
    classes = pd.DataFrame({"txId": [3, 1, 2], "class": ["unknown", "1", "2"]})  # shuffled order
    edges = pd.DataFrame({"txId1": [1], "txId2": [2]})
    m, _ = build_master(feats, classes, edges)
    assert dict(zip(m.txId, m.label)) == {1: 1, 2: 0, 3: -1}


def test_build_master_rejects_duplicate_ids():
    names = ["txId", "time_step"] + LOCAL_COLS + AGG_COLS
    feats = pd.DataFrame(np.zeros((2, len(names)), dtype="float32"), columns=names)
    feats["txId"] = [1, 1]
    classes = pd.DataFrame({"txId": [1, 1], "class": ["1", "2"]})
    with pytest.raises(AssertionError):
        build_master(feats, classes, pd.DataFrame({"txId1": [], "txId2": []}))


# ---------------------------------------------------------------- splitting ------------
def test_temporal_split_is_chronological_and_disjoint():
    tr, va, te = temporal_split(_toy())
    assert tr.time_step.max() < va.time_step.min()
    assert va.time_step.max() < te.time_step.min()
    assert not (set(tr.txId) & set(te.txId)) and not (set(va.txId) & set(te.txId))


def test_temporal_split_rejects_unknown_labels():
    df = _toy()
    df.loc[0, "label"] = -1
    with pytest.raises(AssertionError):
        temporal_split(df)


def test_cv_folds_never_touch_test_period():
    for tr, va in expanding_window_folds(_toy(2000)):
        assert tr.time_step.max() < va.time_step.min()
        assert va.time_step.max() <= 34
    assert CV_FOLDS[-1] == ((1, 27), (28, 34))


def test_random_split_partitions_all_rows():
    df = _toy()
    a, b, c = random_split(df)
    assert len(a) + len(b) + len(c) == len(df)
    assert len(set(a.txId) | set(b.txId) | set(c.txId)) == len(df)


# ---------------------------------------------------------------- preprocessing --------
def test_scaler_uses_training_statistics_only():
    rng = np.random.RandomState(1)
    Xtr = rng.randn(200, 3)
    Xte = rng.randn(50, 3) + 100        # very different distribution
    pipe = make_scaled_pipeline(LogisticRegression())
    pipe.fit(Xtr, (Xtr[:, 0] > 0).astype(int))
    scaler = pipe.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, Xtr.mean(axis=0), atol=1e-8)   # not influenced by Xte
    assert pipe.named_steps["impute"].transform(Xte).mean() > 50            # test left untouched


def test_imputer_uses_training_median():
    Xtr = np.array([[1.0], [2.0], [3.0]])
    pipe = make_scaled_pipeline(LogisticRegression())
    pipe.fit(Xtr, [0, 1, 1])
    out = pipe.named_steps["impute"].transform(np.array([[np.nan]]))
    assert out[0, 0] == 2.0


# ---------------------------------------------------------------- features -------------
def test_graph_degrees_and_neighbour_means():
    names = ["txId", "time_step", "local_1"]
    m = pd.DataFrame({"txId": [10, 20, 30], "time_step": [1, 1, 1], "local_1": [1.0, 2.0, 6.0]})
    edges = pd.DataFrame({"txId1": [10, 10], "txId2": [20, 30]})   # 10 -> 20, 10 -> 30
    f = add_graph_features(m, edges, ["local_1"])
    assert f.loc[0, "out_degree"] == 2 and f.loc[0, "in_degree"] == 0
    assert f.loc[1, "in_degree"] == 1
    assert f.loc[0, "local_1_out_mean"] == pytest.approx(4.0)       # mean(2, 6)
    assert f.loc[1, "local_1_in_mean"] == pytest.approx(1.0)
    assert f["step_tx_count"].tolist() == [3, 3, 3]


# ---------------------------------------------------------------- evaluation -----------
def test_evaluate_known_values():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    r = evaluate(y, p, 0.5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 3)
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5


def test_majority_baseline_has_high_accuracy_but_zero_recall():
    y = np.array([1] * 5 + [0] * 95)
    r = evaluate(y, np.zeros(100), 0.5)
    assert r["accuracy"] == 0.95 and r["recall"] == 0 and r["f1"] == 0


def test_best_threshold_separates_perfect_scores():
    y = np.array([0, 0, 1, 1])
    t = best_f1_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < t <= 0.8


def test_per_step_metrics_nan_without_illicit():
    m = per_step_metrics([1, 1, 2, 2], [0, 0, 1, 0], [0.1, 0.2, 0.9, 0.1], 0.5)
    assert np.isnan(m.loc[1, "recall"]) and m.loc[2, "recall"] == 1.0


def test_scale_pos_weight():
    assert spw([0] * 9 + [1]) == 9.0


# ---------------------------------------------------------------- prediction -----------
@pytest.fixture(scope="module")
def tiny_bundle():
    df = _toy(600)
    model = fit_model("logreg", df[RAW_FEATURE_COLS], df["label"])
    return {"model": model, "features": RAW_FEATURE_COLS, "threshold": 0.5}, df


def test_predict_frame_output(tiny_bundle):
    bundle, df = tiny_bundle
    out = predict_frame(df.head(20), bundle=bundle)
    assert len(out) == 20
    assert out["illicit_probability"].between(0, 1).all()
    assert set(out["predicted_class"]) <= {"illicit", "licit"}
    assert out["txId"].tolist() == df.head(20)["txId"].tolist()


def test_validation_rejects_missing_columns_and_bad_values():
    df = _toy(10)
    with pytest.raises(InputValidationError):
        validate_features(df.drop(columns=["local_5"]))
    bad = df.copy().astype({"local_1": "object"})
    bad.loc[0, "local_1"] = "abc"
    with pytest.raises(InputValidationError):
        validate_features(bad)
    inf = df.copy()
    inf.loc[0, "agg_3"] = np.inf
    with pytest.raises(InputValidationError):
        validate_features(inf)
    with pytest.raises(InputValidationError):
        validate_features(df.iloc[0:0])


def test_nan_features_are_imputed_not_rejected(tiny_bundle):
    bundle, df = tiny_bundle
    d = df.head(5).copy()
    d.loc[d.index[0], "local_2"] = np.nan
    assert predict_frame(d, bundle=bundle)["illicit_probability"].notna().all()
