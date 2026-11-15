"""Train and evaluate all models with a leakage-free chronological split.

Order of operations (the test period is touched ONLY in step 6):
  1. Load master table, keep labelled rows only.
  2. Engineer graph/time features (same-step information only).
  3. Choose feature set by expanding-window validation (steps 1-34).
  4. Tune each model by expanding-window validation (steps 1-34).
  5. Fit on train (1-27); choose the decision threshold on validation (28-34).
  6. Evaluate once on the held-out future test period (35-49); save artefacts.

Usage:  python train.py
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import (MODELS_DIR, RES_DIR, RAW_FEATURE_COLS, LOCAL_COLS, SEED,
                        TRAIN_STEPS, VAL_STEPS, TEST_STEPS)
from src.data_loader import load_master, labelled
from src.evaluation import best_f1_threshold, evaluate
from src.feature_engineering import add_graph_features, engineered_columns
from src.models import MODEL_NAMES, cv_score, fit_model, predict_proba, tune
from src.preprocessing import make_imputed_pipeline
from src.splitting import temporal_split, in_steps


def select_neighbour_columns(train, k=8):
    """Top-k LOCAL features by Random-Forest importance, learned on training rows only."""
    rf = make_imputed_pipeline(RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))
    rf.fit(train[LOCAL_COLS], train["label"])
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=LOCAL_COLS)
    return imp.sort_values(ascending=False).head(k).index.tolist()


def main():
    np.random.seed(SEED)
    t0 = time.time()
    master, edges = load_master()

    # --- 2. features (graph structure never crosses time steps) -------------------
    base_train = labelled(master)
    base_train = base_train[in_steps(base_train, TRAIN_STEPS)]
    nb_cols = select_neighbour_columns(base_train)
    eng = add_graph_features(master, edges, nb_cols)
    full = pd.concat([master, eng], axis=1)
    eng_cols = engineered_columns(nb_cols)
    df = labelled(full)
    train, val, test = temporal_split(df)
    print(f"train {len(train)} (illicit {train.label.sum()}), val {len(val)} ({val.label.sum()}), "
          f"test {len(test)} ({test.label.sum()})")
    trainval = pd.concat([train, val])           # steps 1-34 only; test excluded

    # --- 3. feature-set choice on validation folds -------------------------------
    sets = {"raw_165": RAW_FEATURE_COLS, "raw_165+engineered": RAW_FEATURE_COLS + eng_cols}
    rows = []
    for sname, cols in sets.items():
        mean, folds = cv_score("xgb", trainval, cols)
        rows.append({"feature_set": sname, "n_features": len(cols), "cv_pr_auc_mean": mean,
                     **{f"fold{i + 1}": s for i, s in enumerate(folds)}})
        print(sname, round(mean, 4), [round(f, 4) for f in folds])
    fs = pd.DataFrame(rows)
    fs.to_csv(RES_DIR / "feature_set_selection.csv", index=False)
    chosen = fs.sort_values("cv_pr_auc_mean", ascending=False).iloc[0]["feature_set"]
    cols = sets[chosen]
    print("chosen feature set:", chosen)

    # --- 4/5. tune, fit, choose threshold on validation ---------------------------
    val_rows, bundles, val_pred = [], {}, {}
    for name in MODEL_NAMES:
        best, table = tune(name, trainval, cols)
        table.to_csv(RES_DIR / f"tuning_{name}.csv", index=False)
        model = fit_model(name, train[cols], train["label"], best)
        p_val = predict_proba(model, val[cols])
        thr = best_f1_threshold(val["label"], p_val) if name != "majority" else 0.5
        val_pred[name] = p_val
        m = evaluate(val["label"], p_val, thr)
        val_rows.append({"model": name, **m, "best_params": json.dumps(best, default=str)})
        bundles[name] = {"model": model, "features": cols, "threshold": thr,
                         "params": best, "name": name}
        print(f"[{name}] val PR-AUC {m['pr_auc']:.3f} F1 {m['f1']:.3f} thr {thr:.3f} "
              f"params {best}  ({time.time() - t0:.0f}s)")
    pd.DataFrame(val_rows).to_csv(RES_DIR / "validation_results.csv", index=False)

    # --- 6. single evaluation on the held-out test period -------------------------
    test_rows, preds = [], test[["txId", "time_step", "label"]].copy()
    for name, b in bundles.items():
        p = predict_proba(b["model"], test[cols])
        preds[name] = p
        for tag, thr in (("val_tuned", b["threshold"]), ("default_0.5", 0.5)):
            test_rows.append({"model": name, "threshold_rule": tag, **evaluate(test["label"], p, thr)})
        joblib.dump(b, MODELS_DIR / f"{name}.joblib")
    res = pd.DataFrame(test_rows)
    res.to_csv(RES_DIR / "test_results.csv", index=False)
    preds.to_parquet(RES_DIR / "test_predictions.parquet", index=False)
    vp = val[["txId", "time_step", "label"]].copy()
    for n, p in val_pred.items():
        vp[n] = p
    vp.to_parquet(RES_DIR / "val_predictions.parquet", index=False)

    meta = {"seed": SEED, "train_steps": TRAIN_STEPS, "val_steps": VAL_STEPS,
            "test_steps": TEST_STEPS, "feature_set": chosen, "features": cols,
            "neighbour_cols": nb_cols, "engineered_cols": eng_cols,
            "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    (RES_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 200)
    print(res[["model", "threshold_rule", "precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy"]]
          .round(4).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
