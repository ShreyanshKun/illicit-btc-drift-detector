"""Secondary experiments built on the models trained by train.py.

Stages (run all, or pick some):  python run_analysis.py [random ablation drift unstable importance errors]

  random     RQ1  random split vs temporal split
  ablation   RQ3  local-only / aggregated / engineered feature sets
  drift      RQ2/4 per-step metrics, feature drift (KS), static vs sliding vs expanding retraining
  unstable   RQ4  dropping features that drift between train and validation
  importance      impurity / permutation importance, and importance over time
  errors          FP/FN analysis, model agreement, bootstrap comparison

Test-period rows are used here only to REPORT results; no model/threshold is chosen from them.
"""
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import (MODELS_DIR, RES_DIR, RAW_FEATURE_COLS, LOCAL_COLS, AGG_COLS,
                        SEED, TRAIN_STEPS, VAL_STEPS, TEST_STEPS)
from src.data_loader import load_master, labelled
from src.drift_analysis import (feature_drift_by_step, ks_vs_reference, retrain_experiment,
                                summarise_strategy)
from src.evaluation import best_f1_threshold, evaluate, per_step_metrics
from src.explainability import perm_importance, tree_importance
from src.feature_engineering import add_graph_features
from src.models import MODEL_NAMES, fit_model, predict_proba
from src.splitting import random_split, temporal_split, in_steps

META = json.loads((RES_DIR / "run_metadata.json").read_text())
BUNDLES = {n: joblib.load(MODELS_DIR / f"{n}.joblib") for n in MODEL_NAMES}
PARAMS = {n: BUNDLES[n]["params"] for n in MODEL_NAMES}


def get_data():
    master, edges = load_master()
    eng = add_graph_features(master, edges, META["neighbour_cols"])
    full = pd.concat([master, eng], axis=1)
    df = labelled(full)
    return master, full, df, *temporal_split(df)


def stage_random(df, train, val, test):
    rows = []
    r_tr, r_va, r_te = random_split(df)
    cols = RAW_FEATURE_COLS
    for name in ("logreg", "rf", "xgb"):
        for regime, (a, b, c) in {"temporal": (train, val, test), "random": (r_tr, r_va, r_te)}.items():
            m = fit_model(name, a[cols], a["label"], PARAMS[name])
            thr = best_f1_threshold(b["label"], predict_proba(m, b[cols]))
            res = evaluate(c["label"], predict_proba(m, c[cols]), thr)
            rows.append({"model": name, "split": regime, "n_train": len(a), "n_test": len(c),
                         "test_illicit": int(c["label"].sum()), **res})
            print("random-vs-temporal", name, regime, round(res["f1"], 3), round(res["pr_auc"], 3))
    pd.DataFrame(rows).to_csv(RES_DIR / "random_vs_temporal.csv", index=False)


def stage_ablation(train, val, test):
    sets = {"local_only_93": LOCAL_COLS, "aggregated_only_72": AGG_COLS,
            "raw_165": RAW_FEATURE_COLS,
            "raw_165+engineered": RAW_FEATURE_COLS + META["engineered_cols"]}
    rows = []
    for name in ("rf", "xgb"):
        for sname, cols in sets.items():
            m = fit_model(name, train[cols], train["label"], PARAMS[name])
            pv = predict_proba(m, val[cols])
            thr = best_f1_threshold(val["label"], pv)
            for part, d, p in (("val", val, pv), ("test", test, predict_proba(m, test[cols]))):
                rows.append({"model": name, "feature_set": sname, "n_features": len(cols),
                             "period": part, **evaluate(d["label"], p, thr)})
            print("ablation", name, sname, [round(r["f1"], 3) for r in rows[-2:]])
    pd.DataFrame(rows).to_csv(RES_DIR / "feature_ablation.csv", index=False)


def stage_drift(master, df, train, val, test):
    cols = RAW_FEATURE_COLS
    preds_te = pd.read_parquet(RES_DIR / "test_predictions.parquet")
    preds_va = pd.read_parquet(RES_DIR / "val_predictions.parquet")
    # per-step metrics of the static models (val + test periods)
    rows = []
    for name in MODEL_NAMES:
        thr = BUNDLES[name]["threshold"]
        for part, p in (("val", preds_va), ("test", preds_te)):
            m = per_step_metrics(p["time_step"], p["label"], p[name], thr).reset_index()
            m.insert(0, "period", part)
            m.insert(0, "model", name)
            rows.append(m)
    pd.concat(rows).to_csv(RES_DIR / "per_step_metrics.csv", index=False)

    # feature drift (input distribution), all transactions
    ks = feature_drift_by_step(master)
    ks.to_csv(RES_DIR / "feature_drift_ks.csv")
    print("feature drift computed")

    # retraining strategies
    summaries = []
    for name in ("rf", "xgb"):
        static = preds_te[["txId", "time_step", "label"]].assign(proba=preds_te[name])
        runs = {"static (train 1-27)": static}
        runs["expanding window"] = retrain_experiment(df, cols, name, PARAMS[name], TEST_STEPS, strategy="expanding")
        runs["sliding window (10 steps)"] = retrain_experiment(df, cols, name, PARAMS[name], TEST_STEPS, window=10, strategy="sliding")
        for sname, res in runs.items():
            res.to_parquet(RES_DIR / f"retrain_{name}_{sname.split()[0]}.parquet", index=False)
            s = summarise_strategy(res, 0.5).reset_index().rename(columns={"index": "period"})
            s.insert(0, "strategy", sname)
            s.insert(0, "model", name)
            summaries.append(s)
            print("retrain", name, sname, s.iloc[0][["f1", "pr_auc"]].astype(float).round(3).to_dict())
    pd.concat(summaries).to_csv(RES_DIR / "retraining_summary.csv", index=False)


def stage_unstable(train, val, test):
    cols = RAW_FEATURE_COLS
    ks = ks_vs_reference(train.sample(min(20000, len(train)), random_state=SEED), val, cols)
    ks.sort_values(ascending=False).to_csv(RES_DIR / "ks_train_vs_val.csv", header=["ks"])
    rows = []
    for name in ("rf", "xgb"):
        for frac in (0.0, 0.10, 0.25, 0.50):
            drop = ks.sort_values(ascending=False).head(int(frac * len(cols))).index
            keep = [c for c in cols if c not in set(drop)]
            m = fit_model(name, train[keep], train["label"], PARAMS[name])
            pv = predict_proba(m, val[keep])
            thr = best_f1_threshold(val["label"], pv)
            for part, d, p in (("val", val, pv), ("test", test, predict_proba(m, test[keep]))):
                rows.append({"model": name, "dropped_fraction": frac, "n_features": len(keep),
                             "period": part, **evaluate(d["label"], p, thr)})
            print("unstable", name, frac, [round(r["f1"], 3) for r in rows[-2:]])
    pd.DataFrame(rows).to_csv(RES_DIR / "unstable_feature_removal.csv", index=False)


def stage_importance(df, train, val, test):
    cols = RAW_FEATURE_COLS
    rf, xg = BUNDLES["rf"]["model"], BUNDLES["xgb"]["model"]
    tree = pd.DataFrame({"rf_impurity": tree_importance(rf, cols), "xgb_gain": tree_importance(xg, cols)})
    tree.to_csv(RES_DIR / "importance_tree.csv", index_label="feature")
    early, late = test[test.time_step < 43], test[test.time_step >= 43]
    perm = {
        "xgb_val": perm_importance(xg, val[cols], val["label"]),
        "xgb_test_35-42": perm_importance(xg, early[cols], early["label"]),
        "xgb_test_43-49": perm_importance(xg, late[cols], late["label"]),
        "rf_test": perm_importance(rf, test[cols], test["label"]),
    }
    pd.DataFrame(perm).to_csv(RES_DIR / "importance_permutation.csv", index_label="feature")
    # importance over time: gain importance of XGB trained on different periods
    over = {}
    for tag, steps in {"steps_1-13": (1, 13), "steps_14-27": (14, 27), "steps_28-34": (28, 34)}.items():
        d = df[in_steps(df, steps)]
        m = fit_model("xgb", d[cols], d["label"], PARAMS["xgb"])
        over[tag] = tree_importance(m, cols)
    over = pd.DataFrame(over)
    over.to_csv(RES_DIR / "importance_over_time.csv", index_label="feature")
    corr = over.corr(method="spearman")
    corr.to_csv(RES_DIR / "importance_over_time_spearman.csv")
    print("importance rank correlation across periods:\n", corr.round(3))


def stage_errors(test):
    cols = RAW_FEATURE_COLS
    p = pd.read_parquet(RES_DIR / "test_predictions.parquet")
    rows = []
    for name in ("logreg", "rf", "xgb"):
        thr = BUNDLES[name]["threshold"]
        pred = (p[name] >= thr).astype(int)
        g = p.assign(pred=pred)
        g["fp"] = (g.pred == 1) & (g.label == 0)
        g["fn"] = (g.pred == 0) & (g.label == 1)
        g["tp"] = (g.pred == 1) & (g.label == 1)
        s = g.groupby("time_step")[["fp", "fn", "tp"]].sum().reset_index()
        s.insert(0, "model", name)
        rows.append(s)
    pd.concat(rows).to_csv(RES_DIR / "errors_by_step.csv", index=False)

    # Do RF and XGB fail on the same illicit transactions?
    fn = {n: set(p.loc[(p[n] < BUNDLES[n]["threshold"]) & (p.label == 1), "txId"]) for n in ("rf", "xgb")}
    fp = {n: set(p.loc[(p[n] >= BUNDLES[n]["threshold"]) & (p.label == 0), "txId"]) for n in ("rf", "xgb")}
    agree = {"fn_rf": len(fn["rf"]), "fn_xgb": len(fn["xgb"]), "fn_both": len(fn["rf"] & fn["xgb"]),
             "fn_jaccard": len(fn["rf"] & fn["xgb"]) / max(len(fn["rf"] | fn["xgb"]), 1),
             "fp_rf": len(fp["rf"]), "fp_xgb": len(fp["xgb"]), "fp_both": len(fp["rf"] & fp["xgb"]),
             "fp_jaccard": len(fp["rf"] & fp["xgb"]) / max(len(fp["rf"] | fp["xgb"]), 1)}

    # Bootstrap (resampling transactions) for RF vs XGB difference in PR-AUC / F1
    rng = np.random.RandomState(SEED)
    diffs = []
    y = p["label"].to_numpy()
    for _ in range(300):
        i = rng.randint(0, len(p), len(p))
        a = evaluate(y[i], p["rf"].to_numpy()[i], BUNDLES["rf"]["threshold"])
        b = evaluate(y[i], p["xgb"].to_numpy()[i], BUNDLES["xgb"]["threshold"])
        diffs.append((a["pr_auc"] - b["pr_auc"], a["f1"] - b["f1"]))
    d = np.array(diffs)
    agree.update(pr_auc_diff_rf_minus_xgb_mean=float(d[:, 0].mean()),
                 pr_auc_diff_ci=[float(np.percentile(d[:, 0], 2.5)), float(np.percentile(d[:, 0], 97.5))],
                 f1_diff_rf_minus_xgb_mean=float(d[:, 1].mean()),
                 f1_diff_ci=[float(np.percentile(d[:, 1], 2.5)), float(np.percentile(d[:, 1], 97.5))])
    (RES_DIR / "error_agreement_bootstrap.json").write_text(json.dumps(agree, indent=2))
    print(json.dumps(agree, indent=2))

    # What do missed illicit transactions look like? (standardised mean difference vs caught ones)
    best = "xgb"
    thr = BUNDLES[best]["threshold"]
    t = test.assign(proba=p[best].values)
    t["outcome"] = np.select([(t.proba >= thr) & (t.label == 1), (t.proba < thr) & (t.label == 1),
                              (t.proba >= thr) & (t.label == 0)], ["TP", "FN", "FP"], "TN")
    t[["txId", "time_step", "label", "proba", "outcome"]].to_parquet(RES_DIR / "xgb_test_outcomes.parquet", index=False)
    z = (test[cols] - train[cols].mean()) / train[cols].std().replace(0, 1)
    summ = pd.DataFrame({o: z[t.outcome.values == o].mean() for o in ("TP", "FN", "FP", "TN")})
    summ.to_csv(RES_DIR / "outcome_feature_profile_z.csv", index_label="feature")


if __name__ == "__main__":
    stages = sys.argv[1:] or ["random", "ablation", "drift", "unstable", "importance", "errors"]
    master, full, df, train, val, test = get_data()
    t0 = time.time()
    for s in stages:
        print(f"=== {s} ===", flush=True)
        {"random": lambda: stage_random(df, train, val, test),
         "ablation": lambda: stage_ablation(train, val, test),
         "drift": lambda: stage_drift(master, df, train, val, test),
         "unstable": lambda: stage_unstable(train, val, test),
         "importance": lambda: stage_importance(df, train, val, test),
         "errors": lambda: stage_errors(test)}[s]()
        print(f"--- {s} done ({time.time() - t0:.0f}s)", flush=True)
