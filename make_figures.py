"""Generate every figure + the small data files used by the report and the Streamlit app.

Run after train.py and run_analysis.py:  python make_figures.py
"""
import json

import joblib
import numpy as np
import pandas as pd

from src import plots
from src.config import MODELS_DIR, PROC_DIR, RES_DIR, RAW_FEATURE_COLS, TRAIN_STEPS, SEED
from src.data_loader import load_master, labelled
from src.feature_engineering import add_graph_features
from src.preprocessing import quality_report
from src.splitting import temporal_split

NAMES = ["majority", "logreg", "rf", "xgb"]


def main():
    master, edges = load_master()
    meta = json.loads((RES_DIR / "run_metadata.json").read_text())
    eng = add_graph_features(master, edges, meta["neighbour_cols"])
    full = pd.concat([master, eng], axis=1)
    df = labelled(full)
    train, val, test = temporal_split(df)
    stats = {}

    # ---- EDA ----
    stats["label_counts"] = {k: int(v) for k, v in plots.fig_label_overview(master).items()}
    plots.fig_counts_per_step(master)
    rate = plots.fig_illicit_rate(master)
    stats["illicit_rate_range"] = [float(rate.illicit_rate.min()), float(rate.illicit_rate.max())]
    top = plots.univariate_auc(train)
    stats["top_univariate_auc"] = top.head(10).round(3).to_dict()
    plots.fig_feature_distributions(train, top)
    plots.fig_boxplots(train, top)
    q = quality_report(master, edges)
    plots.fig_outlier_share(q["outlier_share_per_feature"])
    stats["outlier_share_median"] = float(q["outlier_share_per_feature"].median())
    stats["outlier_share_max"] = float(q["outlier_share_per_feature"].max())
    stats["correlation"] = plots.fig_correlation(train, top)
    stats["effect_sizes"] = plots.fig_effect_sizes(train, top).round(2).to_dict()
    plots.fig_feature_drift_means(master, top)
    ks = pd.read_csv(RES_DIR / "feature_drift_ks.csv", index_col=0)
    mks = plots.fig_ks_heatmap(ks)
    stats["mean_ks_by_step"] = mks.round(3).to_dict()
    plots.fig_degree(df)
    stats["quality"] = {k: v for k, v in q.items() if k != "outlier_share_per_feature"}

    # duplicate-row analysis: do identical feature vectors carry conflicting labels?
    dup = master[master.duplicated(RAW_FEATURE_COLS, keep=False)]
    g = dup.groupby(RAW_FEATURE_COLS, sort=False)["label"].agg(lambda s: tuple(sorted(set(s))))
    stats["duplicates"] = {"rows_in_duplicate_groups": int(len(dup)), "groups": int(len(g)),
                           "groups_with_conflicting_labels": int(sum(len(set(t) - {-1}) > 1 for t in g)),
                           "same_time_step_groups": int((dup.groupby(RAW_FEATURE_COLS)["time_step"].nunique() == 1).sum())}

    # ---- evaluation ----
    res = pd.read_csv(RES_DIR / "test_results.csv")
    pred = pd.read_parquet(RES_DIR / "test_predictions.parquet")
    thr = {n: joblib.load(MODELS_DIR / f"{n}.joblib")["threshold"] for n in NAMES}
    plots.fig_pr_roc(pred, NAMES)
    plots.fig_confusion(pred, thr, NAMES)
    plots.fig_model_comparison(res)
    plots.fig_threshold_tradeoff(pred, "xgb", thr["xgb"])
    plots.fig_random_vs_temporal(pd.read_csv(RES_DIR / "random_vs_temporal.csv"))

    # ---- drift ----
    ps = pd.read_csv(RES_DIR / "per_step_metrics.csv")
    pst = ps[ps.period.isin(["val", "test"])]
    plots.fig_per_step(pst, ["logreg", "rf", "xgb"], "f1", "drift_01_per_step_f1")
    plots.fig_per_step(pst, ["logreg", "rf", "xgb"], "recall", "drift_01b_per_step_recall")
    plots.fig_per_step(pst, ["logreg", "rf", "xgb"], "precision", "drift_01c_per_step_precision")
    plots.fig_per_step_prf(pst, "xgb")
    plots.fig_per_step_prf(pst, "rf")
    for name in ("rf", "xgb"):
        runs = {"static (train 1-27)": pred[["txId", "time_step", "label"]].assign(proba=pred[name]),
                "expanding window": pd.read_parquet(RES_DIR / f"retrain_{name}_expanding.parquet"),
                "sliding window (10 steps)": pd.read_parquet(RES_DIR / f"retrain_{name}_sliding.parquet")}
        plots.fig_retraining(runs, None, f"drift_03_retraining_{name}", name)
    plots.fig_unstable(pd.read_csv(RES_DIR / "unstable_feature_removal.csv"))

    # ---- explainability / errors ----
    tree = pd.read_csv(RES_DIR / "importance_tree.csv", index_col=0)
    perm = pd.read_csv(RES_DIR / "importance_permutation.csv", index_col=0)
    plots.fig_importance(tree, perm)
    plots.fig_importance_time(perm)
    errs = pd.read_csv(RES_DIR / "errors_by_step.csv")
    plots.fig_errors_by_step(errs, "xgb")

    # ---- files for the Streamlit app (small, committed to the repo) ----
    rng = np.random.RandomState(SEED)
    lic = test[test.label == 0].sample(1500, random_state=SEED)
    ill = test[test.label == 1].sample(500, random_state=SEED)
    sample = pd.concat([lic, ill]).sample(frac=1, random_state=SEED)
    sample[["txId", "time_step", "label"] + RAW_FEATURE_COLS].to_parquet(PROC_DIR / "app_sample.parquet", index=False)
    # also a ready-made demo CSV for the upload feature
    sample.head(200)[["txId"] + RAW_FEATURE_COLS].to_csv(PROC_DIR / "demo_upload.csv", index=False)
    per_step = master.groupby("time_step")["label"].value_counts().unstack(fill_value=0)
    per_step.columns = ["unknown", "licit", "illicit"]
    per_step.to_csv(RES_DIR / "label_counts_per_step.csv")

    (RES_DIR / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps({k: stats[k] for k in ("label_counts", "duplicates", "correlation", "illicit_rate_range")}, indent=2))


if __name__ == "__main__":
    main()
