"""Static (matplotlib/seaborn) figures used in the notebooks and report.

Each function answers ONE question, saves a PNG to reports/figures and returns the
numbers behind the plot so that the written interpretation can quote them exactly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, precision_recall_curve, roc_curve,
                             roc_auc_score)

from .config import FIG_DIR, RAW_FEATURE_COLS, TRAIN_STEPS, VAL_STEPS, TEST_STEPS

ILLICIT_C, LICIT_C, UNKNOWN_C, ACCENT = "#c0392b", "#2c6fbb", "#9aa0a6", "#e08e0b"
MODEL_COLORS = {"majority": "#9aa0a6", "logreg": "#8e6bbf", "rf": "#2c6fbb", "xgb": "#c0392b"}
MODEL_LABELS = {"majority": "Majority baseline", "logreg": "Logistic Regression",
                "rf": "Random Forest", "xgb": "XGBoost"}

sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def shade_periods(ax, ymax=None, label=True):
    """Shade train / validation / test periods and mark step 43."""
    for (lo, hi), c, t in ((TRAIN_STEPS, "#e8f0fa", "train"), (VAL_STEPS, "#fdf3dc", "validation"),
                           (TEST_STEPS, "#fbe6e3", "test")):
        ax.axvspan(lo - 0.5, hi + 0.5, color=c, zorder=0, label=t if label else None)
    ax.axvline(43, color="black", ls="--", lw=1, label="step 43" if label else None)


# ------------------------------------------------------------------ EDA ---------------
def fig_label_overview(master):
    c = master["label"].map({1: "Illicit", 0: "Licit", -1: "Unknown"}).value_counts()
    c = c.reindex(["Unknown", "Licit", "Illicit"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].bar(c.index, c.values, color=[UNKNOWN_C, LICIT_C, ILLICIT_C])
    for i, v in enumerate(c.values):
        ax[0].text(i, v, f"{v:,}\n({v / c.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax[0].set_ylim(0, c.max() * 1.2)
    ax[0].set_title("All 203,769 transactions")
    ax[0].set_ylabel("Number of transactions")
    lab = c.drop("Unknown")
    ax[1].bar(lab.index, lab.values, color=[LICIT_C, ILLICIT_C])
    for i, v in enumerate(lab.values):
        ax[1].text(i, v, f"{v:,}\n({v / lab.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax[1].set_ylim(0, lab.max() * 1.2)
    ax[1].set_title("Labelled transactions only (used for ML)")
    fig.suptitle("Class distribution: most transactions are unlabelled; illicit is the rare class", y=1.02)
    _save(fig, "eda_01_class_distribution")
    return c.to_dict()


def fig_counts_per_step(master):
    g = master.groupby("time_step")["label"].value_counts().unstack(fill_value=0)
    g = g.rename(columns={1: "Illicit", 0: "Licit", -1: "Unknown"})[["Illicit", "Licit", "Unknown"]]
    fig, ax = plt.subplots(figsize=(11, 4))
    g.plot(kind="bar", stacked=True, ax=ax, color=[ILLICIT_C, LICIT_C, UNKNOWN_C], width=0.85)
    ax.set_xlabel("Time step (~2 weeks each)")
    ax.set_ylabel("Transactions")
    ax.set_title("Number of transactions per time step, split by label")
    ax.legend(title="Label", ncol=3)
    _save(fig, "eda_02_counts_per_step")
    return g


def fig_illicit_rate(master):
    lab = master[master["label"] >= 0]
    g = lab.groupby("time_step")["label"].agg(["mean", "sum", "size"]).rename(
        columns={"mean": "illicit_rate", "sum": "n_illicit", "size": "n_labelled"})
    fig, ax = plt.subplots(figsize=(11, 4))
    shade_periods(ax)
    ax.plot(g.index, g["illicit_rate"] * 100, marker="o", color=ILLICIT_C, zorder=3)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Illicit share of labelled transactions (%)")
    ax.set_title("The illicit share of labelled transactions changes strongly over time")
    ax.legend(loc="upper right", ncol=4)
    _save(fig, "eda_03_illicit_rate_over_time")
    return g


def univariate_auc(train, cols=RAW_FEATURE_COLS):
    """Rank-based separation of illicit vs licit per feature (training rows only)."""
    out = {c: roc_auc_score(train["label"], train[c]) for c in cols}
    s = pd.Series(out)
    return (s - 0.5).abs().sort_values(ascending=False) + 0.5


def fig_feature_distributions(train, top):
    """Histograms of the PERCENTILE RANK of each feature (robust to heavy tails and spikes)."""
    cols = list(top[:6].index)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), cols):
        r = train[c].rank(pct=True)
        for lab, col, nm in ((0, LICIT_C, "Licit"), (1, ILLICIT_C, "Illicit")):
            ax.hist(r[train.label == lab], bins=20, range=(0, 1), density=True, alpha=0.55, color=col, label=nm)
        ax.set_title(c)
        ax.set_xlabel("percentile rank of the value (training rows)")
        ax.set_ylabel("density")
    axes[0, 0].legend()
    fig.suptitle("Most class-separating features: where do illicit vs licit transactions fall in each feature's range?", y=1.01)
    fig.tight_layout()
    _save(fig, "eda_04_feature_distributions")
    return top[:6]


def fig_boxplots(train, top):
    cols = list(top[:10].index)
    z = (train[cols] - train[cols].median()) / (train[cols].quantile(.75) - train[cols].quantile(.25))
    z = z.clip(-15, 15).assign(label=train["label"].map({0: "Licit", 1: "Illicit"}))
    m = z.melt(id_vars="label", var_name="feature", value_name="robust z-score (clipped at ±15)")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.boxplot(data=m, x="feature", y="robust z-score (clipped at ±15)", hue="label", ax=ax,
                palette={"Licit": LICIT_C, "Illicit": ILLICIT_C}, fliersize=1.5, linewidth=0.8)
    ax.set_title("Outlier check: boxplots of the 10 most discriminative features")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "eda_05_boxplots_outliers")


def fig_outlier_share(out_share):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(out_share * 100, bins=30, color=ACCENT)
    ax.axvline(out_share.median() * 100, color="black", ls="--", label=f"median {out_share.median():.1%}")
    ax.set_xlabel("% of values beyond 3×IQR from the quartiles")
    ax.set_ylabel("Number of features")
    ax.set_title("Heavy tails: share of 'outlier' values per feature")
    ax.legend()
    _save(fig, "eda_06_outlier_share")


def fig_correlation(train, top, k=30):
    cols = list(top[:k].index)
    c = train[cols].corr(method="spearman")
    g = sns.clustermap(c, cmap="vlag", center=0, vmin=-1, vmax=1, figsize=(9, 9),
                       cbar_kws={"label": "Spearman correlation"})
    g.fig.suptitle(f"Correlation of the {k} most discriminative features (training rows)", y=1.01)
    g.savefig(FIG_DIR / "eda_07_correlation.png")
    plt.close(g.fig)
    full = train[RAW_FEATURE_COLS].corr(method="spearman").abs().to_numpy()
    iu = np.triu_indices_from(full, k=1)
    return {"share_pairs_abs_corr_gt_0.9": float((full[iu] > 0.9).mean()),
            "n_pairs_gt_0.9": int((full[iu] > 0.9).sum())}


def fig_effect_sizes(train, top):
    cols = list(top[:15].index)
    a, b = train[train.label == 1][cols], train[train.label == 0][cols]
    d = (a.mean() - b.mean()) / train[cols].std()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    d.sort_values().plot(kind="barh", ax=ax, color=[ILLICIT_C if v > 0 else LICIT_C for v in d.sort_values()])
    ax.set_xlabel("Standardised mean difference (illicit − licit)")
    ax.set_title("How far apart are the class means? (top 15 features)")
    _save(fig, "eda_08_effect_sizes")
    return d


def fig_feature_drift_means(master, top, k=4):
    cols = list(top[:k].index)
    g = master.groupby("time_step")[cols].mean()
    z = (g - g.iloc[:27].mean()) / g.iloc[:27].std()
    fig, ax = plt.subplots(figsize=(11, 4))
    shade_periods(ax)
    for c in cols:
        ax.plot(z.index, z[c], marker=".", label=c)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Step mean (z-score vs steps 1-27)")
    ax.set_title("Input drift: per-step mean of the most discriminative features (all transactions)")
    ax.legend(ncol=3, loc="upper left")
    _save(fig, "eda_09_feature_mean_drift")
    return z


def fig_ks_heatmap(ks):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    sns.heatmap(ks.T, cmap="rocket_r", ax=ax[0], cbar_kws={"label": "KS statistic vs steps 1-27"},
                xticklabels=4, yticklabels=False)
    ax[0].set_xlabel("Time step")
    ax[0].set_ylabel("165 features")
    ax[0].set_title("Feature-distribution shift vs training period")
    m = ks.mean(axis=1)
    shade_periods(ax[1], label=False)
    ax[1].plot(m.index, m.values, marker="o", color=ACCENT)
    ax[1].set_xlabel("Time step")
    ax[1].set_ylabel("Mean KS statistic over 165 features")
    ax[1].set_title("Average shift per time step")
    fig.tight_layout()
    _save(fig, "eda_10_feature_drift_ks")
    return m


def fig_degree(df):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, c in zip(ax, ("in_degree", "out_degree")):
        for lab, col, nm in ((0, LICIT_C, "Licit"), (1, ILLICIT_C, "Illicit")):
            v = df.loc[df.label == lab, c]
            vc = v.clip(upper=8).value_counts(normalize=True).sort_index()
            a.plot(vc.index, vc.values, marker="o", color=col, label=nm)
        a.set_xlabel(f"{c} (clipped at 8)")
        a.set_ylabel("share of transactions")
        a.set_title(c)
        a.legend()
    fig.suptitle("Engineered graph features: degree distributions by class", y=1.02)
    _save(fig, "fe_01_degree")


# ------------------------------------------------------------- evaluation --------------
def fig_pr_roc(test_pred, names):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    y = test_pred["label"]
    for n in names:
        p, r, _ = precision_recall_curve(y, test_pred[n])
        ax[0].plot(r, p, color=MODEL_COLORS[n], label=MODEL_LABELS[n])
        fpr, tpr, _ = roc_curve(y, test_pred[n])
        ax[1].plot(fpr, tpr, color=MODEL_COLORS[n], label=MODEL_LABELS[n])
    ax[0].axhline(y.mean(), color="grey", ls=":", label=f"prevalence {y.mean():.1%}")
    ax[0].set(xlabel="Recall (illicit)", ylabel="Precision (illicit)", title="Precision-recall curve (test period)")
    ax[1].plot([0, 1], [0, 1], color="grey", ls=":")
    ax[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve (supplementary)")
    ax[0].legend()
    _save(fig, "eval_01_pr_roc_curves")


def fig_confusion(test_pred, thresholds, names):
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6))
    for ax, n in zip(axes, names):
        pred = (test_pred[n] >= thresholds[n]).astype(int)
        ConfusionMatrixDisplay.from_predictions(test_pred["label"], pred, display_labels=["Licit", "Illicit"],
                                                cmap="Blues", ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"{MODEL_LABELS[n]}\n(threshold {thresholds[n]:.2f})", fontsize=10)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the test period (rows: actual, columns: predicted)", y=1.04)
    _save(fig, "eval_02_confusion_matrices")


def fig_model_comparison(res):
    r = res[res.threshold_rule == "val_tuned"].set_index("model")
    m = r[["precision", "recall", "f1", "pr_auc"]].loc[["majority", "logreg", "rf", "xgb"]]
    fig, ax = plt.subplots(figsize=(9, 4))
    m.rename(index=MODEL_LABELS).plot(kind="bar", ax=ax, width=0.8,
                                      color=["#6c8ebf", "#c0392b", "#2c3e50", "#e08e0b"])
    ax.set_ylabel("Score (illicit class)")
    ax.set_title("Model comparison on the held-out test period (steps 35-49)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(ncol=4, loc="upper left")
    ax.set_ylim(0, 1.05)
    _save(fig, "eval_03_model_comparison")


def fig_per_step(per_step, names, metric="f1", fname="drift_01_per_step_f1"):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    shade_periods(ax)
    for n in names:
        d = per_step[per_step.model == n]
        ax.plot(d.time_step, d[metric], marker="o", ms=4, color=MODEL_COLORS[n], label=MODEL_LABELS[n])
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Illicit-class {metric.upper().replace('_', '-')}")
    ax.set_title(f"Per-time-step {metric.upper().replace('_', '-')} of the static models (trained on steps 1-27)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(27.5, 49.5)
    ax.legend(ncol=4, loc="lower left")
    _save(fig, fname)


def fig_per_step_prf(per_step, name):
    d = per_step[per_step.model == name]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    shade_periods(ax)
    ax.plot(d.time_step, d.precision, marker="o", ms=4, color=LICIT_C, label="precision")
    ax.plot(d.time_step, d.recall, marker="s", ms=4, color=ILLICIT_C, label="recall")
    ax.plot(d.time_step, d.f1, marker="^", ms=4, color="black", label="F1")
    ax2 = ax.twinx()
    ax2.bar(d.time_step, d.n_illicit, color=ACCENT, alpha=0.35, width=0.6, label="# illicit in step")
    ax2.set_ylabel("Illicit transactions in step (bars)")
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Illicit-class score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(27.5, 49.5)
    ax.set_title(f"{MODEL_LABELS[name]}: precision, recall and F1 per time step")
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, loc="lower left", ncol=6, fontsize=8)
    _save(fig, f"drift_02_per_step_prf_{name}")


def fig_random_vs_temporal(rt):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, met, nm in zip(ax, ("f1", "pr_auc"), ("F1 (illicit)", "PR-AUC (illicit)")):
        d = rt.pivot(index="model", columns="split", values=met).loc[["logreg", "rf", "xgb"]]
        d.rename(index={"logreg": "LogReg", "rf": "RF", "xgb": "XGBoost"}).plot(kind="bar", ax=a, color=[LICIT_C, ILLICIT_C], width=0.75)
        for cont in a.containers:
            a.bar_label(cont, fmt="%.2f", fontsize=8)
        a.set_ylabel(nm)
        a.set_ylim(0, 1.1)
        a.set_title(nm)
        a.tick_params(axis="x", rotation=0)
    fig.suptitle("Random split vs temporal split: same models, same data", y=1.02)
    _save(fig, "eval_04_random_vs_temporal")


def fig_retraining(runs, metric_fn, fname, name):
    """runs: {strategy: DataFrame(txId,time_step,label,proba)} -> per-step F1 and PR-AUC."""
    from .evaluation import per_step_metrics
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
    pal = {"static (train 1-27)": "#9aa0a6", "expanding window": LICIT_C, "sliding window (10 steps)": ILLICIT_C}
    for s, r in runs.items():
        m = per_step_metrics(r.time_step, r.label, r.proba, 0.5)
        ax[0].plot(m.index, m["f1"], marker="o", ms=4, color=pal[s], label=s)
        ax[1].plot(m.index, m["pr_auc"], marker="o", ms=4, color=pal[s], label=s)
    for a, t in zip(ax, ("F1 (threshold 0.5)", "PR-AUC")):
        a.axvline(43, color="black", ls="--", lw=1)
        a.set_xlabel("Time step")
        a.set_ylabel(t)
        a.set_ylim(-0.02, 1.02)
        a.set_title(f"{MODEL_LABELS[name]}: {t} per step")
    ax[0].legend(loc="lower left", fontsize=8)
    _save(fig, fname)


def fig_importance(tree, perm, n=15):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    t = tree["xgb_gain"].sort_values().tail(n)
    ax[0].barh(t.index, t.values, color=ILLICIT_C)
    ax[0].set(xlabel="XGBoost gain importance", title=f"Top {n} features (XGBoost, built-in)")
    p = perm["xgb_val"].sort_values().tail(n)
    ax[1].barh(p.index, p.values, color=LICIT_C)
    ax[1].set(xlabel="Drop in PR-AUC when feature is shuffled", title=f"Top {n} features (permutation, validation period)")
    fig.tight_layout()
    _save(fig, "xai_01_importance")


def fig_importance_time(perm, n=12):
    top = perm["xgb_test_35-42"].sort_values(ascending=False).head(n).index
    d = perm.loc[top, ["xgb_test_35-42", "xgb_test_43-49"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d.plot(kind="bar", ax=ax, color=[LICIT_C, ILLICIT_C])
    ax.set_ylabel("Permutation importance (drop in PR-AUC)")
    ax.set_title("Do features matter equally before and after step 43? (XGBoost, test period)")
    ax.legend(["steps 35-42", "steps 43-49"])
    _save(fig, "xai_02_importance_over_time")


def fig_errors_by_step(errs, name):
    d = errs[errs.model == name].set_index("time_step")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axvline(43, color="black", ls="--", lw=1)
    w = 0.28
    ax.bar(d.index - w, d["tp"], width=w, color=LICIT_C, label="True positives (caught)")
    ax.bar(d.index, d["fn"], width=w, color=ILLICIT_C, label="False negatives (missed)")
    ax.bar(d.index + w, d["fp"], width=w, color=ACCENT, label="False positives (false alarms)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Number of transactions")
    ax.set_title(f"{MODEL_LABELS[name]}: where do the errors occur? (test period)")
    ax.legend()
    _save(fig, f"err_01_errors_by_step_{name}")


def fig_threshold_tradeoff(test_pred, name, thr):
    y = test_pred["label"].to_numpy()
    ts = np.linspace(0.02, 0.98, 49)
    fp = [((test_pred[name] >= t) & (y == 0)).sum() for t in ts]
    fn = [((test_pred[name] < t) & (y == 1)).sum() for t in ts]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(ts, fp, color=ACCENT, label="false positives (false alarms)")
    ax.plot(ts, fn, color=ILLICIT_C, label="false negatives (missed illicit)")
    ax.axvline(thr, color="black", ls="--", label=f"validation-tuned threshold {thr:.2f}")
    ax.set_xlabel("Decision threshold on illicit probability")
    ax.set_ylabel("Count (test period)")
    ax.set_title(f"{MODEL_LABELS[name]}: false alarms vs missed illicit transactions")
    ax.legend()
    _save(fig, "eval_05_threshold_tradeoff")


def fig_unstable(u):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, c in (("rf", LICIT_C), ("xgb", ILLICIT_C)):
        d = u[(u.model == name) & (u.period == "test")]
        ax.plot(d.dropped_fraction * 100, d.f1, marker="o", color=c, label=f"{MODEL_LABELS[name]} (test F1)")
        d = u[(u.model == name) & (u.period == "val")]
        ax.plot(d.dropped_fraction * 100, d.f1, marker="o", ls="--", color=c, alpha=0.5, label=f"{MODEL_LABELS[name]} (val F1)")
    ax.set_xlabel("% of most-drifting features removed")
    ax.set_ylabel("Illicit F1")
    ax.set_title("Does removing unstable features help?")
    ax.legend(fontsize=8)
    _save(fig, "drift_04_unstable_features")
