"""Builds the five analysis notebooks (then execute with execute_notebooks.py).

Code cells call the reusable functions in src/; every plot is followed by a written
interpretation that quotes the numbers produced by the pipeline.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = '''import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
from src.config import *
RES = RES_DIR
def show(name, width=900):
    display(Image(filename=str(FIG_DIR / f"{name}.png"), width=width))
'''


def nb(cells, name):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_code_cell(SETUP)] + [
        nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, HERE / name)


md = lambda t: ("md", t)
code = lambda t: ("code", t)

# ===================================================================== 01
nb([
    md("""# 01 · Data understanding

**Question.** Can we load the three Elliptic files, join them reliably and describe what we have?

Elliptic (Weber et al., 2019) is a public Bitcoin graph released by the blockchain-analytics firm Elliptic:
each **node** is a transaction, each **edge** is a flow of coins from one transaction into another, and each node has
166 anonymised features (the first is the time step, leaving 165 model inputs: 93 *local* + 72 *aggregated-neighbour*).

Files: `elliptic_txs_features.csv` (no header), `elliptic_txs_classes.csv` (`1` illicit, `2` licit, `unknown`), `elliptic_txs_edgelist.csv`."""),
    code("""from src.data_loader import load_master, load_raw, build_master
feats, classes, edges = load_raw()
print("features:", feats.shape, "| classes:", classes.shape, "| edges:", edges.shape)
master, edges = load_master()           # joins + verifies ids (assertions inside build_master)
print("master:", master.shape)
master.iloc[:3, :6]"""),
    md("""**Join verification (all done by assertions in `build_master`).** Transaction ids are unique in both the feature and class files,
the two id sets are identical, the join is one-to-one (203,769 rows in, 203,769 rows out), and every edge endpoint exists in the
feature table. Raw label codes were recoded to `1 = illicit, 0 = licit, -1 = unknown` so that "2" is no longer mistaken for the positive class."""),
    code("""from src.preprocessing import quality_report
q = quality_report(master, edges)
pd.Series({k: v for k, v in q.items() if k != "outlier_share_per_feature"})"""),
    md("""## Data-quality decisions

| # | Problem found | Why it matters | Action taken | Why that is appropriate |
|---|---|---|---|---|
| 1 | **157,205 transactions (77.1%) have no label** | A model cannot learn from, or be scored on, unknown ground truth. Treating unknown as licit would inject label noise and flatter the metrics. | Kept in the master table, but **excluded from every train/validation/test set and every supervised metric** (`labelled()` helper). Unknown rows are used only for label-free work: graph degrees and input-drift (KS) measurement. | Keeps evaluation honest; uses the extra data only where no label is required. |
| 2 | **Missing values: 0** (and 0 infinite values) | Would break most models if present. | No imputation was needed on this data; a median imputer fitted on training rows is still part of every pipeline. | Makes the prediction pipeline robust to uploaded CSVs with gaps, without leaking test statistics. |
| 3 | **Duplicate ids: 0, duplicate edges: 0, self-loops: 0. 34 duplicate *feature rows*** (49 rows in 15 groups) | Duplicates could leak between train and test. | Checked: all 15 groups sit **inside the same time step** and **none has conflicting labels**. Rows kept. | They are distinct transactions with identical features (0.017% of rows); they cannot cross the chronological split, so there is no leakage and deleting them would discard real events. |
| 4 | **Heavy tails / outliers**: median feature has 13.4% of values beyond 3×IQR (max 38.3%) | Linear models are distorted by extreme values; but extremes may be genuine behaviour. | **Not removed.** Tree models are scale-invariant. For Logistic Regression: standardise (train statistics) then clip at ±5 SD. | In the boxplot analysis (notebook 02) extremes are overwhelmingly *licit* transactions, so deleting outliers would delete signal and bias the data. |
| 5 | **Data types**: features arrive as float64/int64 | 690 MB CSV; memory. | Features cast to `float32`, `time_step` to `int16`, `label` to `int8`; the headerless feature file is given explicit names (`local_1…93`, `agg_1…72`). | 4× smaller without losing precision relevant to trees/LR. Column names make downstream code and explanations readable. |
| 6 | **Time step is column 2 of the feature file** | If used as an input the model could memorise "when" instead of "what", and could not generalise to unseen future steps. | Used **only** to split the data and to evaluate per step; never a model feature. | Prevents a form of leakage and matches real deployment. |
| 7 | **Edges never cross time steps** (share within the same step = 1.0) | If edges linked to the future, graph features would leak later information. | Verified; so degree and neighbour features are computed per step without using future data. | The graph is effectively 49 separate snapshots. |
| 8 | **Class imbalance**: 9.8% of labelled transactions are illicit | Accuracy is misleading. | Class-weighting / `scale_pos_weight`, illicit-class precision/recall/F1/PR-AUC, majority baseline reported. | See notebook 04. |"""),
    code("""lab = master["label"].map({1: "illicit", 0: "licit", -1: "unknown"}).value_counts()
summary = pd.DataFrame({"count": lab, "share": (lab / lab.sum()).round(4)})
summary.loc["labelled illicit share"] = [np.nan, round(4545 / (4545 + 42019), 4)]
summary"""),
    md("""**Summary.** 203,769 transactions, 234,355 edges, 49 time steps. 4,545 illicit (2.2%), 42,019 licit (20.6%), 157,205 unknown (77.1%).
Only the 46,564 labelled transactions (9.8% illicit) are used for supervised learning. The data is clean in the sense of no missing values or
duplicate ids; the real difficulties are imbalance, scarcity of labels, heavy tails and time (next notebooks)."""),
], "01_data_understanding.ipynb")

# ===================================================================== 02
nb([
    md("""# 02 · Exploratory data analysis

Every figure below answers one question and is followed by a short interpretation. Figures are produced by `src/plots.py`
(saved in `reports/figures/`). Colours: **red = illicit, blue = licit, grey = unknown**. The train/validation/test periods are
shaded on the time-axis plots; the dashed line marks time step 43."""),
    code("""from src import plots
from src.data_loader import load_master, labelled
from src.splitting import temporal_split
master, edges = load_master()
df = labelled(master); train, val, test = temporal_split(df)
print(len(train), len(val), len(test))"""),
    md("### 2.1 How imbalanced is the problem, and how much of the data can we use?"),
    code("""plots.fig_label_overview(master); show("eda_01_class_distribution")"""),
    md("""**Insight.** 77.1% of transactions are unlabelled and unusable for supervised work; of the labelled ones only 9.8% are illicit (4,545 vs 42,019).
A classifier that always answers "licit" would score about 90% accuracy while catching no crime, so accuracy cannot be our metric; we use illicit-class precision, recall, F1 and PR-AUC."""),
    md("### 2.2 How are transactions distributed over time?"),
    code("""plots.fig_counts_per_step(master); show("eda_02_counts_per_step", 1000)"""),
    md("""**Insight.** Steps hold between about 1,100 (step 27) and 7,900 (step 1) transactions, and roughly 78% of every step is unlabelled. The labelled sample per step is therefore small
(step 27 has only 206 labelled transactions), so per-step metrics later on rest on few cases and are noisy."""),
    md("### 2.3 Does the share of illicit transactions stay constant over time? (first evidence of drift)"),
    code("""rate = plots.fig_illicit_rate(master); show("eda_03_illicit_rate_over_time", 1000)
rate.assign(period=pd.cut(rate.index, [0, 27, 34, 42, 49], labels=["1-27 train", "28-34 val", "35-42 test", "43-49 test"])).groupby("period", observed=True).apply(
    lambda g: pd.Series({"illicit_share": g.n_illicit.sum() / g.n_labelled.sum(), "n_illicit": g.n_illicit.sum()})).round(3)"""),
    md("""**Insight.** The illicit share swings from 0.3% (step 46) to 36% (step 13). It averages 9.9% in the training period, **20.2% in validation**, 9.2% in steps 35-42 and then **collapses to 2.5% in steps 43-49** (only 169 illicit transactions). The base rate itself drifts,
which means a threshold tuned on the validation period (20% illicit) is tuned on an unusually illicit-rich period, and precision will change with time even if the model is unchanged. This is the first sign that the future will not look like the past."""),
    md("### 2.4 Do single features separate the classes?"),
    code("""top = plots.univariate_auc(train)          # training rows only
plots.fig_feature_distributions(train, top); show("eda_04_feature_distributions", 1000)
top.head(10).round(3)"""),
    md("""**Insight.** Histograms use the *percentile rank* of each value because these features are spiky and heavy-tailed. For `local_53` and `local_55` illicit transactions sit in the lowest 20% of the range; for `local_90`, `local_47` and `local_41` they pile up in one low bin, whereas licit transactions are spread evenly.
Single features already reach a univariate AUC of 0.89 (`local_53`), so the classes are separable, and combinations of features should do better. Because features are anonymised we do not attach any real-world meaning to these columns."""),
    md("### 2.5 Are the extreme values errors that should be removed?"),
    code("""plots.fig_boxplots(train, top); show("eda_05_boxplots_outliers")
q = json.loads((RES / "eda_stats.json").read_text())
print("median share of values beyond 3*IQR:", round(q["outlier_share_median"], 3), "| max:", round(q["outlier_share_max"], 3))"""),
    md("""**Insight.** The long tails (robust z-scores above 10) belong almost entirely to **licit** transactions; illicit transactions occupy a narrow band near the bottom of the range. These "outliers" are therefore genuine behaviour that helps tell the classes apart, so removing them would discard signal and make the licit class look artificially uniform. We keep them."""),
    code("""plots.fig_outlier_share(__import__("src.preprocessing", fromlist=["x"]).quality_report(master, edges)["outlier_share_per_feature"]); show("eda_06_outlier_share", 600)"""),
    md("""**Insight.** The median feature has 13.4% of its values beyond 3×IQR and one has 38.3%: heavy tails are the norm, not the exception. Tree models ignore monotone rescaling so they are unaffected; Logistic Regression gets standardisation plus clipping at ±5 SD so that outliers cannot dominate."""),
    md("### 2.6 Are the features redundant?"),
    code("""stats = plots.fig_correlation(train, top); show("eda_07_correlation", 700); stats"""),
    md("""**Insight.** There are clear blocks of near-duplicate local features (for example `local_23/25/29/31` and `local_41/43/47/49`). Across all 165 features 131 of 13,530 pairs (1.0%) have |Spearman ρ| above 0.9, so redundancy is real but limited. Consequence for interpretation: when features are correlated, a model can spread importance among them, so individual importances should be read as indicative, not exact."""),
    md("### 2.7 How big are the class differences?"),
    code("""d = plots.fig_effect_sizes(train, top); show("eda_08_effect_sizes", 600); d.round(2).head(5)"""),
    md("""**Insight.** Illicit transactions have *lower* values on all of the 15 most discriminative features. The largest standardised mean difference is only about one standard deviation (`local_53`: -1.04, `local_55`: -0.90, `local_90`: -0.86); no single feature is decisive, which is why multi-feature models are needed."""),
    md("### 2.8 Do the input distributions change over time?"),
    code("""plots.fig_feature_drift_means(master, top); show("eda_09_feature_mean_drift", 1000)"""),
    md("""**Insight.** The per-step average of the most discriminative features mostly stays within about ±2 SD of its training level during steps 1-27 (with occasional spikes, e.g. `local_47` at step 19), but then moves persistently away: `local_90` sits roughly 2 SD *below* its training level from step ~32 onward, and `local_47` has large spikes in the test period. The inputs are shifting even before step 43."""),
    code("""ks = pd.read_csv(RES / "feature_drift_ks.csv", index_col=0)
plots.fig_ks_heatmap(ks); show("eda_10_feature_drift_ks", 1100)
m = ks.mean(axis=1)
pd.DataFrame({p: {"mean KS": m.loc[a:b].mean(), "share of features with KS>0.2": (ks.loc[a:b] > 0.2).mean().mean()}
              for p, (a, b) in {"1-27": (1, 27), "28-34": (28, 34), "35-42": (35, 42), "43-49": (43, 49)}.items()}).T.round(3)"""),
    md("""**Insight.** Measured with the Kolmogorov-Smirnov statistic against the training period (all 203,769 transactions, no labels needed), average feature shift rises steadily from 0.113 (steps 1-27) to 0.142 (28-34), 0.164 (35-42) and 0.166 (43-49); the share of features with KS above 0.2 doubles from 11% to 21%.
The drift in the inputs is **gradual** and does not jump at step 43. Since performance collapses abruptly after step 43 (notebook 05), this suggests that the main problem is a change in *how the features relate to the label* (concept drift), not only a shift in the feature values."""),
    md("""## Summary of EDA findings
1. Strong class imbalance (9.8% illicit among labelled) and 77% unlabelled data.
2. The illicit share is not constant (0.3% to 36%) and falls to 2.5% after step 43.
3. Heavy-tailed, partly redundant features; extremes are informative so outliers are kept.
4. Individual features separate the classes only moderately (AUC up to 0.89), so we need multivariate models.
5. Input distributions drift gradually; the label relationship appears to change sharply later."""),
], "02_eda.ipynb")

# ===================================================================== 03
nb([
    md("""# 03 · Feature engineering

**Question.** Do graph and time features add value beyond the 165 supplied columns, without using information from the future?

Engineered features (`src/feature_engineering.py`):
* `in_degree`, `out_degree`, `degree`: how many transactions feed into / receive from this one;
* `step_tx_count`: number of transactions in the same time step (a market-activity indicator);
* for the 8 most important local features (chosen by a Random Forest fitted on **training rows only**), the **mean over in-neighbours and out-neighbours**.

**No-leakage argument.** All edges join transactions of the same time step (checked in notebook 01), so every feature uses only same-step information. Neighbour *labels* are never used (that would leak the target)."""),
    code("""from src import plots
from src.data_loader import load_master, labelled
from src.feature_engineering import add_graph_features
master, edges = load_master()
meta = json.loads((RES / "run_metadata.json").read_text())
eng = add_graph_features(master, edges, meta["neighbour_cols"])
df = pd.concat([master, eng], axis=1); df = df[df.label >= 0]
print("neighbour columns:", meta["neighbour_cols"]); print("engineered columns:", len(meta["engineered_cols"]))
plots.fig_degree(df); show("fe_01_degree", 850)
df.groupby("label")[["in_degree", "out_degree", "degree"]].mean().round(2)"""),
    md("""**Insight.** Illicit transactions are less connected: mean total degree 2.0 vs 3.1 for licit. Only 6.7% of illicit transactions have an out-degree of 2 or more, versus 23.3% of licit ones. So degree carries some information, but the distributions overlap heavily (median 1 for both classes)."""),
    code("""sel = pd.read_csv(RES / "feature_set_selection.csv"); print("Chosen by expanding-window CV on steps 1-34 (XGBoost, default settings):"); display(sel.round(4))
ab = pd.read_csv(RES / "feature_ablation.csv")
ab.pivot_table(index=["model", "feature_set"], columns="period", values=["f1", "pr_auc"]).round(3)"""),
    md("""**Insight (does feature engineering help?).** No consistently.
* In expanding-window cross-validation (steps 1-34) the 165 original features scored a mean PR-AUC of 0.9464 versus 0.9455 with the engineered features; this validation-only comparison decided to keep the original 165 columns.
* For transparency we also fitted both sets and looked at the held-out test period after that decision: PR-AUC is essentially unchanged (Random Forest 0.784 → 0.786, XGBoost 0.783 → 0.781). F1 moves in opposite directions (Random Forest +0.039, XGBoost -0.014), mainly through precision, which is within the noise of a threshold choice.
* The supplied aggregated features already summarise the neighbourhood, which is a plausible reason additional neighbour means add little (a hypothesis, not tested separately here).

**Local vs aggregated features (supports research question 3).** Using only the 93 local features gives test PR-AUC 0.773 (Random Forest) / 0.750 (XGBoost); only the 72 aggregated features give 0.606 / 0.632; all 165 give 0.784 / 0.783. Most of the signal lives in the local features; the neighbour aggregates add a modest +0.01 to +0.03 PR-AUC.
We did not train a graph neural network, so we cannot say whether it would beat the tree models; we can say that the graph information supplied in tabular form is useful but not dominant."""),
], "03_feature_engineering.ipynb")

# ===================================================================== 04
nb([
    md("""# 04 · Models, validation and evaluation

## Split (chronological, no leakage)
| Period | Time steps | Role | Labelled rows |
|---|---|---|---|
| Train | 1 – 27 | fit models | 24,923 (2,457 illicit) |
| Validation | 28 – 34 | tune hyper-parameters, choose threshold & feature set | 4,971 (1,005 illicit) |
| Test | 35 – 49 | **touched once**, after all choices were made | 16,670 (1,083 illicit) |

Preprocessing (median imputation, standardisation) lives inside scikit-learn pipelines, so it is fitted on the rows passed to `fit()` only (unit-tested in `tests/`). Hyper-parameters are tuned with **expanding-window cross-validation** (train 1-13 → validate 14-20; train 1-20 → validate 21-27; train 1-27 → validate 28-34); test steps never enter. The decision threshold is the one maximising F1 on the validation period.

## Models and why they are included
| Model | Why | Principle | Key hyper-parameters (grid) |
|---|---|---|---|
| Majority baseline | Floor that every model must beat; shows why accuracy is misleading | Always predicts "licit" | none |
| Logistic Regression | Simple, interpretable linear model | Weighted sum of features passed through a sigmoid; class-weighted | C ∈ {0.01, 0.1, 1} |
| Random Forest | Captures non-linear interactions; robust to heavy tails | Many decorrelated decision trees vote; `class_weight="balanced_subsample"` | max_depth, min_samples_leaf, max_features |
| XGBoost | Strong tabular learner; fine control of imbalance | Trees added sequentially, each correcting the previous errors; `scale_pos_weight` = licit/illicit in training data | max_depth, learning_rate, n_estimators |

A graph neural network and an MLP are *optional* extensions that we did not run; see the limitations."""),
    code("""tune = {n: pd.read_csv(RES / f"tuning_{n}.csv") for n in ["logreg", "rf", "xgb"]}
for n, t in tune.items():
    print(n, "- top 3 of", len(t), "settings (mean PR-AUC over 3 expanding-window folds)"); display(t.head(3).round(4))"""),
    md("""**Insight.** Tuning changes little for the tree models: across the whole grid mean CV PR-AUC varies by only 0.013 (Random Forest) and 0.005 (XGBoost). Logistic Regression is clearly worse in every fold (e.g. fold PR-AUCs 0.83 / 0.56 / 0.64 vs 0.98 / 0.85 / 0.99 for Random Forest), so the gap between linear and tree models is not a tuning artefact. Its best setting (C = 1) lies at the edge of the grid; a larger C could be tried, but the margin to the tree models is far too large for that to matter."""),
    code("""val = pd.read_csv(RES / "validation_results.csv")
val[["model", "precision", "recall", "f1", "pr_auc", "threshold", "best_params"]].round(3)"""),
    md("""**Insight (validation, steps 28-34).** Random Forest (PR-AUC 0.989, F1 0.965) and XGBoost (0.985, 0.950) are close; Logistic Regression is far behind (0.636, 0.650). On the validation score alone, **Random Forest is the best model**: this is the rule fixed before looking at the test period."""),
    code("""res = pd.read_csv(RES / "test_results.csv")
display(res[res.threshold_rule == "val_tuned"][["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy", "tp", "fp", "fn", "tn"]].round(3))
print("Default threshold 0.5 for reference:")
res[res.threshold_rule == "default_0.5"][["model", "precision", "recall", "f1", "pr_auc"]].round(3)"""),
    md("""**Insight (test, steps 35-49, illicit class).**
* **Majority baseline**: accuracy 93.5% but recall 0 and F1 0: the textbook case of accuracy being misleading.
* **Logistic Regression**: recall 0.92 but precision 0.15 (5,734 false alarms for 996 hits), F1 0.255, PR-AUC 0.257. A linear boundary cannot separate the classes.
* **Random Forest**: precision 0.80, recall 0.72, F1 0.757, PR-AUC 0.784.
* **XGBoost**: precision 0.91, recall 0.68, F1 0.776, PR-AUC 0.783.
Both tree models beat the baseline and the linear model by a wide margin. Between them: the PR-AUC difference is 0.001 with a 95% bootstrap interval of [-0.005, +0.005] (a tie), while XGBoost's F1 is higher by about 0.02 (interval [0.007, 0.033]) because it raises far fewer false alarms (69 vs 201) at the cost of missing more illicit transactions (352 vs 301)."""),
    code("""show("eval_03_model_comparison", 800); show("eval_01_pr_roc_curves", 1000)"""),
    md("""**Insight.** The precision-recall curves of Random Forest and XGBoost overlap almost entirely and sit far above the linear model and the prevalence line (6.5% illicit in the test period); ROC curves look good for all models (AUC 0.86-0.93) even where the precision is poor, which is why ROC is only supplementary on imbalanced data."""),
    code("""show("eval_02_confusion_matrices", 1100)"""),
    md("""**Insight (false positives vs false negatives).** A false negative is a missed illicit transaction, a false positive is a wrongly flagged innocent one. Logistic Regression buys a few extra detections with thousands of false alarms; Random Forest and XGBoost sit at different points of the same trade-off (RF: 201 FP / 301 FN; XGBoost: 69 FP / 352 FN). Which is better depends on the cost of an investigator's time versus the cost of missing crime."""),
    code("""show("eval_05_threshold_tradeoff", 650)"""),
    md("""**Insight.** Raising the threshold reduces false alarms and increases misses, and vice versa. The validation-tuned threshold (dashed) is a compromise that maximises F1; an investigator team with limited capacity might choose a higher threshold, a compliance team that cannot tolerate misses a lower one."""),
    md("""### Research question 1: how much does a random split overstate performance?"""),
    code("""rt = pd.read_csv(RES / "random_vs_temporal.csv")
show("eval_04_random_vs_temporal", 850)
rt.pivot(index="model", columns="split", values=["f1", "pr_auc"]).round(3)"""),
    md("""**Insight.** With *identical* models and the same labelled data, a random split reports F1 0.945 (Random Forest) and 0.955 (XGBoost) against 0.757 and 0.776 for the chronological split, and PR-AUC 0.98-0.99 against 0.78. Logistic Regression drops from 0.77 to 0.26. A random split lets transactions from the same period (and their strongly related neighbours) appear in both training and test data, so the model is graded on situations it has effectively seen. **The temporal split is the honest estimate**, because in deployment a model must score transactions that occur after its training data."""),
    md("""### Which model is best?
* **By the pre-declared validation rule: Random Forest** (highest validation PR-AUC and F1).
* **On the test period**, Random Forest and XGBoost are statistically tied on PR-AUC; XGBoost has the higher F1 and precision (fewer false alarms), Random Forest the higher recall.
* We use **XGBoost as the default model in the demo application** because it gives fewer false alarms and supports exact per-prediction feature contributions. This is a judgement made after seeing the test results; it does not change the model-selection result above, and the app offers all three models.
* Neither model should be trusted after step 43 without retraining (notebook 05)."""),
], "04_model_training.ipynb")

# ===================================================================== 05
nb([
    md("""# 05 · Concept drift analysis

**Questions.** Where does performance collapse (RQ2)? Can retraining or removing unstable features reduce the damage (RQ4)? Which features matter and does that change? What do the errors look like?

> Step 43 is commonly linked to the shutdown of a major darknet market. This is a **plausible hypothesis, not something this dataset can prove**: we observe *when* behaviour changes, not *why*."""),
    code("""from src import plots
ps = pd.read_csv(RES / "per_step_metrics.csv")
pst = ps[ps.period.isin(["val", "test"])]
plots.fig_per_step(pst, ["logreg", "rf", "xgb"], "f1", "drift_01_per_step_f1"); show("drift_01_per_step_f1", 1000)"""),
    md("""**Insight (RQ2).** All models are strong from step 28 to 42 and then collapse at **step 43**. Random Forest and XGBoost score per-step F1 between 0.63 and 0.95 for steps 35-42, and then 0.00 at step 43, near zero (0.00-0.06) in steps 44-49 apart from a single noisy spike at step 46, which has only 2 illicit transactions. Logistic Regression is poor throughout."""),
    code("""plots.fig_per_step_prf(pst, "xgb"); show("drift_02_per_step_prf_xgb", 1000)
t = pst[(pst.period == "test") & (pst.model == "xgb")].set_index("time_step")[["precision", "recall", "f1", "n_illicit"]].round(2); t.T"""),
    md("""**Insight.** Before step 43 the XGBoost model has precision 0.87-0.98 and recall between 0.50 and 0.97; after it, recall is 0 at steps 43, 45, 47, 48 and about 0.02-0.04 at steps 44 and 49 (step 46 shows 0.50, but on only 2 illicit transactions). It still has alarms (low false-positive counts), but it simply does not recognise the illicit transactions of the new period. The bars show that several late steps contain few illicit transactions, so single-step values must be read with caution; the pooled picture (3 of 169 illicit transactions caught in steps 43-49) is the reliable one."""),
    code("""show("drift_01b_per_step_recall", 1000)"""),
    md("""**Insight.** Recall tells the same story as F1: the loss is a loss of *detection*, not a flood of false positives."""),
    md("""### Why do the models fail after step 43? Error analysis"""),
    code("""show("err_01_errors_by_step_xgb", 1000)
oc = pd.read_parquet(RES / "xgb_test_outcomes.parquet")
display(oc.groupby([oc.time_step >= 43, "outcome"]).size().unstack(fill_value=0).rename(index={False: "steps 35-42", True: "steps 43-49"}))
fn, tp = oc[oc.outcome == "FN"], oc[oc.outcome == "TP"]
print("mean illicit probability  caught (TP): %.3f | missed (FN): %.3f | share of FN with probability < 0.1: %.0f%%" % (tp.proba.mean(), fn.proba.mean(), 100 * (fn.proba < 0.1).mean()))"""),
    md("""**Insight (XGBoost, test period).**
* Of its 352 false negatives, **166 (47%) fall in steps 43-49** even though those steps contain only 169 of the 1,083 illicit test transactions (16%): it catches 3 of 169 there, versus 728 of 914 before step 43.
* **False negatives are confident misses**: 79% of them received an illicit probability below 0.1 (mean 0.08, against 0.975 for caught ones). The model does not sit on the fence; the late-period illicit transactions look licit to it, which is the signature of concept drift rather than of noise.
* False positives are rare and stable (69 in total, about 2-12 per step), so the failure is one of recall."""),
    code("""prof = pd.read_csv(RES / "outcome_feature_profile_z.csv", index_col=0)
prof.loc[["local_53", "local_55", "local_90", "local_41", "local_59"]].round(2)"""),
    md("""**Insight.** On the most discriminative features (shown as z-scores against the training period), caught illicit transactions (TP) are far below average (-1.00 on `local_53`), licit transactions are near the middle (-0.31), and the **missed illicit transactions (FN) lie in between (-0.62)**: they look partly like licit ones, so the learned decision boundary no longer isolates them. We do not interpret what these features measure."""),
    code("""ag = json.loads((RES / "error_agreement_bootstrap.json").read_text()); ag"""),
    md("""**Do models fail in the same places?** Largely yes for misses: all 301 illicit transactions missed by Random Forest are also missed by XGBoost (Jaccard overlap 0.86), so the two tree models share the same blind spot, consistent with a data-level cause (drift) rather than a model quirk. For false alarms they differ: only 40 transactions are false positives for both (Jaccard 0.17), so the models disagree about which licit transactions look suspicious."""),
    md("""### Does retraining reduce the damage? (RQ4)
We walk forward through the test period. For each step *t* the model is (re)trained on labelled data from earlier steps only and then scores step *t*:
* **Static**: trained once on steps 1-27;
* **Expanding window**: all labelled steps before *t*;
* **Sliding window**: the last 10 labelled steps before *t*.

*Assumption:* labels for step *t-1* are available when scoring step *t*. This is optimistic (confirming illicit activity takes time). All use the same hyper-parameters and a fixed threshold of 0.5 for a fair comparison."""),
    code("""show("drift_03_retraining_xgb", 1100)
rs = pd.read_csv(RES / "retraining_summary.csv")
rs[["model", "strategy", "period", "precision", "recall", "f1", "pr_auc"]].round(3)"""),
    md("""**Insight.**
* Pooled over steps 35-49, retraining raises XGBoost F1 from 0.771 (static) to **0.873 (expanding)** and **0.864 (sliding)**, and PR-AUC from 0.783 to 0.915 / 0.913. For Random Forest, F1 rises from 0.693 to 0.820 / 0.810.
* **In the collapsed period (steps 43-49)** XGBoost's recall goes from 0.018 to 0.391 and F1 from 0.027 to 0.543 (expanding) / 0.532 (sliding): retraining recovers *some* ability once labels from the new regime arrive. Random Forest at threshold 0.5 still misses nearly everything there (recall 0.018) although its PR-AUC rises from 0.045 to 0.565, i.e. its ranking improves but its probabilities stay too low for the fixed threshold.
* Sliding and expanding windows perform alike (differences of about 0.01 F1); the sliding window is cheaper and forgets old behaviour but has less data. Expanding is slightly better on average here; neither fully restores pre-collapse performance (F1 0.54 vs about 0.9 earlier).
* **Trade-offs:** needs a steady supply of fresh labels (an operational cost), repeated retraining, and monitoring. Drift is **reduced, not solved.**"""),
    md("""### Does removing unstable features help? (RQ4)
Features were ranked by their Kolmogorov-Smirnov shift between the training (1-27) and validation (28-34) periods (no test data used), and the most-shifted fraction was dropped before fitting."""),
    code("""show("drift_04_unstable_features", 700)
u = pd.read_csv(RES / "unstable_feature_removal.csv")
u.pivot_table(index=["model", "dropped_fraction"], columns="period", values=["f1", "pr_auc"]).round(3)"""),
    md("""**Insight.** Dropping the 10% most-shifted features improved validation F1 for both models (RF 0.965 → 0.967, XGBoost 0.950 → 0.971) and test F1 (RF 0.757 → 0.787, XGBoost 0.776 → 0.791); dropping 50% hurt XGBoost (test F1 0.709). Since validation also prefers the 10% setting, this is a legitimate modest gain of about 0.015-0.03 F1, but it is small compared with the collapse after step 43 and is **not a fix for the drift**: those steps are untouched by this change."""),
    md("""### Feature importance and its stability"""),
    code("""show("xai_01_importance", 1100)
imp = pd.read_csv(RES / "importance_tree.csv", index_col=0)
imp.sort_values("xgb_gain", ascending=False).head(8).round(4)"""),
    md("""**Insight.** XGBoost relies heavily on a few local features (`local_41` alone carries 18% of the total gain; `local_53` 9%, `local_47` 7%), plus aggregated features `agg_10` and `agg_7`. Random Forest spreads importance more widely over the correlated block `local_41/43/47/49/53/55`. Permutation importance (drop in PR-AUC on validation data) ranks `local_59`, `local_53` and `local_90` highest, so different methods emphasise different members of correlated groups: individual rankings should not be over-interpreted, and the columns have no known real-world meaning."""),
    code("""show("xai_02_importance_over_time", 900)
pd.read_csv(RES / "importance_over_time_spearman.csv", index_col=0).round(2)"""),
    md("""**Insight (do features stop mattering?).** Yes. Features that matter before step 43 lose power afterwards: permutation importance of `local_59` halves (0.021 → 0.009), `local_90` falls from 0.012 to 0.003, `local_52` and `agg_70` go to about zero, and `local_41` becomes *negative* (-0.006): shuffling it slightly *improves* PR-AUC, i.e. the model's use of it now misleads. In addition, the rank correlation of XGBoost's importance between models trained on different periods is only 0.32-0.40, so which features matter itself changes over time."""),
    md("""## Conclusions on the research questions
1. **Random vs temporal split**: a random split overstates F1 by 0.18-0.19 for the tree models (0.945/0.955 vs 0.757/0.776) and by 0.51 for Logistic Regression.
2. **Per-step collapse**: strong through step 42, collapse at step 43 (3 of 169 illicit transactions caught in steps 43-49). The cause appears to be a changed relationship between features and label rather than input shift alone; linking it to a darknet-market shutdown is a hypothesis.
3. **Tree vs graph model**: tree models are strong (test PR-AUC 0.78). Local features carry most of the signal; the supplied aggregated neighbour features add 0.01-0.03 PR-AUC. We did not train a GNN, so the tree-vs-GNN comparison remains open.
4. **Reducing drift damage**: retraining on recent labelled data helps substantially (F1 0.77 → 0.87 pooled; 0.03 → 0.54 in steps 43-49 for XGBoost) but needs fresh labels and does not restore early performance; dropping the 10% most-shifted features gives a small gain.
5. **Extension (Elliptic++ wallet addresses)**: not attempted (optional, time-permitting)."""),
], "05_concept_drift_analysis.ipynb")
print("notebooks written")
