"""Build reports/Presentation.pptx from the real results and figures.

Usage:  python docs/build_slides.py
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
NAVY, RED, GREY, LIGHT = RGBColor(0x1F, 0x2A, 0x44), RGBColor(0xC0, 0x39, 0x2B), RGBColor(0x55, 0x5B, 0x66), RGBColor(0xF2, 0xF4, 0xF7)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def textbox(slide, x, y, w, h, text, size=18, bold=False, color=NAVY, align=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold, color
    if align:
        p.alignment = align
    return tb


def new_slide(title, notes=""):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RED
    bar.line.fill.background()
    textbox(s, 0.6, 0.35, 12.1, 0.9, title, 30, True)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def bullets(slide, items, x=0.6, y=1.5, w=6.0, h=5.4, size=18):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for k, it in enumerate(items):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        bold = it.startswith("**")
        txt = it.strip("*") if bold else it
        r = p.add_run()
        r.text = ("" if bold else "•  ") + txt
        r.font.size, r.font.bold = Pt(size), bold
        r.font.color.rgb = NAVY if bold else GREY
        p.space_after = Pt(9)


def image(slide, name, x, y, w, h):
    """Place a figure inside the box (x, y, w, h) preserving its aspect ratio."""
    path = FIG / f"{name}.png"
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, rows, x, y, w, col_w=None, size=14, row_h=0.42):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))).table
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cell = t.cell(r, c)
            cell.text = str(v)
            para = cell.text_frame.paragraphs[0]
            para.runs[0].font.size = Pt(size)
            para.runs[0].font.bold = r == 0
            if c > 0:
                para.alignment = PP_ALIGN.CENTER
    if col_w:
        for c, cw in enumerate(col_w):
            t.columns[c].width = Inches(cw)


# 1 ------------------------------------------------------------------ title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
textbox(s, 0.9, 2.2, 11.5, 1.6, "Detecting Illicit Bitcoin Transactions Under Concept Drift", 44, True, RGBColor(255, 255, 255))
textbox(s, 0.9, 4.2, 11.5, 0.8, "Introduction to Data Science for Engineers · Project (Part A)", 22, False, RGBColor(0xD0, 0xD6, 0xE2))
textbox(s, 0.9, 5.0, 11.5, 0.8, "Elliptic Bitcoin dataset · temporal evaluation · drift analysis", 18, False, RGBColor(0xD0, 0xD6, 0xE2))
s.notes_slide.notes_text_frame.text = "One-sentence pitch: we ask not only whether machine learning can find illicit Bitcoin transactions, but whether it keeps working when criminals change behaviour."

# 2 ------------------------------------------------------------------ problem
s = new_slide("Problem: can a model spot crime, and does it keep working?",
              "Binary classification of transactions: illicit vs licit. The second half of the question (drift) is what makes this more than a routine classifier.")
bullets(s, ["**The question**",
            "Can a model tell whether a Bitcoin transaction is illicit or licit?",
            "Does that ability survive when criminal behaviour changes over time (concept drift)?",
            "**Why it is hard**",
            "Illicit transactions are rare (9.8% of labelled)",
            "77% of transactions have no label",
            "166 anonymised features, no meaning attached",
            "False negatives and false positives cost differently → accuracy is the wrong metric"], w=7.2)
bullets(s, ["**Research questions**", "1. Random vs temporal split: how big is the gap?", "2. Where does performance collapse over time?",
            "3. Do aggregated neighbour features explain tree-model strength?", "4. Does retraining / dropping unstable features help?"], x=8.2, w=4.6, size=17)

# 3 ------------------------------------------------------------------ dataset
s = new_slide("Dataset: Elliptic Bitcoin transaction graph",
              "Real, public, published with a peer-reviewed paper (Weber et al. 2019). Node = transaction, edge = flow of coins. Edges never cross time steps, so graph features cannot leak the future.")
table(s, [["Item", "Value"], ["Transactions (nodes)", "203,769"], ["Payment flows (edges)", "234,355"], ["Time steps", "49 (~2 weeks each)"],
          ["Features used", "165 = 93 local + 72 aggregated"], ["Labelled illicit", "4,545 (2.2%)"], ["Labelled licit", "42,019 (20.6%)"],
          ["Unknown (no label)", "157,205 (77.1%)"]], 0.6, 1.6, 6.2, [3.0, 3.2], 16, 0.5)
image(s, "eda_01_class_distribution", 7.1, 1.6, 5.8, 3.3)
bullets(s, ["Only the 46,564 labelled transactions are used for supervised learning; unknown labels are excluded, not assumed licit."], x=7.1, y=5.2, w=5.8, h=1.5, size=15)

# 4 ------------------------------------------------------------------ cleaning
s = new_slide("Data preparation: every decision explained",
              "Walk through each: problem, why it matters, action, why appropriate. Emphasise that outliers were kept on purpose.")
table(s, [["Finding", "Action"],
          ["77% unlabelled", "Excluded from train/val/test and metrics"],
          ["No missing values or duplicate ids", "Median imputer kept inside pipelines (train-fit only)"],
          ["34 duplicate feature rows", "Kept: same step, no label conflicts, no leakage"],
          ["Heavy tails (median 13% beyond 3×IQR)", "Kept: extremes are mostly licit; LR clipped at ±5 SD"],
          ["Time step column", "Used for splitting only, never as a feature"],
          ["Edges stay within a time step", "Graph features cannot see the future"],
          ["9.8% illicit among labelled", "Class weights + illicit-class metrics"]], 0.6, 1.6, 12.1, [5.3, 6.8], 16, 0.58)

# 5 ------------------------------------------------------------------ EDA 1
s = new_slide("EDA: the illicit share is not constant",
              "Illicit share ranges from 0.3% to 36% and falls to 2.5% after step 43. The base rate itself drifts, so thresholds tuned on the 20%-illicit validation period transfer imperfectly.")
image(s, "eda_03_illicit_rate_over_time", 0.5, 1.4, 8.3, 4.6)
bullets(s, ["Train: 9.9% illicit", "Validation: 20.2%", "Steps 35–42: 9.2%", "Steps 43–49: **2.5%** (169 illicit)", "A first sign that the future differs from the past"], x=9.0, y=1.8, w=4.0, size=17)

# 6 ------------------------------------------------------------------ EDA 2
s = new_slide("EDA: separable, heavy-tailed, and drifting inputs",
              "Single features reach AUC 0.89 but no single feature is decisive. Input drift is gradual (KS rises from 0.113 to 0.166), no jump at step 43.")
image(s, "eda_04_feature_distributions", 0.4, 1.4, 6.4, 3.8)
image(s, "eda_10_feature_drift_ks", 6.9, 1.4, 6.1, 3.0)
bullets(s, ["Illicit transactions sit at the low end of the top features (univariate AUC up to 0.89)",
            "Outliers are mostly licit → kept, not removed",
            "Mean KS shift vs training: 0.113 → 0.142 → 0.164 → 0.166; share of drifting features doubles (11% → 21%)",
            "Input drift is gradual: the later collapse points to a changed feature–label relationship"], x=0.6, y=5.3, w=12.3, h=2.0, size=15)

# 7 ------------------------------------------------------------------ method
s = new_slide("Method: chronological split, no leakage",
              "Training on the past, tuning on the next period, testing once on the future. Preprocessing is inside pipelines fitted on training rows only. Tuning uses expanding-window CV that never touches test steps.")
for k, (lbl, steps, n, col) in enumerate([("Train", "steps 1–27", "24,923 rows", RGBColor(0x2C, 0x6F, 0xBB)),
                                           ("Validation", "steps 28–34", "4,971 rows", RGBColor(0xE0, 0x8E, 0x0B)),
                                           ("Test (touched once)", "steps 35–49", "16,670 rows", RED)]):
    w = [5.0, 2.6, 4.2][k]
    x = 0.6 + sum([5.0, 2.6, 4.2][:k]) + 0.0
    box = s.shapes.add_shape(1, Inches(x), Inches(1.6), Inches(w - 0.1), Inches(1.1))
    box.fill.solid(); box.fill.fore_color.rgb = col; box.line.fill.background()
    tf = box.text_frame
    tf.text = f"{lbl}\n{steps} · {n}"
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size, r.font.color.rgb, r.font.bold = Pt(16), RGBColor(255, 255, 255), True
bullets(s, ["Unknown labels removed before splitting; split asserted strictly chronological",
            "Imputation / scaling fitted on training rows only (unit-tested)",
            "Hyper-parameters tuned with expanding-window CV (train 1–13→14–20, 1–20→21–27, 1–27→28–34)",
            "Decision threshold (F1-optimal) chosen on validation data",
            "Time step never used as a feature; class weights from training labels only",
            "Random seed 42; re-running reproduces stored predictions exactly"], y=3.1, w=12.2, size=18)

# 8 ------------------------------------------------------------------ models
s = new_slide("Models: from trivial to advanced, each with a reason",
              "Baseline plus three approaches. A GNN/MLP were optional and not implemented; we say so openly.")
table(s, [["Model", "Why included", "Key settings"],
          ["Majority baseline", "Floor; shows accuracy misleads", "—"],
          ["Logistic Regression", "Simple, interpretable linear model", "C=1, class-weighted, clip ±5 SD"],
          ["Random Forest", "Non-linear, robust to heavy tails", "300 trees, balanced weights, √p features"],
          ["XGBoost", "Strong boosted trees; fine imbalance control", "depth 8, rate 0.1, 200 trees, scale_pos_weight"]],
      0.6, 1.7, 12.1, [2.8, 4.6, 4.7], 16, 0.7)
bullets(s, ["Tuning is cheap: grid varies PR-AUC by only 0.013 (RF) and 0.005 (XGBoost)", "Logistic Regression is worse in every validation fold"], y=5.4, w=12, size=17)

# 9 ------------------------------------------------------------------ results
s = new_slide("Results on the held-out future (steps 35–49)",
              "All numbers are for the illicit class. Majority baseline: 93.5% accuracy, zero recall. RF and XGBoost tie on PR-AUC (bootstrap CI of difference includes 0); XGBoost has higher F1 through fewer false alarms.")
table(s, [["Model", "Precision", "Recall", "F1", "PR-AUC", "Accuracy"],
          ["Majority baseline", "0.000", "0.000", "0.000", "0.065", "0.935"],
          ["Logistic Regression", "0.148", "0.920", "0.255", "0.257", "0.651"],
          ["Random Forest", "0.796", "0.722", "0.757", "0.784", "0.970"],
          ["XGBoost", "0.914", "0.675", "0.776", "0.783", "0.975"]], 0.6, 1.6, 7.4, [2.6, 1.0, 0.9, 0.9, 1.0, 1.0], 14, 0.55)
bullets(s, ["Baseline: 93.5% accuracy but catches no crime",
            "RF vs XGBoost: PR-AUC tie; XGBoost raises fewer false alarms (69 vs 201), RF misses fewer (301 vs 352)",
            "Best by pre-declared validation rule: Random Forest",
            "Demo default: XGBoost (explanations) — a post-hoc judgement"], y=4.5, w=7.4, h=2.8, size=15)
image(s, "eval_02_confusion_matrices", 8.2, 1.7, 4.9, 2.0)
image(s, "eval_01_pr_roc_curves", 8.2, 3.9, 4.9, 2.4)

# 10 ----------------------------------------------------------------- random vs temporal
s = new_slide("Research Q1: a random split lies",
              "Same models, same data. A random split puts near-copies of test transactions in the training set. Temporal split is the honest estimate.")
image(s, "eval_04_random_vs_temporal", 0.5, 1.4, 7.8, 4.4)
bullets(s, ["Random split F1: RF 0.945, XGBoost 0.955", "Temporal split F1: RF 0.757, XGBoost 0.776", "Overstatement ≈ +0.18–0.19 (trees), +0.51 (logistic regression)",
            "Deployment scores the future, so only the temporal number is honest"], x=8.6, y=1.8, w=4.4, size=17)

# 11 ----------------------------------------------------------------- drift
s = new_slide("Research Q2: performance collapses at time step 43",
              "Strong through step 42; then near-zero recall. Treat the darknet-shutdown explanation as a hypothesis. Several late steps have very few illicit transactions so single-step values are noisy; the pooled figure (3 of 169) is the reliable one.")
image(s, "drift_02_per_step_prf_xgb", 0.4, 1.3, 8.3, 4.5)
bullets(s, ["Steps 35–42: per-step F1 0.63–0.95", "Step 43: F1 = 0, then ≈ 0 in steps 44–49", "Static XGBoost catches **3 of 169** illicit transactions after step 43 (728 of 914 before)",
            "A loss of detection, not a flood of false alarms", "Darknet shutdown: plausible hypothesis, not proven"], x=8.9, y=1.5, w=4.2, size=15)

# 12 ----------------------------------------------------------------- errors
s = new_slide("Error analysis: confident misses",
              "47% of false negatives fall in steps 43-49, which hold 16% of illicit test transactions. 79% of FNs have probability < 0.1: the model is confident, not uncertain, which is how drift looks.")
image(s, "err_01_errors_by_step_xgb", 0.4, 1.4, 7.8, 4.2)
bullets(s, ["352 false negatives: **166 (47%)** in steps 43–49", "79% of misses scored < 0.1 illicit probability (caught ones: 0.975)",
            "RF and XGBoost miss the same illicit transactions (Jaccard 0.86)", "Missed transactions sit between licit and caught-illicit on the top features",
            "False alarms are rare and stable (69 total)"], x=8.4, y=1.5, w=4.7, size=15)

# 13 ----------------------------------------------------------------- retraining
s = new_slide("Research Q4: retraining helps, drift is not solved",
              "Walk-forward: for each step, retrain on earlier steps only. Assumes labels of the previous step are available (optimistic). Fixed threshold 0.5 for fairness.")
image(s, "drift_03_retraining_xgb", 0.3, 1.3, 7.9, 3.1)
table(s, [["XGBoost", "F1 35–49", "PR-AUC", "Recall 43–49", "F1 43–49"], ["Static", "0.771", "0.783", "0.018", "0.027"],
          ["Expanding", "0.873", "0.915", "0.391", "0.543"], ["Sliding (10)", "0.864", "0.913", "0.391", "0.532"]], 0.6, 4.7, 7.4, [1.9, 1.3, 1.2, 1.7, 1.3], 14, 0.45)
bullets(s, ["Pooled F1 0.77 → 0.87; after step 43 recall 0.02 → 0.39", "Needs fresh confirmed labels, repeated training, monitoring", "Does not restore early performance (F1 0.54 vs ≈ 0.9)",
            "Dropping the 10% most-shifted features: small gain (+0.015–0.03 F1)"], x=8.4, y=1.5, w=4.7, size=15)

# 14 ----------------------------------------------------------------- features
s = new_slide("Which features matter — and do they keep mattering?",
              "Features are anonymised, so we report which columns matter, never what they mean. After step 43 some features' importance drops to zero or turns negative.")
image(s, "xai_02_importance_over_time", 0.4, 1.4, 7.4, 4.3)
bullets(s, ["Top features (XGBoost gain): local_41 (18%), local_53, local_47, agg_10, agg_7", "After step 43 `local_59` importance halves; `local_52`, `agg_70` ≈ 0; `local_41` turns negative",
            "Importance rank correlation across training periods only 0.32–0.40", "Local features carry most signal; neighbour aggregates add +0.01–0.03 PR-AUC",
            "Features have no known real-world meaning"], x=8.0, y=1.5, w=5.1, size=15)

# 15 ----------------------------------------------------------------- conclusion
s = new_slide("Conclusions, limitations and next steps",
              "Finish by tying back to the question: models work in-regime, drift breaks them, retraining recovers part. Be upfront that GNN/MLP/Elliptic++ were optional and not done.")
bullets(s, ["**Findings**", "Tree models detect illicit transactions well in-regime (test PR-AUC 0.78); a random split would have hidden how fragile that is",
            "At step 43 detection collapses (3 of 169): the feature–label relationship changed",
            "Retraining recovers part of the loss but needs fresh labels",
            "**Use**", "Decision support for investigators, with drift monitoring and retraining; never proof of crime"], x=0.6, w=6.3, size=16)
bullets(s, ["**Limitations**", "Anonymised features; 77% unlabelled; one company's labels; one time window",
            "Step-43 cause is a hypothesis; late steps have few illicit cases",
            "Retraining assumes immediate labels; demo model chosen post hoc",
            "GNN, MLP and Elliptic++ extension not implemented",
            "**Next**", "GCN/GraphSAGE comparison; Elliptic++ wallet data; automatic drift triggers; probability calibration"], x=7.1, w=5.9, size=16)

# 16 ----------------------------------------------------------------- demo
s = new_slide("Live demo: Streamlit application",
              "Show: Dashboard, upload demo CSV (data/processed/demo_upload.csv), single-transaction explanation (XGBoost), Concept Drift page. Remind the audience that a prediction is not proof.")
bullets(s, ["**Pages**", "Overview (3-D drift tunnel) · Scanner · Universe (3-D explorer) · Model Lab · Drift Radar · Explainer · Field Guide",
            "**Transaction Detection**", "Upload a CSV (input validation, batch scoring, download), pick a sample transaction, or edit features; risk score, class and per-feature contributions",
            "**Run locally**", "streamlit run app/streamlit_app.py",
            "Models are loaded from disk — nothing is retrained at start-up", "Every prediction carries the disclaimer: a statistical signal, not proof of criminal activity"], size=18, w=12)

OUT = ROOT / "reports" / "Presentation.pptx"
prs.save(OUT)
print("wrote", OUT, "with", len(prs.slides), "slides")
