"""Streamlit demo: Detecting Illicit Bitcoin Transactions Under Concept Drift.

Everything shown here is read from artefacts produced by train.py / run_analysis.py
(models/*.joblib, reports/results/*). Nothing is retrained when the app starts.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import precision_recall_curve

import fx
import three_scenes
import ui
from src.config import PROC_DIR, RAW_FEATURE_COLS, RES_DIR
from src.evaluation import per_step_metrics
from src.explainability import xgb_contributions
from src.inference import (DISCLAIMER, InputValidationError, load_bundle, predict_frame,
                           validate_features)

st.set_page_config(page_title="ChainSight · Illicit Bitcoin Detection", page_icon=":material/radar:", layout="wide")
ui.inject_css()
fx.inject()

LABELS = {"majority": "Majority baseline", "logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
MCOLOR = {"Majority baseline": "#8b8b9c", "Logistic Regression": ui.GOLD, "Random Forest": ui.SKY, "XGBoost": ui.BTC}
BLURB = {
    "majority": ("The lazy baseline", "Always answers “licit”. Looks 93% accurate and catches no crime. It exists to prove accuracy is a misleading score."),
    "logreg": ("The straight-line thinker", "Adds up the features with weights. Simple and explainable, but it cannot capture how features combine, so it raises thousands of false alarms."),
    "rf": ("The committee of trees", "300 decision trees each vote on a transaction. Strong, steady, and rarely misses (highest recall of the good models)."),
    "xgb": ("The self-correcting forest", "Trees built one after another, each fixing the previous mistakes. Fewest false alarms and gives exact reasons for each prediction."),
}


# ------------------------------------------------------------------ loaders -----------
@st.cache_resource
def get_bundle(name):
    return load_bundle(name)


@st.cache_data
def csv(name, **kw):
    return pd.read_csv(RES_DIR / name, **kw)


@st.cache_data
def parquet(name):
    return pd.read_parquet(RES_DIR / name)


@st.cache_data
def sample():
    return pd.read_parquet(PROC_DIR / "app_sample.parquet")


@st.cache_data
def eda_stats():
    return json.loads((RES_DIR / "eda_stats.json").read_text())


def model_confusion(y, p, thr):
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


# ------------------------------------------------------------------ explanation helpers
def describe_level(pct):
    return ("very low" if pct <= 10 else "low" if pct <= 30 else "typical" if pct < 70
            else "high" if pct < 90 else "very high")


def reasons(bundle, X_row, k=6):
    """Top-k feature contributions in plain words (XGBoost TreeSHAP values)."""
    contrib, _ = xgb_contributions(bundle["model"], X_row)
    s = contrib.iloc[0]
    top = s.reindex(s.abs().sort_values(ascending=False).head(k).index)
    ref = sample()[RAW_FEATURE_COLS]
    rows = []
    for f, c in top.items():
        v = float(X_row[f].iloc[0])
        pct = float((ref[f] < v).mean() * 100)
        rows.append(dict(feature=f, value=v, contribution=float(c), pct=pct))
    return rows


def render_reasons(rows):
    html = ""
    for r in rows:
        toward = r["contribution"] > 0
        col = ui.ILLICIT if toward else ui.LICIT
        word = "toward <b>illicit</b>" if toward else "toward <b>licit</b>"
        html += (f'<div class="reason"><span class="dot" style="background:{col}"></span>'
                 f'<span><b>{r["feature"]}</b> is <b>{describe_level(r["pct"])}</b> '
                 f'(higher than {r["pct"]:.0f}% of sampled transactions) &rarr; pushes the score {word} '
                 f'<span class="small">({r["contribution"]:+.2f} log-odds)</span></span></div>')
    st.markdown(html, unsafe_allow_html=True)


def gauge(p, thr, key):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Illicit risk score"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": ui.ILLICIT if p >= thr else ui.LICIT, "thickness": 0.3},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, thr * 100], "color": "rgba(52,211,153,0.18)"},
                         {"range": [thr * 100, 100], "color": "rgba(255,71,87,0.20)"}],
               "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.85, "value": thr * 100}}))
    fig = ui.style(fig, 270, False)
    fig.update_layout(margin=dict(l=30, r=40, t=50, b=10))
    st.plotly_chart(fig, width="stretch", key=key)
    st.markdown(f'<div class="small" style="text-align:center;margin-top:-14px">White line = decision threshold '
                f'({thr:.0%}). Above it the model says <b>illicit</b>.</div>', unsafe_allow_html=True)


def single_result(bundle, name, row_df, thr, actual=None, key="single"):
    out = predict_frame(row_df, name, thr, bundle)
    p = float(out["illicit_probability"].iloc[0])
    bad = p >= thr
    ui.steps([("Validated", "165 numeric features found"), ("Filled gaps", "none needed" if not row_df[bundle["features"]].isna().any().any() else "medians used"),
              ("Scored", f"{LABELS[name]} computed a probability"), ("Compared", f"{p:.1%} vs threshold {thr:.0%}")], done_upto=3)
    c1, c2 = st.columns([1, 1.15])
    with c1:
        gauge(p, thr, f"{key}_gauge")
    with c2:
        if bad:
            st.markdown(f'<div class="verdict bad"><div class="t">⚠ Flagged as likely illicit</div><div class="d">'
                        f'The {LABELS[name]} gives this transaction a <b>{p:.1%}</b> chance of being illicit, above the '
                        f'{thr:.0%} alarm line. An investigator should look at it. This is a lead, not proof.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict good"><div class="t">✓ Looks like normal (licit) activity</div><div class="d">'
                        f'Only a <b>{p:.1%}</b> chance of being illicit, below the {thr:.0%} alarm line. '
                        f'Remember: after time step 43 the model can miss new kinds of illicit behaviour.</div></div>', unsafe_allow_html=True)
        if actual is not None:
            right = (actual == 1) == bad
            st.markdown(f'Recorded label in the dataset: <span class="pill {"bad" if actual == 1 else "good"}">'
                        f'{"illicit" if actual == 1 else "licit"}</span> &nbsp; The model was <span class="pill {"good" if right else "warn"}">'
                        f'{"correct" if right else "wrong"}</span> on this one.', unsafe_allow_html=True)
    if actual == 1 and not bad and "time_step" in row_df.columns and int(row_df["time_step"].iloc[0]) >= 43:
        ui.callout(f"<b>Concept drift in action.</b> This really is an illicit transaction from time step {int(row_df['time_step'].iloc[0])}, "
                   "after the model's world changed. It scores as ordinary because its pattern no longer resembles the illicit examples the model learned from.", "bad")
    if name == "xgb":
        X = validate_features(row_df, bundle["features"])
        rows = reasons(bundle, X)
        st.markdown("**Why this score? The 6 features that moved it most**")
        render_reasons(rows)
        top = pd.DataFrame(rows).sort_values("contribution")
        fig = go.Figure(go.Bar(x=top["contribution"], y=top["feature"], orientation="h",
                               marker_color=[ui.ILLICIT if v > 0 else ui.LICIT for v in top["contribution"]]))
        fig.update_layout(xaxis_title="push toward illicit (+) or licit (−), in log-odds")
        st.plotly_chart(ui.style(fig, 300, False), width="stretch", key=f"{key}_bar")
        st.caption("The 165 features are anonymised by the dataset authors, so we can say *which* columns drove the score and "
                   "whether the value is high or low, but not what they mean in the real world.")
    else:
        st.info("Switch to **XGBoost** to see which features drove this score.")


# ------------------------------------------------------------------ pages -------------
def page_overview():
    cls = {"t": 0, "g": 1, "a": 2, "r": 3, "n": 4}
    components.html(three_scenes.tunnel_html([cls[c] for c, _ in chain_blocks()]), height=560)
    c = eda_stats()["label_counts"]
    res = csv("test_results.csv")
    best = res[(res.threshold_rule == "val_tuned") & (res.model == "xgb")].iloc[0]
    ui.kpis([("Transactions", f"{sum(c.values()):,}", "nodes in the Bitcoin graph", "neutral"),
             ("Known illicit", f"{c['Illicit']:,}", "2.2% of everything", "bad"),
             ("Known licit", f"{c['Licit']:,}", "20.6% of everything", "good"),
             ("No label", f"{c['Unknown']:,}", "77% — excluded from training", "warn"),
             ("Best test PR-AUC", f"{best['pr_auc']:.2f}", "on the unseen future", "good")])
    st.markdown("&nbsp;")
    ui.section("The 49 blocks of time", "Each block is two weeks of Bitcoin activity. Hover for details; colours show how well the model coped")
    st.markdown(f'<div class="chainbox">{ui.chain(chain_blocks(), big=True, current=42)}{ui.chain_legend()}</div>', unsafe_allow_html=True)
    st.markdown("&nbsp;")
    ui.section("How this project works", "From raw data to a drift-aware verdict, in five steps")
    ui.steps([("Clean", "Join three files, drop unlabelled rows from training, keep real outliers"),
              ("Split by time", "Learn from steps 1–27, tune on 28–34, test once on 35–49"),
              ("Teach 4 models", "Baseline, Logistic Regression, Random Forest, XGBoost"),
              ("Judge honestly", "Illicit-class precision, recall, F1, PR-AUC — not accuracy"),
              ("Watch for drift", "Track performance per step; test retraining")], done_upto=4)
    left, right = st.columns([1.15, 1])
    with left:
        ps = csv("per_step_metrics.csv")
        d = ps[(ps.model == "xgb") & ps.period.isin(["val", "test"])]
        fig = go.Figure()
        fig.add_vrect(x0=27.5, x1=34.5, fillcolor=ui.GOLD, opacity=0.08, line_width=0)
        fig.add_vrect(x0=34.5, x1=49.5, fillcolor=ui.ILLICIT, opacity=0.06, line_width=0)
        fig.add_scatter(x=d.time_step, y=d.f1, mode="lines+markers", name="F1", line=dict(color=ui.BTC, width=3))
        fig.add_vline(x=43, line_dash="dash", line_color="white", annotation_text="step 43", annotation_font_color="white")
        fig.update_layout(title="Headline: XGBoost F1 on illicit transactions, per step",
                          xaxis_title="Time step", yaxis_title="F1 on illicit class", yaxis_range=[0, 1.02])
        st.plotly_chart(ui.style(fig, 360, False), width="stretch", key="ov_f1")
    with right:
        ui.callout("<b>The big finding.</b> Up to step 42 the model finds most illicit transactions. At <b>step 43</b> it collapses: "
                   "in steps 43–49 it catches <b>3 of 169</b> illicit transactions. Criminal behaviour changed and the old patterns stopped working "
                   "(concept drift). A darknet-market shutdown is a plausible reason — a hypothesis, not something this data proves.", "bad")
        ui.callout("<b>Why it matters.</b> A random train/test split would have hidden this and reported F1 ≈ 0.95 instead of the honest ≈ 0.78.", "warn")
        ui.callout("<b>What helps.</b> Retraining on recent labelled data lifts F1 from 0.77 to 0.87, but needs fresh labels and does not fully fix it.", "good")
    ui.section("Where to go next")
    cols = st.columns(4)
    for col, (t, b) in zip(cols, [("Scanner", "Upload a CSV or pick a transaction and get a risk score with reasons."),
                                  ("Model Lab", "Compare models and drag the threshold to see the false-alarm trade-off."),
                                  ("Drift Radar", "Step through time and watch the model's health change."),
                                  ("Field Guide", "Plain-English guide, glossary and limitations.")]):
        col.markdown(ui.card(t, b), unsafe_allow_html=True)


@st.cache_data
def universe_data():
    return json.loads((RES_DIR / "universe.json").read_text())


def page_universe():
    ui.hero("3-D explorer", "The transaction universe",
            "5,906 real transactions, squeezed from 165 dimensions into 3 so you can fly around them. Drag to orbit, scroll to zoom, hover for details, and scrub through time.")
    ui.explain("A sample of real transactions (up to 85 licit and 45 illicit per time step). Each dot is one transaction; nearby dots have similar features.",
               "Switch the colouring: <b>Truth</b> (is it really illicit?), <b>Model verdict</b> (what XGBoost got right or wrong) or <b>Period</b>. Drag the time slider or press play to watch the data grow step by step.",
               "In <b>Model verdict</b> view, scrub past step 43 and watch red 'missed' dots appear inside the licit cloud: illicit transactions the model can no longer tell apart.")
    d = universe_data()
    components.html(three_scenes.universe_html(d), height=700)
    X = np.c_[d["x"], d["y"], d["z"]]
    step, lab, p = np.array(d["step"]), np.array(d["label"]), np.array(d["p"])

    def shift(cls):
        a, b = X[(step <= 27) & (lab == cls)], X[(step >= 43) & (lab == cls)]
        return float(np.linalg.norm(a.mean(0) - b.mean(0)))
    late_ill = (step >= 43) & (lab == 1)
    early_ill = (step >= 35) & (step <= 42) & (lab == 1)
    miss_late = float((p[late_ill] < d["thr"]).mean())
    miss_early = float((p[early_ill] < d["thr"]).mean())
    ui.kpis([("Illicit cloud shift", f"{shift(1):.1f}", "train to steps 43-49 (PCA units)", "bad"),
             ("Licit cloud shift", f"{shift(0):.1f}", "same measurement for licit", "good"),
             ("Missed, steps 35-42", f"{miss_early:.0%}", f"of {int(early_ill.sum())} sampled illicit", "warn"),
             ("Missed, steps 43-49", f"{miss_late:.0%}", f"of {int(late_ill.sum())} sampled illicit", "bad")])
    ui.callout(f"<b>How to read the numbers.</b> The centroid of the illicit transactions moved <b>{shift(1):.1f}</b> units between the training period and steps 43-49, "
               f"against <b>{shift(0):.1f}</b> for licit ones. These three axes capture only {sum(d['explained']):.0%} of the variance, so this view illustrates drift rather than proving it; "
               "the evidence is the per-step performance on the Drift Radar page. The embedding was fitted on training steps only, and the model saw steps 1-27 during training, "
               "so dots from those steps are not a fair test.", "info")


def page_scan():
    ui.hero("Detection", "Scan transactions for illicit risk",
            "Give the model a transaction (or thousands) and it returns a risk score, a verdict and — for XGBoost — the reasons.")
    ui.callout(DISCLAIMER, "warn")
    ui.explain("A risk score from 0–100% for each transaction: the model's estimated chance that it is illicit.",
               "Above the white threshold line the model flags it. A lower threshold catches more crime but raises more false alarms.",
               "Use the score to prioritise what a human investigator checks first, never as proof.")
    c1, c2 = st.columns([2, 1])
    name = c1.selectbox("Model", ["xgb", "rf", "logreg"], format_func=LABELS.get,
                        help="XGBoost also explains each prediction. Random Forest misses fewer cases. Logistic Regression is the weakest.")
    bundle = get_bundle(name)
    thr = c2.slider("Alarm threshold", 0.05, 0.95, float(round(bundle["threshold"], 2)), 0.01,
                    help="Default = the value that maximised F1 on the validation period (steps 28–34).")
    t_up, t_pick, t_wi = st.tabs(["📁 Upload CSV (batch)", "🎯 Pick a sample transaction", "🧪 What-if: edit features"])

    with t_up:
        ui.steps([("Upload", "CSV with 165 feature columns"), ("Validate", "columns, numbers, no infinities"), ("Score", "model returns probabilities"), ("Review", "sorted list + download")])
        demo = PROC_DIR / "demo_upload.csv"
        if demo.exists():
            st.download_button("⬇ Download a ready-made demo CSV (200 test-period transactions)", demo.read_bytes(), "demo_upload.csv", "text/csv")
        st.caption("Required columns: `local_1 … local_93` and `agg_1 … agg_72`. Optional `txId` is carried through. Missing cells are filled with training medians.")
        up = st.file_uploader("Drop your CSV here", type="csv")
        if up is not None:
            try:
                df = pd.read_csv(up)
                out = predict_frame(df, name, thr, bundle)
            except InputValidationError as e:
                st.error(f"Cannot score this file: {e}")
            except Exception as e:
                st.error(f"Could not read the file as CSV: {e}")
            else:
                n_missing = int(df[RAW_FEATURE_COLS].isna().sum().sum())
                n_ill = int((out["predicted_class"] == "illicit").sum())
                ui.steps([("Validated", f"{len(df):,} rows × 165 columns OK"), ("Filled gaps", f"{n_missing:,} missing cells → medians"),
                          ("Scored", f"{LABELS[name]} on every row"), ("Flagged", f"{n_ill:,} above {thr:.0%}")], done_upto=3)
                ui.kpis([("Rows scored", f"{len(out):,}", "", "neutral"), ("Flagged illicit", f"{n_ill:,}", f"{n_ill / len(out):.1%} of rows", "bad"),
                         ("Highest risk", f"{out.illicit_probability.max():.1%}", "", "bad"), ("Median risk", f"{out.illicit_probability.median():.2%}", "", "good")])
                fig = px.histogram(out, x="illicit_probability", nbins=40, color="predicted_class",
                                   color_discrete_map={"illicit": ui.ILLICIT, "licit": ui.LICIT}, title="How risky does this batch look?")
                fig.add_vline(x=thr, line_dash="dash", line_color="white")
                st.plotly_chart(ui.style(fig, 300), width="stretch", key="batch_hist")
                st.caption("Most transactions should pile up near 0%. The bars right of the dashed line are the flagged ones.")
                st.markdown("**Most suspicious first**")
                st.dataframe(out.sort_values("illicit_probability", ascending=False), width="stretch", hide_index=True,
                             column_config={"illicit_probability": st.column_config.ProgressColumn("Illicit probability", min_value=0, max_value=1, format="%.3f")})
                st.download_button("⬇ Download predictions (CSV)", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

    smp = sample()
    with t_pick:
        st.caption("Labelled transactions from the unseen test period (steps 35–49). Illicit cases are over-represented here so the demo is interesting.")
        f = st.radio("Show", ["All", "Illicit only", "Licit only"], horizontal=True)
        pool = smp if f == "All" else smp[smp.label == (1 if f == "Illicit only" else 0)]
        tx = st.selectbox("Transaction (time step)", pool["txId"].tolist()[:300],
                          format_func=lambda t: f"{t}  ·  step {int(pool.loc[pool.txId == t, 'time_step'].iloc[0])}")
        row = pool[pool.txId == tx].iloc[[0]]
        single_result(bundle, name, row, thr, int(row["label"].iloc[0]), key="pick")

    with t_wi:
        st.write("Start from a real transaction, change a few features, and watch the score react.")
        base_tx = st.selectbox("Template transaction", smp["txId"].tolist()[:100], key="tmpl")
        base = smp[smp.txId == base_tx].iloc[[0]].copy()
        imp = csv("importance_tree.csv", index_col=0)["xgb_gain"].sort_values(ascending=False)
        feats = st.multiselect("Features to edit (most influential first)", imp.index.tolist()[:30], default=imp.index.tolist()[:3])
        cols = st.columns(3)
        for i, f_ in enumerate(feats):
            base[f_] = cols[i % 3].number_input(f_, value=float(base[f_].iloc[0]), format="%.5f", key=f"in_{f_}")
        single_result(bundle, name, base, thr, key="whatif")


def page_models():
    ui.hero("Evaluation", "Four models, one fair test",
            "Every model learned from steps 1–27 and was tested on the unseen future (steps 35–49). Scores are for the illicit class only.")
    cols = st.columns(4)
    for col, n in zip(cols, ["majority", "logreg", "rf", "xgb"]):
        col.markdown(ui.card(f"{LABELS[n]}<br><span class='small'>{BLURB[n][0]}</span>", BLURB[n][1]), unsafe_allow_html=True)
    st.markdown("&nbsp;")
    res, val = csv("test_results.csv"), csv("validation_results.csv")
    t = res[res.threshold_rule == "val_tuned"].copy()
    ui.explain("Precision = of the transactions flagged, how many were truly illicit. Recall = of all illicit ones, how many were caught.",
               "F1 balances both. PR-AUC summarises quality across every possible threshold. Higher is better for all four.",
               "Notice the baseline: 93.5% accuracy but 0 recall. That is why accuracy is not used.")
    tt = t.assign(model=t.model.map(LABELS))
    m = tt.melt(id_vars="model", value_vars=["precision", "recall", "f1", "pr_auc"], var_name="metric")
    fig = px.bar(m, x="metric", y="value", color="model", barmode="group", color_discrete_map=MCOLOR, title="Test-period scores (illicit class)")
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(ui.style(fig, 360), width="stretch", key="mod_bar")

    ui.section("Threshold explorer", "Drag the alarm line and see the trade-off between false alarms and missed crime on the real test data")
    pred = parquet("test_predictions.parquet")
    c1, c2 = st.columns([1, 2])
    with c1:
        mname = st.selectbox("Model", ["xgb", "rf", "logreg"], format_func=LABELS.get, key="te_model")
        default_thr = float(round(get_bundle(mname)["threshold"], 2))
        thr = st.slider("Alarm threshold", 0.02, 0.98, default_thr, 0.01, key=f"te_thr_{mname}")
        cm = model_confusion(pred.label.values, pred[mname].values, thr)
        ui.kpis([("Caught", f"{cm['tp']:,}", f"of {cm['tp'] + cm['fn']:,} illicit", "good")])
        ui.kpis([("Missed", f"{cm['fn']:,}", "illicit slipped through", "bad")])
        ui.kpis([("False alarms", f"{cm['fp']:,}", f"of {cm['fp'] + cm['tn']:,} licit", "warn")])
    with c2:
        ui.callout(f"At <b>{thr:.0%}</b>, {LABELS[mname]} catches <b>{cm['recall']:.0%}</b> of illicit transactions (recall) and "
                   f"<b>{cm['precision']:.0%}</b> of what it flags is truly illicit (precision) &rarr; F1 <b>{cm['f1']:.2f}</b>. "
                   f"Lower the line to catch more crime; raise it to bother investigators less.", "info")
        z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig = px.imshow(z, text_auto=True, x=["Model says licit", "Model says illicit"], y=["Actually licit", "Actually illicit"],
                        color_continuous_scale=[[0, "#101c18"], [1, ui.LICIT_D]], title="The four outcomes")
        fig.update_coloraxes(showscale=False)
        fig.update_traces(texttemplate="%{z:,}", textfont_size=18)
        st.plotly_chart(ui.style(fig, 300, False), width="stretch", key="te_cm")
    p, r, th = precision_recall_curve(pred.label, pred[mname])
    fig = go.Figure()
    for n in ["logreg", "rf", "xgb"]:
        pp, rr, _ = precision_recall_curve(pred.label, pred[n])
        fig.add_scatter(x=rr, y=pp, name=LABELS[n], line=dict(color=MCOLOR[LABELS[n]], width=3 if n == mname else 1.5), opacity=1 if n == mname else 0.5)
    fig.add_scatter(x=[cm["recall"]], y=[cm["precision"]], mode="markers", name="your threshold", marker=dict(size=14, color="white", line=dict(width=2, color=ui.ILLICIT)))
    fig.update_layout(title="Precision–recall curves: you are the white dot", xaxis_title="Recall (crime caught)", yaxis_title="Precision (alarms that were right)")
    st.plotly_chart(ui.style(fig, 360), width="stretch", key="te_pr")

    t1, t2, t3 = st.tabs(["Full table", "Validation vs test", "Why a random split lies"])
    with t1:
        st.dataframe(tt[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy", "tp", "fp", "fn", "tn"]], width="stretch", hide_index=True)
    with t2:
        v = val.assign(model=val.model.map(LABELS))
        st.dataframe(v[["model", "precision", "recall", "f1", "pr_auc"]], width="stretch", hide_index=True)
        ui.callout("Random Forest scores PR-AUC 0.989 on validation but 0.784 on test. That drop <b>is</b> concept drift showing up in the numbers.", "warn")
    with t3:
        rvt = csv("random_vs_temporal.csv")
        rvt = rvt.assign(model=rvt.model.map(LABELS))
        fig = px.bar(rvt, x="model", y="f1", color="split", barmode="group", color_discrete_map={"random": ui.SKY, "temporal": ui.ILLICIT},
                     title="Same models, same data — only the split changes")
        st.plotly_chart(ui.style(fig, 340), width="stretch", key="rvt")
        ui.callout("A random split lets the model peek at neighbours from the same period it is tested on, so it looks ~0.18 F1 better than it will be in real use. "
                   "Testing on the <b>future</b> is the honest way.", "info")


def page_drift():
    ui.hero("Concept drift", "Watch a good model stop working",
            "Move through time. See the model's health, what changed in the data, and whether retraining brings it back.")
    ui.explain("Performance of the models on each time step (about two weeks of Bitcoin activity).",
               "Steps 28–34 are validation, 35–49 the unseen future. The dashed line at step 43 is where trouble starts.",
               "A model is only trustworthy while its health stays green. Monitoring this is how real systems catch drift.")
    ps = csv("per_step_metrics.csv")
    ps = ps[ps.period.isin(["val", "test"])].copy()
    ui.section("Model health monitor", "Pick a time step and see how the model is doing right then")
    c1, c2 = st.columns([1, 2])
    with c1:
        mname = st.selectbox("Model", ["xgb", "rf", "logreg"], format_func=LABELS.get, key="dr_model")
        step = st.slider("Time step", 28, 49, 40)
        d = ps[(ps.model == mname) & (ps.time_step == step)].iloc[0]
        if d.n_illicit < 10:
            status, tone, msg = "TOO FEW CASES", "warn", f"Only {int(d.n_illicit)} illicit transactions this step, so the score is unreliable."
        elif d.f1 >= 0.7:
            status, tone, msg = "HEALTHY", "good", "The model is finding most illicit transactions."
        elif d.f1 >= 0.3:
            status, tone, msg = "DEGRADED", "warn", "Many illicit transactions are slipping through."
        else:
            status, tone, msg = "FAILING", "bad", "The model is missing almost all illicit transactions."
        st.markdown(f'<div class="verdict {"good" if tone == "good" else "bad"}"><div class="t">Step {step}: {status}</div><div class="d">{msg}</div></div>', unsafe_allow_html=True)
        ui.kpis([("F1", f"{d.f1:.2f}" if d.f1 == d.f1 else "n/a", "", tone)])
        ui.kpis([("Recall", f"{d.recall:.2f}" if d.recall == d.recall else "n/a", f"{int(d.n_illicit)} illicit in step", tone)])
    with c2:
        models = st.multiselect("Compare models", ["Logistic Regression", "Random Forest", "XGBoost"], default=["Random Forest", "XGBoost"])
        metric = st.radio("Metric", ["f1", "recall", "precision"], horizontal=True, format_func=str.upper)
        dd = ps.assign(model=ps.model.map(LABELS))
        dd = dd[dd.model.isin(models)]
        fig = px.line(dd, x="time_step", y=metric, color="model", markers=True, color_discrete_map=MCOLOR)
        fig.add_vrect(x0=27.5, x1=34.5, fillcolor=ui.GOLD, opacity=0.08, line_width=0, annotation_text="validation", annotation_position="top left")
        fig.add_vline(x=43, line_dash="dash", line_color="white", annotation_text="step 43")
        fig.add_vline(x=step, line_color=ui.SKY, opacity=0.6)
        fig.update_layout(title=f"{metric.upper()} per time step (models trained once, on steps 1–27)", xaxis_title="Time step", yaxis_range=[-0.02, 1.02])
        st.plotly_chart(ui.style(fig, 360), width="stretch", key="dr_line")
    ui.callout("In steps 43–49 the static XGBoost model catches <b>3 of 169</b> illicit transactions (versus 728 of 914 before). "
               "Late steps hold few illicit cases, so single-step values jump around; the pooled count is the reliable number.", "bad")

    ui.section("What changed in the data?")
    t1, t2, t3, t4 = st.tabs(["Illicit share", "Input drift", "Anatomy of the misses", "Drift surface (3-D)"])
    with t1:
        lab = csv("label_counts_per_step.csv")
        lab["share"] = lab.illicit / (lab.illicit + lab.licit)
        fig = px.area(lab, x="time_step", y="share", color_discrete_sequence=[ui.ILLICIT])
        fig.add_vline(x=43, line_dash="dash", line_color="white")
        fig.update_layout(title="Share of labelled transactions that are illicit", yaxis_tickformat=".0%", xaxis_title="Time step")
        st.plotly_chart(ui.style(fig, 320, False), width="stretch", key="dr_share")
        st.caption("Illicit activity fell from ~20% (validation) to 2.5% after step 43. The mix of transactions itself changed.")
    with t2:
        ks = csv("feature_drift_ks.csv", index_col=0)
        fig = px.line(x=ks.index, y=ks.mean(axis=1), markers=True, color_discrete_sequence=[ui.GOLD])
        fig.add_vline(x=43, line_dash="dash", line_color="white")
        fig.update_layout(title="Average distance of every feature from its training-period distribution (KS)", xaxis_title="Time step", yaxis_title="KS statistic")
        st.plotly_chart(ui.style(fig, 320, False), width="stretch", key="dr_ks")
        st.caption("The inputs drift slowly and steadily (0.11 → 0.17), with no jump at step 43. So the sudden collapse suggests the *meaning* of the patterns changed, not just the numbers.")
    with t3:
        oc = parquet("xgb_test_outcomes.parquet")
        fn = oc[oc.outcome == "FN"]
        tp = oc[oc.outcome == "TP"]
        fig = go.Figure()
        fig.add_histogram(x=tp.proba, name="caught", marker_color=ui.LICIT, nbinsx=25, opacity=0.8)
        fig.add_histogram(x=fn.proba, name="missed", marker_color=ui.ILLICIT, nbinsx=25, opacity=0.8)
        fig.update_layout(barmode="overlay", title="Probability the model gave to illicit transactions", xaxis_title="Predicted illicit probability", yaxis_title="Transactions")
        st.plotly_chart(ui.style(fig, 320), width="stretch", key="dr_miss")
        st.caption(f"79% of the {len(fn)} missed illicit transactions got a probability under 10%: the model was confidently wrong, "
                   "the typical signature of concept drift.")

    with t4:
        ks3 = csv("feature_drift_ks.csv", index_col=0)
        ks3 = ks3[ks3.mean(axis=0).sort_values(ascending=False).index]
        fig = go.Figure(go.Surface(z=ks3.values, x=list(ks3.columns), y=list(ks3.index),
                                   colorscale=[[0, "#15151c"], [0.35, "#7a3b06"], [0.7, ui.BTC], [1, ui.ILLICIT]],
                                   colorbar=dict(title="KS"), contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True)))))
        fig.update_layout(scene=dict(xaxis_title="features (most drifting first)", yaxis_title="time step", zaxis_title="KS shift",
                                     xaxis=dict(showticklabels=False, backgroundcolor="rgba(0,0,0,0)", gridcolor="#2a2a35"),
                                     yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#2a2a35"),
                                     zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#2a2a35"),
                                     camera=dict(eye=dict(x=1.7, y=-1.6, z=0.9))),
                          title="How far every feature has drifted from its training distribution")
        st.plotly_chart(ui.style(fig, 520, False), width="stretch", key="dr_surface")
        st.caption("Rotate it. A handful of features (the ridge on the left) are far from their training distribution in every period; "
                   "for many others the shift grows towards the later steps (front of the surface).")

    ui.section("Can retraining fix it?", "Walk forward in time. Before scoring each step, retrain on earlier steps only (assumes last step's labels are known).")
    c1, c2 = st.columns([1, 2])
    with c1:
        rm = st.radio("Model", ["xgb", "rf"], format_func=LABELS.get, horizontal=True, key="rt_model")
        picks = st.multiselect("Strategies", ["Static (train 1-27)", "Expanding window", "Sliding window (10 steps)"],
                               default=["Static (train 1-27)", "Expanding window"])
        mm = st.selectbox("Metric", ["f1", "pr_auc", "recall", "precision"], format_func=str.upper)
    pred = parquet("test_predictions.parquet")
    runs = {"Static (train 1-27)": pred[["txId", "time_step", "label"]].assign(proba=pred[rm]),
            "Expanding window": parquet(f"retrain_{rm}_expanding.parquet"),
            "Sliding window (10 steps)": parquet(f"retrain_{rm}_sliding.parquet")}
    rows = []
    for s, r in runs.items():
        if s in picks:
            x = per_step_metrics(r.time_step, r.label, r.proba, 0.5).reset_index()
            x["strategy"] = s
            rows.append(x)
    with c2:
        if rows:
            fig = px.line(pd.concat(rows), x="time_step", y=mm, color="strategy", markers=True,
                          color_discrete_map={"Static (train 1-27)": "#8b8b9c", "Expanding window": ui.LICIT, "Sliding window (10 steps)": ui.ILLICIT})
            fig.add_vline(x=43, line_dash="dash", line_color="white")
            fig.update_layout(title=f"{LABELS[rm]}: {mm.upper()} per step (threshold 0.5)", xaxis_title="Time step", yaxis_range=[-0.02, 1.02])
            st.plotly_chart(ui.style(fig, 340), width="stretch", key="rt_line")
    summ = csv("retraining_summary.csv")
    summ = summ[(summ.model == rm)][["strategy", "period", "precision", "recall", "f1", "pr_auc"]]
    st.dataframe(summ, width="stretch", hide_index=True)
    ui.callout("<b>Verdict:</b> retraining helps a lot (pooled F1 0.77 → 0.87 for XGBoost; recall after step 43 0.02 → 0.39) but needs a steady supply "
               "of fresh, confirmed labels, and F1 after step 43 is still only ≈ 0.54. Drift is <b>reduced, not solved</b>. "
               "Random Forest at a fixed 0.5 threshold barely recovers even though its ranking (PR-AUC) improves.", "warn")


def page_xai():
    ui.hero("Explainability", "What is the model actually looking at?",
            "Which of the 165 features drive decisions, and do they keep driving them after the world changes?")
    ui.callout("<b>Why can't we name the features?</b> The dataset authors anonymised all 165 columns to protect privacy and the data owner's methods. "
               "We can show <i>which</i> columns matter and whether a value is high or low, but we will never claim to know what they measure.", "info")
    tree = csv("importance_tree.csv", index_col=0)
    perm = csv("importance_permutation.csv", index_col=0)
    ui.explain("How strongly each feature influences the model's decisions.",
               "Longer bar = the model relies on it more. 'Permutation' shuffles one feature and measures how much accuracy drops.",
               "A few features carry most of the signal, and correlated features share credit, so rankings are indicative, not exact.")
    n = st.slider("How many features to show", 5, 30, 12)
    c1, c2 = st.columns(2)
    for col, key, title, color in [(c1, "xgb_gain", "XGBoost: share of total gain", ui.ILLICIT), (c2, "rf_impurity", "Random Forest: impurity importance", ui.SKY)]:
        t = tree[key].sort_values().tail(n)
        fig = go.Figure(go.Bar(x=t.values, y=t.index, orientation="h", marker_color=color))
        fig.update_layout(title=title)
        col.plotly_chart(ui.style(fig, 420, False), width="stretch", key=f"imp_{key}")
    ui.section("Do features keep mattering after step 43?")
    top = perm["xgb_test_35-42"].sort_values(ascending=False).head(n).index
    d = perm.loc[top, ["xgb_test_35-42", "xgb_test_43-49"]].rename(columns={"xgb_test_35-42": "steps 35–42", "xgb_test_43-49": "steps 43–49"})
    fig = px.bar(d.reset_index().melt(id_vars="feature"), x="feature", y="value", color="variable", barmode="group",
                 color_discrete_sequence=[ui.LICIT, ui.ILLICIT], labels={"value": "accuracy lost when shuffled", "variable": ""})
    st.plotly_chart(ui.style(fig, 340), width="stretch", key="imp_time")
    ui.callout("Features that mattered before step 43 lose power afterwards, and <b>local_41 turns negative</b>: shuffling it slightly <i>improves</i> the score, "
               "meaning the model's old reliance on it now misleads it. This is drift seen from the feature side.", "warn")
    ui.section("Explain one transaction")
    smp = sample()
    tx = st.selectbox("Transaction", smp["txId"].tolist()[:200])
    row = smp[smp.txId == tx].iloc[[0]]
    b = get_bundle("xgb")
    single_result(b, "xgb", row, b["threshold"], int(row["label"].iloc[0]), key="xai")


def page_learn():
    ui.hero("Learn", "The whole project in plain English", "Everything the app assumes, explained without jargon.")
    t1, t2, t3, t4 = st.tabs(["The pipeline", "Glossary", "FAQ", "Limits & honesty"])
    with t1:
        ui.steps([("Load", "3 files: features, labels, edges. Every join verified."), ("Clean", "Unknown labels excluded; outliers kept on purpose"),
                  ("Engineer", "Degrees & neighbour means tested — no gain, so dropped"), ("Split", "Past → train, next → tune, future → test"),
                  ("Tune", "Expanding-window validation; threshold chosen on validation"), ("Test once", "Future steps 35–49"),
                  ("Analyse drift", "Per-step scores, retraining, error analysis")], done_upto=6)
        fig = go.Figure()
        for lo, hi, name, col in [(1, 27, "Train (24,923 rows)", ui.SKY), (28, 34, "Validation (4,971)", ui.GOLD), (35, 49, "Test (16,670) — touched once", ui.ILLICIT)]:
            fig.add_bar(x=[hi - lo + 1], y=["Time"], base=lo - 1, orientation="h", name=name, marker_color=col, text=f"steps {lo}–{hi}", textposition="inside")
        fig.update_layout(barmode="stack", title="Chronological split: learn from the past, judge on the future", xaxis_title="Time step", yaxis_visible=False)
        st.plotly_chart(ui.style(fig, 230), width="stretch", key="split")
        st.markdown("**No leakage** means the model never sees information it could not have had at prediction time: unknown labels removed first, "
                    "scaling fitted on training rows only, tuning never touches the test steps, the time step is not an input, and neighbour labels are never used.")
    with t2:
        gl = {"Illicit / licit": "Criminal / legitimate. Illicit is the 'positive' class we try to find.",
              "Concept drift": "The link between a transaction's features and whether it is illicit changes over time, so old patterns stop working.",
              "Precision": "Of everything flagged illicit, the share that really is. High = few false alarms.",
              "Recall": "Of all truly illicit transactions, the share caught. High = little crime slips through.",
              "F1": "A single score that balances precision and recall.", "PR-AUC": "Quality across every possible threshold; the standard score for rare-event problems.",
              "Threshold": "The probability above which the model raises an alarm.", "False positive / negative": "False alarm / missed illicit transaction.",
              "Data leakage": "Accidentally letting the model see information from the test period, which makes results look better than reality.",
              "Temporal split": "Training on earlier data and testing on later data.", "KS statistic": "How different two distributions are (0 = same, 1 = completely different).",
              "Retraining window": "Expanding = all past data; sliding = only the most recent steps."}
        for k, v in gl.items():
            st.markdown(f"**{k}** — {v}")
    with t3:
        for q, a in [("Why not just report accuracy?", "Only ~10% of labelled transactions are illicit. Saying 'licit' every time gets ~93% accuracy and catches nothing."),
                     ("Is the darknet shutdown the cause of step 43?", "It is a plausible hypothesis. The data shows when behaviour changed, not why."),
                     ("Which model should I trust?", "Random Forest and XGBoost tie on PR-AUC. XGBoost raises fewer false alarms; Random Forest misses fewer. Neither is reliable after step 43 without retraining."),
                     ("Why wasn't a graph neural network built?", "It was an optional extension. The aggregated neighbour features already give tabular models some graph information (+0.01 to +0.03 PR-AUC), but we cannot say whether a GNN would do better."),
                     ("Does retraining solve drift?", "It helps substantially but needs fresh confirmed labels and does not restore early performance.")]:
            with st.expander(q):
                st.write(a)
    with t4:
        for b in ["Features are anonymised: no real-world meaning is claimed.", "77% of transactions are unlabelled; labels come from one company's methods.",
                  "One time window of 49 steps; findings may not hold for Bitcoin today.", "Late steps contain few illicit cases, so per-step scores are noisy.",
                  "Validation had 20% illicit vs 6.5% in test, so thresholds transfer imperfectly.",
                  "The retraining experiment assumes the previous step's labels are available immediately.",
                  "GNN, MLP and the Elliptic++ wallet-address extension were not implemented.",
                  "The demo default (XGBoost) was chosen after seeing test results; by the pre-declared validation rule Random Forest is best."]:
            st.markdown(f"- {b}")
        ui.callout(DISCLAIMER, "warn")


def chain_blocks():
    """One block per time step: training steps dim, steps 28-49 coloured by XGBoost F1 health."""
    ps = csv("per_step_metrics.csv")
    d = ps[(ps.model == "xgb") & ps.period.isin(["val", "test"])].set_index("time_step")
    blocks = []
    for t in range(1, 50):
        if t < 28:
            blocks.append(("t", f"Step {t} · training"))
        else:
            r = d.loc[t]
            if r.n_illicit < 10:
                blocks.append(("n", f"Step {t} · only {int(r.n_illicit)} illicit cases"))
            else:
                cls = "g" if r.f1 >= 0.7 else "a" if r.f1 >= 0.3 else "r"
                blocks.append((cls, f"Step {t} · F1 {r.f1:.2f}"))
    return blocks


st.logo(str(Path(__file__).resolve().parent / "assets" / "logo.svg"), size="large")
pages = [st.Page(page_overview, title="Overview", icon=":material/home:", url_path="overview", default=True),
         st.Page(page_scan, title="Scanner", icon=":material/radar:", url_path="scanner"),
         st.Page(page_universe, title="Universe", icon=":material/blur_on:", url_path="universe"),
         st.Page(page_models, title="Model Lab", icon=":material/science:", url_path="model-lab"),
         st.Page(page_drift, title="Drift Radar", icon=":material/monitoring:", url_path="drift-radar"),
         st.Page(page_xai, title="Explainer", icon=":material/psychology:", url_path="explainer"),
         st.Page(page_learn, title="Field Guide", icon=":material/menu_book:", url_path="field-guide")]
st.navigation(pages, position="top").run()
st.markdown('<div class="small" style="text-align:center;margin:36px 0 8px 0">Scores are statistical signals, not proof of criminal activity. '
            'Intro to Data Science for Engineers · Elliptic dataset</div>', unsafe_allow_html=True)
