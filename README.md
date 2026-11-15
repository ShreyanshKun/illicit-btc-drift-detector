# Detecting Illicit Bitcoin Transactions Under Concept Drift

Introduction to Data Science for Engineers: project (Part A).
Can a machine-learning model tell whether a Bitcoin transaction is **illicit** or **licit**, and does that ability survive when criminal behaviour changes over time (**concept drift**)? The project uses the public **Elliptic** dataset (203,769 transactions, 234,355 edges, 49 time steps, 165 anonymised features used) and a leakage-free **chronological** evaluation.

## Headline results

Test period = time steps 35–49 (never used for any choice), illicit class, thresholds chosen on validation steps 28–34.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Majority baseline | 0.000 | 0.000 | 0.000 | 0.065 |
| Logistic Regression | 0.148 | 0.920 | 0.255 | 0.257 |
| Random Forest | 0.796 | 0.722 | 0.757 | 0.784 |
| XGBoost | 0.914 | 0.675 | 0.776 | 0.783 |

* A **random split** overstates F1 by about 0.18 (RF 0.945 vs 0.757; XGBoost 0.955 vs 0.776).
* Performance **collapses at time step 43**: the static XGBoost model catches 3 of 169 illicit transactions in steps 43–49 (728 of 914 before). Linking this to a darknet-market shutdown is a hypothesis, not something the data proves.
* **Retraining** on recent labels raises pooled XGBoost F1 from 0.771 to 0.873 (recall after step 43: 0.018 → 0.391) but needs fresh labels and does not restore early performance.
* Random Forest and XGBoost tie on PR-AUC; Random Forest wins on the pre-declared validation rule, XGBoost has fewer false alarms and gives per-prediction explanations (used as the demo default; chosen after seeing test results).

Full write-up: [reports/REPORT.md](reports/REPORT.md) (also [REPORT.docx](reports/REPORT.docx)), slides [reports/Presentation.pptx](reports/Presentation.pptx), viva notes [docs/VIVA_PREP.md](docs/VIVA_PREP.md).

## Repository layout

```
data/raw/            Elliptic CSV files (not in git; see "Get the data")
data/processed/      master tables (not in git), app_sample.parquet + demo_upload.csv (small, in git)
notebooks/           01 data understanding · 02 EDA · 03 feature engineering · 04 models · 05 concept drift (executed)
src/                 data_loader, preprocessing, feature_engineering, splitting, models, evaluation,
                     drift_analysis, explainability, inference, plots, config
train.py             trains + evaluates all models on the temporal split, saves models/ and results
run_analysis.py      random-vs-temporal, ablations, drift, retraining, importance, error analysis
make_figures.py      regenerates every figure and the app's sample data
predict.py           command-line batch prediction
app/                 Streamlit application: top navigation, Three.js drift tunnel and 3-D transaction universe
                     (streamlit_app.py, ui.py theme, fx.py page effects, three_scenes.py, assets/logo.svg)
models/              saved models (*.joblib)
reports/             REPORT.md/.docx, Presentation.pptx, figures/, results/ (all CSV/JSON/parquet outputs)
tests/               17 unit tests
docs/                viva preparation, report/slide builders
```

## Get the data

The raw files are not committed (≈ 690 MB). Download the Elliptic dataset (e.g. from Elliptic's release, Kaggle `ellipticco/elliptic-data-set`, or the PyTorch Geometric mirror `https://data.pyg.org/datasets/elliptic/elliptic_txs_{features,classes,edgelist}.csv.zip`) and place the three unzipped files in `data/raw/`:

```
data/raw/elliptic_txs_features.csv
data/raw/elliptic_txs_classes.csv
data/raw/elliptic_txs_edgelist.csv
```

## Reproduce everything

Python 3.12/3.13. Seeds are fixed (42) and versions pinned.

```bash
pip install -r requirements-dev.txt
python train.py                        # ~5 min: tune, train, evaluate -> models/, reports/results/
python run_analysis.py                 # ~15 min: drift, retraining, ablations, importance, errors
python make_figures.py                 # figures + app sample data
python make_universe.py                # 3-D embedding used by the app's Universe page
python notebooks/build_notebooks.py && python notebooks/execute_notebooks.py   # optional
python -m pytest tests -q              # 17 tests
```

Re-fitting the saved configurations reproduces the stored predictions exactly (verified: maximum difference 0.0).

## Run the application

Models and result tables are committed, so the app starts without the raw data and **does not retrain anything**:

```bash
streamlit run app/streamlit_app.py
python predict.py --demo                                        # CLI check
python predict.py --input data/processed/demo_upload.csv --model xgb --output scores.csv
```

Pages: **Overview** (3-D drift tunnel: one glowing ring per time step), **Scanner**, **Universe** (orbitable 3-D cloud of real transactions with time-travel), **Model Lab** (threshold explorer), **Drift Radar** (model-health monitor, 3-D drift surface, retraining simulator), **Explainer**, **Field Guide**. The 3-D scenes use three.js from a CDN and need WebGL and an internet connection.

Upload format: a CSV with the 165 columns `local_1…local_93` and `agg_1…agg_72` (optional `txId`). Missing values are imputed with training medians; missing columns, non-numeric or infinite values are rejected with a clear message. **A score is a statistical signal, not proof of criminal activity.**

### Deployment

* **Streamlit Community Cloud:** push this repo to GitHub, then at share.streamlit.io choose the repository, branch `main` and main file `app/streamlit_app.py` (dependencies come from `requirements.txt`).
* **Docker:** `docker build -t illicit-btc . && docker run -p 8501:8501 illicit-btc`.

## Methodology in brief

1. Join the three files with verification; recode labels (1 illicit, 0 licit, −1 unknown); exclude unknown from supervised work.
2. EDA with interpretation under every plot; document each cleaning decision.
3. Feature engineering (degrees, neighbour means, step size) using same-step information only; kept only if validation shows a gain (it did not).
4. Chronological split: train 1–27, validation 28–34, test 35–49; preprocessing fitted on training rows only; tuning by expanding-window cross-validation; threshold chosen on validation.
5. Models: majority baseline, Logistic Regression, Random Forest, XGBoost.
6. Illicit-class precision/recall/F1/PR-AUC, confusion matrices, per-time-step evaluation, error analysis, importance (impurity/gain, permutation, TreeSHAP contributions).
7. Drift: static vs expanding vs sliding retraining, unstable-feature removal.

## Limitations

Anonymised features (no real-world meaning is claimed); 77% of transactions unlabelled; labels from one company; a single time window; the step-43 explanation is a hypothesis; few illicit cases in late steps make single-step scores noisy; the retraining experiment assumes the previous step's labels are immediately available; graph neural network, MLP and the Elliptic++ wallet-address extension were optional and **not implemented**.

## References

Weber et al. (2019), *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*, arXiv:1908.02591 · Elmougy & Liu (2023), Elliptic++, arXiv:2306.06108 · Breiman (2001) Random Forests · Chen & Guestrin (2016) XGBoost · Pedregosa et al. (2011) scikit-learn · Gama et al. (2014) *A survey on concept drift adaptation*.
