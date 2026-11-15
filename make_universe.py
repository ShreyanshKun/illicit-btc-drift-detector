"""Build reports/results/universe.json: a 3-D PCA embedding of real labelled transactions
for the app's "Transaction Universe" view.

The embedding (quantile-transform + PCA) is fitted on TRAINING steps (1-27) only. Points are
a stratified sample: up to 85 licit and 45 illicit transactions per time step.
Each point stores its step, true label and the XGBoost illicit probability.

Usage:  python make_universe.py
"""
import json

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

from src.config import MODELS_DIR, RAW_FEATURE_COLS, RES_DIR, SEED
from src.data_loader import labelled, load_master


def main():
    master, _ = load_master()
    df = labelled(master)
    train = df[df.time_step <= 27]
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED)
    qt.fit(train[RAW_FEATURE_COLS].to_numpy(dtype="float64"))
    pca = PCA(n_components=3, random_state=SEED).fit(qt.transform(train[RAW_FEATURE_COLS].to_numpy(dtype="float64")))

    rng = np.random.RandomState(SEED)
    parts = []
    for t, g in df.groupby("time_step"):
        ill, lic = g[g.label == 1], g[g.label == 0]
        parts += [ill.sample(min(45, len(ill)), random_state=rng), lic.sample(min(85, len(lic)), random_state=rng)]
    s = np.concatenate([p.index.values for p in parts])
    sample = df.loc[s]
    Z = pca.transform(qt.transform(sample[RAW_FEATURE_COLS].to_numpy(dtype="float64")))
    Z = Z / np.percentile(np.abs(Z), 98, axis=0) * 10          # scale to a pleasant viewing box
    bundle = joblib.load(MODELS_DIR / "xgb.joblib")
    p = bundle["model"].predict_proba(sample[RAW_FEATURE_COLS])[:, 1]
    out = {"x": np.round(Z[:, 0], 2).tolist(), "y": np.round(Z[:, 1], 2).tolist(), "z": np.round(Z[:, 2], 2).tolist(),
           "step": sample.time_step.astype(int).tolist(), "label": sample.label.astype(int).tolist(),
           "p": np.round(p, 3).tolist(), "thr": float(bundle["threshold"]),
           "explained": np.round(pca.explained_variance_ratio_, 3).tolist(), "n": int(len(sample))}
    (RES_DIR / "universe.json").write_text(json.dumps(out, separators=(",", ":")))
    print("points", out["n"], "explained variance", out["explained"], "illicit", sum(out["label"]))


if __name__ == "__main__":
    main()
