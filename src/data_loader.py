"""Load the three Elliptic files and build one clean master table."""
import pandas as pd

from .config import (RAW_DIR, PROC_DIR, LOCAL_COLS, AGG_COLS, ILLICIT, LICIT, UNKNOWN)

MASTER_PATH = PROC_DIR / "master.parquet"
EDGES_PATH = PROC_DIR / "edges.parquet"


def load_raw(raw_dir=RAW_DIR):
    """Return (features, classes, edges) as stored on disk, with column names added."""
    names = ["txId", "time_step"] + LOCAL_COLS + AGG_COLS
    dtypes = {n: "float32" for n in names[2:]}
    dtypes.update({"txId": "int64", "time_step": "int16"})
    feats = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None,
                        names=names, dtype=dtypes)
    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    return feats, classes, edges


def build_master(feats, classes, edges):
    """Join features and labels; recode label (raw: '1' illicit, '2' licit, 'unknown').

    Returns master (one row per transaction, label in {1, 0, -1}) and the edges.
    Every join is verified with assertions.
    """
    assert feats["txId"].is_unique, "duplicate transaction ids in features"
    assert classes["txId"].is_unique, "duplicate transaction ids in classes"
    assert set(feats["txId"]) == set(classes["txId"]), "feature/label ids differ"

    mapping = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}
    classes = classes.assign(label=classes["class"].astype(str).map(mapping))
    assert classes["label"].notna().all(), "unexpected class value"

    master = feats.merge(classes[["txId", "label"]], on="txId", how="left",
                         validate="one_to_one")
    assert len(master) == len(feats)
    master["label"] = master["label"].astype("int8")

    ids = set(master["txId"])
    ok = edges["txId1"].isin(ids) & edges["txId2"].isin(ids)
    assert ok.all(), "edges reference unknown transactions"
    return master, edges


def load_master(rebuild=False):
    """Cached master table + edges (parquet). Built from the raw CSVs on first call."""
    if MASTER_PATH.exists() and EDGES_PATH.exists() and not rebuild:
        return pd.read_parquet(MASTER_PATH), pd.read_parquet(EDGES_PATH)
    feats, classes, edges = load_raw()
    master, edges = build_master(feats, classes, edges)
    master.to_parquet(MASTER_PATH, index=False)
    edges.to_parquet(EDGES_PATH, index=False)
    return master, edges


def labelled(master):
    """Rows usable for supervised learning/evaluation (unknown labels removed)."""
    return master[master["label"] != UNKNOWN].copy()
