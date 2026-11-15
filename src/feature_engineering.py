"""Graph and time features that use ONLY same-time-step information (no future leakage).

In Elliptic every edge joins two transactions in the same time step, so degrees and
neighbour means computed within a step cannot contain information from later steps.
Neighbour LABELS are deliberately never used (they would leak the target).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import ENG_COLS_BASE


def _adjacency(master, edges):
    idx = pd.Series(np.arange(len(master)), index=master["txId"].values)
    src = idx.loc[edges["txId1"].values].values
    dst = idx.loc[edges["txId2"].values].values
    n = len(master)
    # A[i, j] = 1 if money flows i -> j
    return sp.csr_matrix((np.ones(len(src), dtype="float32"), (src, dst)), shape=(n, n))


def add_graph_features(master, edges, neighbour_cols=()):
    """Return a DataFrame (aligned to master.index) with engineered columns:
    in_degree, out_degree, degree, step_tx_count and, for each column in
    `neighbour_cols`, the mean of that column over in- and out-neighbours.
    """
    A = _adjacency(master, edges)
    in_deg = np.asarray(A.sum(axis=0)).ravel()
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    out = pd.DataFrame({
        "in_degree": in_deg, "out_degree": out_deg, "degree": in_deg + out_deg,
        "step_tx_count": master.groupby("time_step")["txId"].transform("size").values,
    }, index=master.index).astype("float32")

    if len(neighbour_cols):
        X = master[list(neighbour_cols)].to_numpy(dtype="float32")
        in_mean = (A.T @ X) / np.maximum(in_deg, 1)[:, None]    # predecessors
        out_mean = (A @ X) / np.maximum(out_deg, 1)[:, None]    # successors
        extra = {}
        for j, c in enumerate(neighbour_cols):
            extra[f"{c}_in_mean"] = in_mean[:, j].astype("float32")
            extra[f"{c}_out_mean"] = out_mean[:, j].astype("float32")
        out = pd.concat([out, pd.DataFrame(extra, index=master.index)], axis=1)
    return out


def engineered_columns(neighbour_cols):
    cols = list(ENG_COLS_BASE)
    for c in neighbour_cols:
        cols += [f"{c}_in_mean", f"{c}_out_mean"]
    return cols
