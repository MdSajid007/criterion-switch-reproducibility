"""Standalone tie-neutral fixed-fraction retrieval evaluation.

The selected mass is K=max(1, floor(q*N)); if the boundary falls within a tied
score class, that class is included fractionally so row ordering cannot change
the metric.
"""
from __future__ import annotations

import numpy as np

MASS_TOL = 1e-9
EQ_TOL = 1e-12

class TieNeutralView:
    """Score-level view of one (cell, family) vector, ready for any replicate.

    Rows are ordered by score DESC.  Within a tied score block the row order is
    arbitrary and provably immaterial: every quantity the metric uses is read at
    a block BOUNDARY, i.e. from a sum over the whole block.
    """

    __slots__ = ("y_sorted", "cluster_sorted", "level_end", "n", "n_clusters",
                 "n_levels", "levels", "order")

    def __init__(self, y, score, cluster_idx, n_clusters, order=None):
        s = np.asarray(score, dtype=np.float64)
        if order is None:
            # score DESC; ties in whatever order the stable sort yields.  No
            # canonical_row_key is consulted: it carries no ranking authority.
            order = np.argsort(-s, kind="stable")
        self.order = np.asarray(order, dtype=np.int64)
        ss = s[self.order]
        self.y_sorted = np.asarray(y, dtype=np.float64)[self.order]
        self.cluster_sorted = np.asarray(cluster_idx, dtype=np.int64)[self.order]
        self.n = int(len(ss))
        self.n_clusters = int(n_clusters)
        d = np.flatnonzero(np.diff(ss) != 0.0)
        self.level_end = np.concatenate((d, np.array([self.n - 1], dtype=np.int64)))
        self.levels = ss[self.level_end]
        self.n_levels = int(len(self.level_end))

    # Core fixed-fraction boundary rule
    def _fixed_fraction(self, v: np.ndarray, k_fn) -> dict:
        cw = np.cumsum(v)
        cwy = np.cumsum(v * self.y_sorted)
        n_b = float(cw[-1])
        pos_b = float(cwy[-1])
        k_b = k_fn(n_b)

        CW = cw[self.level_end]
        CP = cwy[self.level_end]
        j = int(np.searchsorted(CW, k_b, side="left"))
        j = min(j, self.n_levels - 1)

        w_above = float(CW[j - 1]) if j > 0 else 0.0
        tp_above = float(CP[j - 1]) if j > 0 else 0.0
        w_tie = float(CW[j]) - w_above
        p_tie = float(CP[j]) - tp_above
        q = float(k_b) - w_above
        alpha = (q / w_tie) if w_tie > 0.0 else 0.0
        tp_eff = tp_above + alpha * p_tie

        prevalence = (pos_b / n_b) if n_b > 0 else float("nan")
        recall = (tp_eff / pos_b) if pos_b > 0 else float("nan")
        precision = (tp_eff / k_b) if k_b > 0 else float("nan")
        lift = (precision / prevalence) if (prevalence and prevalence > 0) else float("nan")
        return {
            "n": n_b, "n_positive": pos_b, "prevalence": prevalence,
            "k": k_b, "cutoff_level_index": j, "cutoff_score": float(self.levels[j]),
            "n_above": w_above, "n_tie": w_tie, "q": q, "alpha": alpha,
            "tie_block_positives": p_tie,
            "effective_tp_at_K": tp_eff,
            "inclusion_mass": w_above + alpha * w_tie,
            "recall_at_5pct": recall,
            "precision_at_5pct": precision,
            "alert_lift_at_5pct": lift,
        }

    def full_sample(self, k_fn) -> dict:
        return self._fixed_fraction(np.ones(self.n, dtype=np.float64), k_fn)

    def replicate(self, mult: np.ndarray, k_fn) -> dict:
        return self._fixed_fraction(
            np.asarray(mult, dtype=np.float64)[self.cluster_sorted], k_fn)

    def replicate_arrays(self, mult_matrix: np.ndarray, k_fn) -> dict[str, np.ndarray]:
        b = mult_matrix.shape[0]
        out = {m: np.full(b, np.nan) for m in
               ("recall_at_5pct", "precision_at_5pct", "alert_lift_at_5pct")}
        mass_ok = np.ones(b, dtype=bool)
        for i in range(b):
            r = self.replicate(mult_matrix[i], k_fn)
            for m in out:
                out[m][i] = r[m]
            mass_ok[i] = abs(r["inclusion_mass"] - r["k"]) <= MASS_TOL
        out["_mass_ok"] = mass_ok
        return out

    # Tie-attainability diagnostic bounds
    def tie_bounds(self, k_fn) -> tuple[float, float]:
        """Attainable min / max Recall@5% under arbitrary within-block ordering."""
        v = np.ones(self.n, dtype=np.float64)
        cw = np.cumsum(v)
        k = k_fn(float(cw[-1]))
        pos = float(self.y_sorted.sum())
        if pos <= 0:
            return float("nan"), float("nan")
        CW = cw[self.level_end]
        cwy = np.cumsum(self.y_sorted)
        CP = cwy[self.level_end]
        j = int(np.searchsorted(CW, k, side="left"))
        j = min(j, self.n_levels - 1)
        n_above = float(CW[j - 1]) if j > 0 else 0.0
        tp_above = float(CP[j - 1]) if j > 0 else 0.0
        n_tie = float(CW[j]) - n_above
        p_tie = float(CP[j]) - tp_above
        neg_tie = n_tie - p_tie
        q = float(k) - n_above
        lo = tp_above + max(0.0, q - neg_tie)
        hi = tp_above + min(q, p_tie)
        return lo / pos, hi / pos
