from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true, probabilities, short_mask, threshold: float = 0.5, rows=None, cjk_proxy_mask=None) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    short_mask = np.asarray(short_mask, dtype=bool)
    if len(y_true) != len(probabilities) or len(y_true) != len(short_mask):
        raise ValueError("labels, probabilities, and short mask must align")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    pred = (probabilities >= threshold).astype(int)
    overall = float(f1_score(y_true, pred, labels=[0, 1], average="macro", zero_division=0)) if len(y_true) else None
    short_count = int(short_mask.sum())
    short = float(f1_score(y_true[short_mask], pred[short_mask], labels=[0, 1], average="macro", zero_division=0)) if short_count else None
    combined = (0.7 * overall + 0.3 * short) if overall is not None and short is not None else None
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel() if len(y_true) else (0, 0, 0, 0)
    result = {"labels": [0, 1], "threshold": threshold, "macro_f1": overall,
            "short_macro_f1": short, "short_count": short_count,
            "combined_score": combined, "combined_weights": {"macro_f1": 0.7, "short_macro_f1": 0.3},
            "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "human_false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
            "machine_recall": float(tp / (tp + fn)) if tp + fn else None,
            "precision": float(precision_score(y_true, pred, labels=[0, 1], average="binary", zero_division=0)) if len(y_true) else None,
            "recall": float(recall_score(y_true, pred, labels=[0, 1], average="binary", zero_division=0)) if len(y_true) else None,
            "accuracy": float(accuracy_score(y_true, pred)) if len(y_true) else None}
    if cjk_proxy_mask is not None:
        cjk_mask = np.asarray(cjk_proxy_mask, dtype=bool)
        count = int(cjk_mask.sum())
        result["cjk_proxy_count"] = count
        result["cjk_proxy_macro_f1"] = float(f1_score(y_true[cjk_mask], pred[cjk_mask], labels=[0, 1], average="macro", zero_division=0)) if count else None
    if rows is not None:
        def summarize(indexes):
            ids = np.asarray(indexes, dtype=int)
            if not len(ids): return {"count": 0, "macro_f1": None, "machine_recall": None, "human_false_positive_rate": None}
            yt, pr = y_true[ids], pred[ids]
            a, b, c, d = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
            return {"count": int(len(ids)), "macro_f1": float(f1_score(yt, pr, labels=[0, 1], average="macro", zero_division=0)),
                    "machine_recall": float(d / (d + c)) if d + c else None,
                    "human_false_positive_rate": float(b / (a + b)) if a + b else None}
        for field in ("source", "language"):
            values = sorted({str(r.get(field)) for r in rows if r.get(field) not in (None, "")})
            if values:
                result[f"by_{field}"] = {v: summarize([i for i, r in enumerate(rows) if str(r.get(field)) == v or (field == "source" and v in r.get("duplicate_sources", []))]) for v in values}
        result["by_length"] = {
            "short": summarize(np.flatnonzero(short_mask)),
            "long": summarize(np.flatnonzero(~short_mask)),
        }
        families = sorted({fam for r in rows if int(r.get("label", 0)) == 1 for fam in r.get("duplicate_generator_families", [str(r.get("generator_family", "unknown"))])})
        result["machine_recall_by_generator"] = {
            family: summarize([i for i, r in enumerate(rows) if int(r.get("label", 0)) == 1 and family in r.get("duplicate_generator_families", [str(r.get("generator_family", "unknown"))])])
            for family in families
        }
    return result

