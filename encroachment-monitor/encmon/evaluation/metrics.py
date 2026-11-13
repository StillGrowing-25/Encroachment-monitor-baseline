"""Evaluation metrics: precision, recall, F1, IoU, confusion matrix, CI."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> list[dict]:
    """Per-class precision / recall / F1 from a confusion matrix."""
    out = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out.append(dict(precision=float(p), recall=float(r), f1=float(f1)))
    return out


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """TP / FP / FN / TN for boolean masks."""
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


def prf_iou(counts: dict) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(precision=p, recall=r, f1=f1, iou=iou, **counts)


def mean_std_ci(values: list[float], conf: float = 0.95) -> dict:
    """Mean, sample std, and t-based confidence interval over resamples."""
    import scipy.stats as st

    v = np.asarray(values, float)
    n = len(v)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        t = st.t.ppf(0.5 + conf / 2, df=n - 1)
        half = t * std / np.sqrt(n)
    else:
        half = 0.0
    return dict(mean=mean, std=std, ci=(mean - half, mean + half), values=[float(x) for x in v])
