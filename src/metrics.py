from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def multiclass_metrics(y_true, y_pred, probabilities, class_names):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )

    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    try:
        auc = roc_auc_score(y_true_bin, probabilities, average="macro", multi_class="ovr")
    except ValueError:
        # This can happen for tiny or hand-filtered evaluation samples.
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "auc_roc_ovr_macro": float(auc),
        "per_class": {
            class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i in range(len(class_names))
        },
    }, confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
