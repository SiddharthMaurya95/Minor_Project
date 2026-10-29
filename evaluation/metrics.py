"""
Evaluation Metrics for Medical Image Classification.
Calculates:
- Threshold-0.5 Accuracy
- ROC-AUC (Binary and Macro OvR for Multiclass)
- Macro-Precision
- Macro-Recall (Sensitivity)
- Specificity (for binary tasks)
- Macro-F1
- Confusion Matrix
- Optional Logit Adjustment (Disabled by default)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def find_optimal_threshold(y_true, y_pred_prob, metric="f1", num_steps=100):
    """
    Finds the optimal decision threshold on validation data maximizing a specified metric (e.g. 'f1' or 'balanced_acc').
    y_true: 1D ground truth binary array (N,)
    y_pred_prob: 1D or (N, 2) predicted probabilities
    """
    y_true = np.asarray(y_true).squeeze()
    y_pred_prob = np.asarray(y_pred_prob)
    if y_pred_prob.ndim == 2:
        scores = y_pred_prob[:, 1]
    else:
        scores = y_pred_prob

    best_thresh = 0.5
    best_score = -1.0
    thresholds = np.linspace(0.05, 0.95, num_steps)

    for t in thresholds:
        m = compute_metrics(y_true, scores, threshold=t, is_multiclass=False)
        if metric == "accuracy":
            score = m['accuracy']
        elif metric == "balanced_acc":
            score = 0.5 * (m['recall'] + m.get('specificity', 0.0))
        elif metric == "youden":
            score = m['recall'] + m.get('specificity', 0.0) - 1.0
        elif metric == "precision":
            score = m['precision']
        elif metric == "recall":
            score = m['recall']
        else: # default f1
            score = m['f1']

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh, best_score


def compute_metrics(y_true, y_pred_prob, threshold=0.5, is_multiclass=False):
    """
    Computes standard medical classification metrics.
    y_true: 1D array of ground truth integer labels (N,)
    y_pred_prob: 1D or 2D array of predicted probabilities:
        - For binary: (N,) or (N, 2)
        - For multiclass: (N, C)
    """
    y_true = np.asarray(y_true).squeeze()
    y_pred_prob = np.asarray(y_pred_prob)

    if y_true.size == 0 or y_pred_prob.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    if len(y_true) != len(y_pred_prob):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} samples, y_pred_prob has {len(y_pred_prob)}.")

    if y_pred_prob.ndim > 2:
        raise ValueError(f"y_pred_prob cannot have more than 2 dimensions, got ndim={y_pred_prob.ndim}.")

    unique_true = np.unique(y_true)

    if not is_multiclass and (len(unique_true) <= 2 and (y_pred_prob.ndim == 1 or y_pred_prob.shape[1] == 2)):
        # Binary Classification
        if y_pred_prob.ndim == 2:
            scores_pos = y_pred_prob[:, 1]
        else:
            scores_pos = y_pred_prob

        y_pred = (scores_pos >= threshold).astype(int)

        acc = float(accuracy_score(y_true, y_pred))

        # ROC-AUC calculation (handle degenerate case where single class present)
        if len(unique_true) < 2:
            auc = 0.5
        else:
            try:
                auc = float(roc_auc_score(y_true, scores_pos))
            except ValueError:
                auc = 0.5

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Specificity & Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        return {
            'accuracy': acc,
            'roc_auc': auc,
            'precision': prec,
            'recall': rec,
            'sensitivity': rec,
            'specificity': specificity,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        }
    else:
        # Multiclass Classification
        if y_pred_prob.ndim != 2:
            raise ValueError("Multiclass probabilities must be 2D array of shape (N, C).")

        y_pred = np.argmax(y_pred_prob, axis=1)
        acc = float(accuracy_score(y_true, y_pred))

        try:
            auc = float(roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average='macro'))
        except (ValueError, IndexError):
            auc = 0.5

        prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

        num_classes = y_pred_prob.shape[1]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

        return {
            'accuracy': acc,
            'roc_auc': auc,
            'precision': prec,
            'recall': rec,
            'sensitivity': rec,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
        }


def apply_logit_adjustment(logits, class_priors, tau=1.0, eps=1e-8):
    """
    Applies post-hoc logit adjustment based on training class priors:
        logits_adj = logits - tau * log(pi_c + eps)
    WARNING: Only valid if class-prior shift is justified.
    Kept disabled by default in benchmark comparisons.
    """
    class_priors = np.asarray(class_priors, dtype=np.float32)
    priors_norm = class_priors / np.sum(class_priors)
    adjustment = tau * np.log(priors_norm + eps)
    return logits - adjustment
