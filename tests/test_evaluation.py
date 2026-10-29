"""
Automated Unit Tests for Evaluation Metrics and Statistics:
- Known example calculations.
- Perfect synthetic predictions (AUC=1.0, Acc=1.0).
- Inverted synthetic predictions (AUC=0.0, Acc=0.0).
- Random-score sanity check (AUC ~ 0.5 across trials).
- Missing classes handling.
- Invalid prediction shapes.
- McNemar's test sanity checks.
- Confidence interval calculations.
"""

import numpy as np
import pytest
from evaluation.metrics import compute_metrics, apply_logit_adjustment
from evaluation.statistics import compute_confidence_interval, mcnemar_test, aggregate_seed_results


def test_perfect_predictions():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred_prob = np.array([0.05, 0.95, 0.02, 0.88, 0.10, 0.99])
    res = compute_metrics(y_true, y_pred_prob)

    assert res['accuracy'] == 1.0
    assert res['roc_auc'] == 1.0
    assert res['precision'] == 1.0
    assert res['recall'] == 1.0
    assert res['f1'] == 1.0
    assert res['specificity'] == 1.0
    assert res['tp'] == 3
    assert res['tn'] == 3
    assert res['fp'] == 0
    assert res['fn'] == 0


def test_inverted_predictions():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred_prob = np.array([0.95, 0.05, 0.88, 0.02, 0.99, 0.10])
    res = compute_metrics(y_true, y_pred_prob)

    assert res['accuracy'] == 0.0
    assert res['roc_auc'] == 0.0
    assert res['f1'] == 0.0
    assert res['specificity'] == 0.0
    assert res['tp'] == 0
    assert res['tn'] == 0


def test_random_scores_auc_sanity():
    # Over 50 repeated trials of 200 samples, mean AUC should be close to 0.5
    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(50):
        y_true = rng.randint(0, 2, size=200)
        y_scores = rng.rand(200)
        res = compute_metrics(y_true, y_scores)
        aucs.append(res['roc_auc'])

    mean_auc = np.mean(aucs)
    assert 0.47 <= mean_auc <= 0.53, f"Expected random AUC ~ 0.5, got {mean_auc:.4f}"


def test_missing_classes():
    # Only class 1 present
    y_true = np.array([1, 1, 1, 1])
    y_pred_prob = np.array([0.8, 0.9, 0.7, 0.85])
    res = compute_metrics(y_true, y_pred_prob)
    assert res['accuracy'] == 1.0
    assert res['roc_auc'] == 0.5 # Handled degenerate AUC gracefully


def test_multiclass_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    # 2D probabilities: (6, 3)
    y_pred_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.1, 0.7],
        [0.9, 0.05, 0.05],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7],
    ])
    res = compute_metrics(y_true, y_pred_prob, is_multiclass=True)
    assert res['accuracy'] == 1.0
    assert res['roc_auc'] == 1.0
    assert res['f1'] == 1.0
    assert len(res['confusion_matrix']) == 3


def test_mcnemar_identical_models():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    pred_a = np.array([0, 1, 0, 1, 0, 1])
    pred_b = np.array([0, 1, 0, 1, 0, 1])
    res = mcnemar_test(y_true, pred_a, pred_b)
    assert res['statistic'] == 0.0
    assert res['p_value'] == 1.0


def test_mcnemar_discordant_models():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1] * 5)
    pred_a = y_true.copy() # 100% accurate
    pred_b = 1 - y_true # 0% accurate
    res = mcnemar_test(y_true, pred_a, pred_b)
    assert res['significant_05'] is True
    assert res['p_value'] < 0.001


def test_confidence_interval():
    data = [0.85, 0.86, 0.84, 0.85, 0.87]
    mean, std, (ci_low, ci_high) = compute_confidence_interval(data)
    assert 0.85 <= mean <= 0.86
    assert ci_low < mean < ci_high


def test_known_predictions():
    """Verify hand-calculated ground-truth numbers on a known benchmark test case."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_prob = np.array([0.2, 0.6, 0.4, 0.8])
    res = compute_metrics(y_true, y_pred_prob, threshold=0.5)

    assert res['tp'] == 1
    assert res['tn'] == 1
    assert res['fp'] == 1
    assert res['fn'] == 1
    assert res['accuracy'] == 0.5
    assert res['precision'] == 0.5
    assert res['recall'] == 0.5
    assert res['specificity'] == 0.5
    assert res['f1'] == 0.5
    assert res['roc_auc'] == 0.75


def test_invalid_shapes_raise_error():
    """Verify that invalid inputs, empty arrays, and mismatched shapes raise explicit ValueErrors."""
    # Empty array
    with pytest.raises(ValueError, match="Cannot compute metrics on empty arrays"):
        compute_metrics([], [])

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_metrics(np.array([0, 1]), np.array([0.5]))

    # 3D array
    with pytest.raises(ValueError, match="cannot have more than 2 dimensions"):
        compute_metrics(np.array([0, 1]), np.zeros((2, 2, 2)))

    # Multiclass 1D array
    with pytest.raises(ValueError, match="Multiclass probabilities must be 2D array"):
        compute_metrics(np.array([0, 1, 2]), np.array([0.2, 0.5, 0.3]), is_multiclass=True)


def test_optimal_threshold_tuning():
    """Verify optimal decision threshold selection on validation data."""
    from evaluation.metrics import find_optimal_threshold
    y_val = np.array([0, 0, 0, 1, 1, 1])
    # Predicted probabilities where standard 0.5 gives bad recall, but 0.3 gives perfect F1
    y_probs = np.array([0.1, 0.2, 0.25, 0.32, 0.35, 0.45])

    thresh, best_f1 = find_optimal_threshold(y_val, y_probs, metric="f1")
    assert 0.25 < thresh <= 0.32
    assert best_f1 == 1.0


def test_evaluate_checkpoint_e2e(tmp_path):
    """Verify evaluate_checkpoint end-to-end execution, JSON export, and CSV summary appending."""
    import os
    import torch
    from models.physsl_net import PhySSLNet
    from scripts.evaluate import evaluate_checkpoint

    model = PhySSLNet(in_channels=1, num_classes=2)
    ckpt_path = os.path.join(tmp_path, "dummy_model_seed42_best.pth")
    torch.save({
        'model_name': 'physsl_net',
        'seed': 42,
        'label_fraction': 1.0,
        'model_state_dict': model.state_dict(),
    }, ckpt_path)

    out_dir = os.path.join(tmp_path, "results")
    metrics, targets, probs = evaluate_checkpoint(
        checkpoint_path=ckpt_path,
        dataset_name="pneumoniamnist",
        data_dir="./data",
        split="test",
        batch_size=128,
        threshold=0.5,
        use_tta=True,
        device="cpu",
        output_dir=out_dir,
    )

    # Verify return values
    assert len(targets) == 624
    assert probs.shape == (624, 2)
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['roc_auc'] <= 1.0
    assert metrics['tta_applied'] is True
    assert metrics['num_parameters'] == 46294
    assert metrics['latency_ms_per_sample'] > 0

    # Verify JSON export
    json_path = os.path.join(out_dir, "dummy_model_seed42_best_test_eval.json")
    assert os.path.exists(json_path)

    # Verify CSV export
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    assert os.path.exists(csv_path)

