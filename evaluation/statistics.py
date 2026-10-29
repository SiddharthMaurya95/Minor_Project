"""
Statistical Analysis Utilities for Medical Image Classification Research.
Implements:
- Multi-seed aggregation (Mean, Standard Deviation, 95% Confidence Interval)
- McNemar's Test for paired classification discordance on identical test sets
- Paired statistical hypothesis tests
"""

import numpy as np
from scipy import stats


def compute_confidence_interval(values, confidence=0.95):
    """
    Computes standard error and confidence interval based on Student's t-distribution.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        return float(np.mean(values)), 0.0, (float(values[0]), float(values[0]))

    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, float(np.std(values, ddof=1)), (mean - h, mean + h)


def aggregate_seed_results(results_per_seed):
    """
    results_per_seed: list of dicts, each containing metrics from a run with a distinct seed.
    Returns: dict mapping metric_name -> {mean, std, ci95, all_values}
    """
    if not results_per_seed:
        return {}

    metric_keys = [k for k, v in results_per_seed[0].items() if isinstance(v, (int, float))]
    aggregated = {}

    for key in metric_keys:
        vals = [res[key] for res in results_per_seed if key in res]
        mean, std, ci = compute_confidence_interval(vals)
        aggregated[key] = {
            'mean': mean,
            'std': std,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'values': vals,
        }

    return aggregated


def mcnemar_test(y_true, y_pred_a, y_pred_b, continuity_correction=True):
    """
    Performs McNemar's test for paired classification discordance on the exact same test partition.
    y_true: ground truth labels (N,)
    y_pred_a: predictions from Model A (N,)
    y_pred_b: predictions from Model B (N,)

    Contingency matrix:
        a: both correct
        b: Model A correct, Model B incorrect
        c: Model A incorrect, Model B correct
        d: both incorrect
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b)) # A correct, B wrong
    c = int(np.sum(~correct_a & correct_b)) # A wrong, B correct
    a = int(np.sum(correct_a & correct_b))
    d = int(np.sum(~correct_a & ~correct_b))

    total_discordant = b + c

    if total_discordant == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'contingency': {'both_correct': a, 'a_correct_b_wrong': b, 'b_correct_a_wrong': c, 'both_wrong': d},
            'interpretation': "Identical error patterns between Model A and Model B.",
        }

    # If discordant pairs are small (<25), use exact two-sided binomial test
    if total_discordant < 25:
        p_val = float(stats.binomtest(min(b, c), total_discordant, 0.5, alternative='two-sided').pvalue)
        stat = float((b - c) ** 2 / total_discordant)
    else:
        # Edwards continuity correction
        if continuity_correction:
            stat = float((abs(b - c) - 1.0) ** 2 / total_discordant)
        else:
            stat = float((b - c) ** 2 / total_discordant)
        p_val = float(stats.chi2.sf(stat, df=1))

    return {
        'statistic': stat,
        'p_value': p_val,
        'contingency': {'both_correct': a, 'a_correct_b_wrong': b, 'b_correct_a_wrong': c, 'both_wrong': d},
        'significant_05': p_val < 0.05,
        'significant_01': p_val < 0.01,
        'interpretation': f"p-value = {p_val:.4e} ({'Statistically significant' if p_val < 0.05 else 'No statistically significant difference'}).",
    }
