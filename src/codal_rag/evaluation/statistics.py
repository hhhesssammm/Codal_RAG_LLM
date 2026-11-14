"""Paired company-cluster bootstrap. Never resample individual questions."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np


def paired_cluster_bootstrap(left, right, metric, *, resamples=10000, seed=42):
    if type(resamples) is not int or resamples < 1:
        raise ValueError('Bootstrap resamples must be a positive integer')
    if len({r['id'] for r in left}) != len(left) or len({r['id'] for r in right}) != len(right):
        raise ValueError('Duplicate question IDs in paired comparison')
    a, b = {r['id']: r for r in left}, {r['id']: r for r in right}
    if set(a) != set(b):
        raise ValueError('Paired comparisons require identical questions')
    groups = defaultdict(list)
    for key in sorted(a):
        if a[key]['company_id'] != b[key]['company_id'] or a[key]['type'] != b[key]['type']:
            raise ValueError('Paired comparison company/type mismatch')
        x, y = a[key]['metrics'].get(metric), b[key]['metrics'].get(metric)
        if x is None or y is None:
            continue
        if type(x) not in (float, int) or type(y) not in (float, int) or not math.isfinite(x+y):
            raise ValueError('Bootstrap metric must be finite numeric or null')
        groups[a[key]['company_id']].append(y-x)
    total = sum(map(len, groups.values()))
    result = {'metric': metric, 'effect': 'right_minus_left_question_weighted_mean',
              'n_questions': total, 'n_companies': len(groups), 'resamples': resamples,
              'seed': seed, 'mean_difference': None, 'ci95': None}
    if not groups:
        return result
    sums = np.array([sum(v) for v in groups.values()], dtype=float)
    sizes = np.array([len(v) for v in groups.values()], dtype=float)
    result['mean_difference'] = float(sums.sum()/sizes.sum())
    if len(groups) < 2:
        result['limitation'] = 'At least two company clusters required for a meaningful interval'
        return result
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        indexes = rng.integers(0, len(groups), size=len(groups))
        values.append(sums[indexes].sum()/sizes[indexes].sum())
    result['ci95'] = [float(v) for v in np.quantile(values, [0.025, 0.975])]
    return result
