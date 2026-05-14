"""Faithfulness metrics for token-level attribution.

Operationalizations follow DeYoung et al. (2020), ERASER: A Benchmark to
Evaluate Rationalized NLP Models, ACL 2020. The paper defines sufficiency
and comprehensiveness for rationales (subsets of input tokens marked as
"important"). For attribution methods, we form rationales by selecting
the top-k tokens by attribution score.

The "removal-as-prediction-change" metric is the average L1 distance
between the full-prompt response distribution and the single-token-removed
response distributions, taken over all tokens. The notebook documents
this choice and the rationale (single-token removals isolate per-token
effect without coalition interactions).

All functions accept callbacks that return a scalar prediction or a
probability vector. The spike runs in a text-only setting; the
"prediction" is typically the next-token log-probability of a chosen
target, or a one-hot indicator of the argmax token.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol

# Type aliases for clarity.
PredictionFn = Callable[[Sequence[int]], float]
"""Maps a coalition of kept token indices to a scalar prediction."""

DistributionFn = Callable[[Sequence[int]], Sequence[float]]
"""Maps a coalition of kept token indices to a probability vector."""


class AttributionScores(Protocol):
    """Protocol for indexable attribution score containers."""

    def __getitem__(self, index: int) -> float: ...

    def __len__(self) -> int: ...


def top_k_indices(
    scores: AttributionScores,
    k: int,
    *,
    rank_by_absolute: bool = True,
) -> list[int]:
    """Return indices of the top-k tokens by attribution score.

    Args:
        scores: Per-token attribution scores.
        k: Number of indices to return. Clamped to ``len(scores)``.
        rank_by_absolute: If True, rank by absolute value (default).
            If False, rank by signed value.

    Returns:
        A list of token indices, longest-first by score.
    """
    n = len(scores)
    if k <= 0:
        return []
    k = min(k, n)
    indexed: list[tuple[int, float]] = [(i, scores[i]) for i in range(n)]
    if rank_by_absolute:
        indexed.sort(key=lambda pair: abs(pair[1]), reverse=True)
    else:
        indexed.sort(key=lambda pair: pair[1], reverse=True)
    return [pair[0] for pair in indexed[:k]]


def sufficiency(
    prediction_fn: PredictionFn,
    scores: AttributionScores,
    k: int,
    *,
    rank_by_absolute: bool = True,
) -> float:
    """Compute sufficiency per DeYoung et al. (2020), Equation 1.

    Sufficiency is defined as ``p(y | x) - p(y | rationale)``: the drop in
    target prediction when the model sees only the top-k attributed
    tokens. A low (near-zero) sufficiency indicates the rationale is
    sufficient to recover the original prediction.

    Args:
        prediction_fn: Callback returning the target prediction (e.g.,
            log-prob of the original completion) given a coalition.
        scores: Per-token attribution scores over the full input.
        k: Number of top tokens to keep.
        rank_by_absolute: Rank by absolute attribution if True.

    Returns:
        Sufficiency value. Lower is better.

    References:
        DeYoung et al. 2020, Section 4.1, Equation 1.
    """
    n = len(scores)
    full_coalition = list(range(n))
    full_pred = prediction_fn(full_coalition)
    kept = top_k_indices(scores, k, rank_by_absolute=rank_by_absolute)
    rationale_pred = prediction_fn(kept)
    return full_pred - rationale_pred


def comprehensiveness(
    prediction_fn: PredictionFn,
    scores: AttributionScores,
    k: int,
    *,
    rank_by_absolute: bool = True,
) -> float:
    """Compute comprehensiveness per DeYoung et al. (2020), Equation 2.

    Comprehensiveness is defined as ``p(y | x) - p(y | x \\ rationale)``:
    the drop in target prediction when the top-k attributed tokens are
    removed. A high comprehensiveness indicates the rationale captures
    the tokens the model relies on.

    Args:
        prediction_fn: Callback returning the target prediction given a
            coalition of kept tokens.
        scores: Per-token attribution scores.
        k: Number of top tokens to remove.
        rank_by_absolute: Rank by absolute attribution if True.

    Returns:
        Comprehensiveness value. Higher is better.

    References:
        DeYoung et al. 2020, Section 4.1, Equation 2.
    """
    n = len(scores)
    full_coalition = list(range(n))
    full_pred = prediction_fn(full_coalition)
    to_remove = set(top_k_indices(scores, k, rank_by_absolute=rank_by_absolute))
    remaining = [i for i in range(n) if i not in to_remove]
    ablated_pred = prediction_fn(remaining)
    return full_pred - ablated_pred


def removal_as_prediction_change(
    distribution_fn: DistributionFn,
    n_tokens: int,
) -> float:
    """Average L1 shift over single-token removals.

    For each token index ``i``, compute the L1 distance between the full
    prompt's next-token distribution and the distribution produced when
    token ``i`` is removed. The metric is the mean over all tokens.

    This is the spike's operationalization of the brief's
    "removal-as-prediction-change" axis. It isolates per-token effects
    without confounding by coalition interactions. The notebook documents
    why we picked L1 over KL or argmax-flip.

    Args:
        distribution_fn: Callback returning a probability vector given a
            coalition of kept tokens.
        n_tokens: Number of tokens in the full input.

    Returns:
        Average L1 distance across single-token removals.

    References:
        DeYoung et al. 2020, Section 4 (removal-based faithfulness family).
    """
    full_dist = list(distribution_fn(list(range(n_tokens))))
    distances: list[float] = []
    for i in range(n_tokens):
        kept = [j for j in range(n_tokens) if j != i]
        ablated_dist = list(distribution_fn(kept))
        distances.append(_l1(full_dist, ablated_dist))
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def _l1(a: Sequence[float], b: Sequence[float]) -> float:
    """L1 distance between two equal-length vectors.

    The function pads the shorter vector with zeros if lengths differ;
    callers should already pass matched lengths.
    """
    n = max(len(a), len(b))
    total = 0.0
    for i in range(n):
        ai = a[i] if i < len(a) else 0.0
        bi = b[i] if i < len(b) else 0.0
        total += abs(ai - bi)
    return total


def bootstrap_ci(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap mean and percentile confidence interval.

    Used to summarize the 5+ repeats per benchmark cell. The notebook
    documents the bootstrap choice (over normal approximation) because
    sample sizes per cell are small and the underlying distributions are
    not assumed normal.

    Args:
        samples: Observed values.
        confidence: Confidence level in (0, 1). Default 0.95.
        n_resamples: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        A tuple ``(mean, lower, upper)`` where the bounds are the
        percentile-based confidence interval endpoints.
    """
    import random

    n = len(samples)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    if n == 1:
        only = float(samples[0])
        return (only, only, only)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lower_idx = int((1 - confidence) / 2 * n_resamples)
    upper_idx = int((1 + confidence) / 2 * n_resamples) - 1
    upper_idx = min(upper_idx, n_resamples - 1)
    mean_value = sum(samples) / n
    return (mean_value, means[lower_idx], means[upper_idx])
