"""Naive Monte Carlo Shapley sampling.

This implementation follows Strumbelj & Kononenko (2014), "Explaining
prediction models and individual predictions with feature contributions"
(Knowledge and Information Systems 41(3), 647-665). Castro et al. (2009),
"Polynomial calculation of the Shapley value based on sampling"
(Computers & Operations Research 36(5), 1726-1730), is the prior canonical
reference for permutation sampling Shapley.

The estimator draws permutations uniformly from the symmetric group over
players, then accumulates marginal contributions to each player as it is
inserted at its position in the permutation. The mean over samples is an
unbiased estimate of the Shapley value.

The implementation is a baseline for the E2 spike comparison; it is not
optimized. Production-quality versions cache coalition values, use
antithetic pairs, or apply control variates.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

ValueFn = Callable[[frozenset[int]], float]
"""Maps a coalition (kept token indices) to a scalar value."""


def shapley_monte_carlo(
    players: Sequence[int],
    value_fn: ValueFn,
    *,
    n_samples: int,
    seed: int = 0,
) -> list[float]:
    """Estimate Shapley values by permutation sampling.

    For each sampled permutation ``pi``, walk the permutation left to
    right. The marginal contribution of player ``pi[k]`` is the value of
    the coalition ``{pi[0], ..., pi[k]}`` minus the value of the coalition
    ``{pi[0], ..., pi[k-1]}``. Average over samples.

    Args:
        players: Player indices (e.g., token positions).
        value_fn: Callback returning the value of a coalition. Must accept
            an empty frozenset.
        n_samples: Number of permutation samples to draw.
        seed: RNG seed.

    Returns:
        A list of estimated Shapley values, one per player, in the same
        order as ``players``.

    References:
        Strumbelj & Kononenko 2014, Section 3.1, Algorithm 1.
        Castro et al. 2009, Section 3 (canonical permutation-sampling
        formulation).

    Raises:
        ValueError: If ``n_samples`` is non-positive.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = random.Random(seed)
    n = len(players)
    if n == 0:
        return []

    index_of: dict[int, int] = {player: idx for idx, player in enumerate(players)}
    contributions: list[float] = [0.0 for _ in range(n)]

    for _ in range(n_samples):
        permutation = list(players)
        rng.shuffle(permutation)
        prefix: set[int] = set()
        prev_value = value_fn(frozenset(prefix))
        for player in permutation:
            prefix.add(player)
            new_value = value_fn(frozenset(prefix))
            marginal = new_value - prev_value
            contributions[index_of[player]] += marginal
            prev_value = new_value

    return [c / n_samples for c in contributions]


def antithetic_shapley_monte_carlo(
    players: Sequence[int],
    value_fn: ValueFn,
    *,
    n_samples: int,
    seed: int = 0,
) -> list[float]:
    """Variance-reduced Shapley estimator using antithetic permutations.

    For each base permutation, also evaluate its reverse. The reverse
    permutation is negatively correlated with the original for monotone
    value functions, which reduces estimator variance. Reference: Mitchell
    et al. (2022), "Sampling Permutations for Shapley Value Estimation",
    JMLR 23(43), 1-46.

    This is provided for follow-up work, not the headline E2 comparison.
    The notebook compares the basic estimator against TokenSHAP and
    KernelSHAP, so this helper exists for the recommendation's risk
    section ("if naive MC underperforms, antithetic sampling is the
    obvious next step").

    Args:
        players: Player indices.
        value_fn: Coalition value callback.
        n_samples: Number of base permutations; total evaluations are
            ``2 * n_samples``.
        seed: RNG seed.

    Returns:
        A list of estimated Shapley values, one per player.

    Raises:
        ValueError: If ``n_samples`` is non-positive.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = random.Random(seed)
    n = len(players)
    if n == 0:
        return []

    index_of: dict[int, int] = {player: idx for idx, player in enumerate(players)}
    contributions: list[float] = [0.0 for _ in range(n)]

    for _ in range(n_samples):
        permutation = list(players)
        rng.shuffle(permutation)
        for direction in (permutation, list(reversed(permutation))):
            prefix: set[int] = set()
            prev_value = value_fn(frozenset(prefix))
            for player in direction:
                prefix.add(player)
                new_value = value_fn(frozenset(prefix))
                marginal = new_value - prev_value
                contributions[index_of[player]] += marginal
                prev_value = new_value

    denom = 2 * n_samples
    return [c / denom for c in contributions]
