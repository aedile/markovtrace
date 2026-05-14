"""KernelSHAP adapter for token-level attribution.

Wraps the ``shap`` library's ``KernelExplainer`` to compute Shapley value
estimates over token-level binary features. Each token is a feature; the
value 1 means the token is present in the input, 0 means it is replaced
by a mask placeholder.

Caveat: KernelSHAP was designed for tabular data where features have
independent meaning. Applying it to tokens has two known pitfalls.

1. Tokens are not independent. Removing a token from the middle of a
   sentence may yield a malformed prompt. The spike uses position-aware
   masking: the token is replaced by a single space, preserving overall
   sequence length and word boundaries where possible.
2. The weighted regression KernelSHAP performs assumes the coalition
   sample distribution matches the SHAP kernel weights. The library
   handles this internally; we surface its sample count knob as
   ``nsamples`` for direct comparison against the naive Monte Carlo
   budget.

The spike documents these caveats in the notebook and in
``RECOMMENDATION.md``.

References:
    Lundberg & Lee 2017, "A Unified Approach to Interpreting Model
    Predictions", NeurIPS 2017. Algorithm 1 (KernelSHAP).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# numpy and shap are runtime-only dependencies for the spike. They are
# imported lazily inside the function so the helper module imports
# cleanly under mypy --strict without the packages installed. PM
# execution installs both before running the notebook.


def kernel_shap_token_attribution(
    tokens: Sequence[str],
    prediction_fn: Callable[[Sequence[str]], float],
    *,
    nsamples: int,
    mask_token: str = " ",
    seed: int = 0,
) -> list[float]:
    """Compute KernelSHAP attribution over token-level binary features.

    Args:
        tokens: The token sequence to attribute over.
        prediction_fn: Callback that takes the reconstructed token list
            (with mask substitutions applied) and returns a scalar
            prediction (e.g., log-prob of the original completion).
        nsamples: Number of coalition samples for KernelSHAP. Equivalent
            to the budget knob in the other methods.
        mask_token: String inserted in place of removed tokens. Default
            is a single space; the notebook documents the alternatives.
        seed: RNG seed.

    Returns:
        A list of per-token attribution scores.

    Raises:
        ValueError: If the token sequence is empty.

    References:
        Lundberg & Lee 2017, NeurIPS 2017.
    """
    if not tokens:
        raise ValueError("tokens must be non-empty")

    # Lazy imports: numpy and shap may not be installed in the static-check
    # environment. They are required at execution time and PM execution
    # installs them before running the notebook.
    import numpy as _np  # type: ignore[import-not-found]
    import shap as _shap  # type: ignore[import-not-found]

    n = len(tokens)

    def model_fn(mask_matrix: Any) -> Any:  # noqa: ANN401  # numpy ndarray surface
        """Score every coalition row of the mask matrix.

        Each row is a binary vector of length ``n``. A 1 means keep the
        original token; a 0 means substitute the mask token. The function
        returns a 1-D float array of predictions.
        """
        rows = mask_matrix.shape[0]
        out = _np.zeros(rows, dtype=_np.float64)
        for i in range(rows):
            mask = mask_matrix[i]
            reconstructed = [tokens[j] if mask[j] >= 0.5 else mask_token for j in range(n)]
            out[i] = prediction_fn(reconstructed)
        return out

    # Background row: all-zeros (every token masked). This is the
    # canonical reference state for KernelSHAP.
    background = _np.zeros((1, n), dtype=_np.float64)
    explainer = _shap.KernelExplainer(model_fn, background, seed=seed)
    foreground = _np.ones((1, n), dtype=_np.float64)
    raw = explainer.shap_values(foreground, nsamples=nsamples, silent=True)

    # shap returns either ndarray or list-of-ndarray depending on output
    # shape; coerce to a 1-D Python list.
    if isinstance(raw, list):
        array = _np.asarray(raw[0], dtype=_np.float64)
    else:
        array = _np.asarray(raw, dtype=_np.float64)
    flat = array.flatten().tolist()
    return [float(x) for x in flat[:n]]
