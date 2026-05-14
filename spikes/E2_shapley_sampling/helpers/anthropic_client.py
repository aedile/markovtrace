"""Thin Anthropic client wrapper with disk-backed response caching.

This module exists so the E2 spike notebook can iterate without re-paying
for identical API calls. Keys are hashed over the prompt, system prompt,
temperature, and model. Responses persist under ``.cache/`` next to the
notebook.

The wrapper is intentionally minimal. It does not implement retries,
streaming, or rate-limit handling; the spike runs at low concurrency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Anthropic Haiku 4-5 pricing per the 2026-05-14 snapshot used in the
# notebook spend tracker. Values in USD per 1M tokens. Update the date
# in the notebook cost cell when these change.
PRICE_INPUT_PER_M = 1.00
PRICE_OUTPUT_PER_M = 5.00
DEFAULT_MODEL = "claude-haiku-4-5"


@dataclass(frozen=True)
class Usage:
    """Token usage counters returned by the Anthropic API."""

    input_tokens: int
    output_tokens: int

    def cost_usd(
        self,
        input_price_per_m: float = PRICE_INPUT_PER_M,
        output_price_per_m: float = PRICE_OUTPUT_PER_M,
    ) -> float:
        """Return the cost of this call in USD.

        Args:
            input_price_per_m: USD per million input tokens.
            output_price_per_m: USD per million output tokens.

        Returns:
            Cost in USD as a float.
        """
        return (
            self.input_tokens * input_price_per_m / 1_000_000
            + self.output_tokens * output_price_per_m / 1_000_000
        )


@dataclass(frozen=True)
class CompletionResult:
    """Container for a single Anthropic completion."""

    text: str
    usage: Usage
    model: str
    cached: bool


def _cache_key(
    prompt: str,
    system: str,
    temperature: float,
    model: str,
    max_tokens: int,
) -> str:
    """Hash the parameters that determine a deterministic completion."""
    payload = json.dumps(
        {
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
            "model": model,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cache(cache_dir: Path, key: str) -> CompletionResult | None:
    """Return a cached completion if present, otherwise ``None``."""
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("cache read failed for %s: %s", path, exc)
        return None
    usage = Usage(
        input_tokens=int(data["usage"]["input_tokens"]),
        output_tokens=int(data["usage"]["output_tokens"]),
    )
    return CompletionResult(
        text=str(data["text"]),
        usage=usage,
        model=str(data["model"]),
        cached=True,
    )


def _save_cache(cache_dir: Path, key: str, result: CompletionResult) -> None:
    """Persist a completion to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    payload: dict[str, Any] = {
        "text": result.text,
        "model": result.model,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def complete(
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.0,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 256,
    cache_dir: Path | str = ".cache",
    client: Any | None = None,  # noqa: ANN401  # untyped anthropic.Anthropic
) -> CompletionResult:
    """Return a completion, hitting the disk cache before the network.

    Args:
        prompt: The user prompt.
        system: Optional system prompt; default is empty.
        temperature: Sampling temperature; default 0.0.
        model: Anthropic model id; default ``claude-haiku-4-5``.
        max_tokens: Maximum completion tokens.
        cache_dir: Directory for the on-disk cache.
        client: An optional pre-constructed ``anthropic.Anthropic`` client.
            If ``None``, a new client is constructed from the environment
            variable ``ANTHROPIC_API_KEY``.

    Returns:
        A :class:`CompletionResult`. The ``cached`` flag indicates whether
        the response came from disk.

    Raises:
        RuntimeError: If ``ANTHROPIC_API_KEY`` is unset and ``client`` is
            ``None``.
    """
    cache_dir_path = Path(cache_dir)
    key = _cache_key(prompt, system, temperature, model, max_tokens)
    cached = _load_cache(cache_dir_path, key)
    if cached is not None:
        return cached

    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set and no client was provided. "
                "Set the environment variable before calling complete()."
            )
        # Imported lazily so the spike module can be imported without the
        # anthropic package installed (e.g. during static checks).
        import anthropic  # type: ignore[import-not-found]

        client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system if system else None,
        messages=[{"role": "user", "content": prompt}],
    )

    # Anthropic returns a list of content blocks; concatenate text blocks.
    text_parts: list[str] = []
    for block in response.content:
        block_text = getattr(block, "text", None)
        if isinstance(block_text, str):
            text_parts.append(block_text)
    text = "".join(text_parts)

    usage = Usage(
        input_tokens=int(response.usage.input_tokens),
        output_tokens=int(response.usage.output_tokens),
    )
    result = CompletionResult(text=text, usage=usage, model=model, cached=False)
    _save_cache(cache_dir_path, key, result)
    return result


def estimate_token_count(text: str) -> int:
    """Rough character-based token estimate used for size bucketing.

    The spike does not depend on an exact tokenizer for sizing; the prompt
    sets are handcrafted to land in 1k, 4k, and 16k token brackets. This
    helper is a fast sanity check, not an authoritative count.

    Args:
        text: The text to estimate.

    Returns:
        An integer estimate based on the four-characters-per-token heuristic
        documented by Anthropic in the public pricing materials.
    """
    return max(1, len(text) // 4)
