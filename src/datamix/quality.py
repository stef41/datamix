"""Data quality filters — deduplication, length filtering, scoring."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def length_filter(
    examples: List[str],
    min_length: int = 0,
    max_length: int = 0,
    min_words: int = 0,
    max_words: int = 0,
) -> Tuple[List[str], Dict[str, int]]:
    """Filter examples by character or word length.

    Returns (filtered_examples, stats_dict).
    """
    kept: List[str] = []
    removed = 0

    for text in examples:
        n_chars = len(text)
        n_words = len(text.split())

        if min_length > 0 and n_chars < min_length:
            removed += 1
            continue
        if max_length > 0 and n_chars > max_length:
            removed += 1
            continue
        if min_words > 0 and n_words < min_words:
            removed += 1
            continue
        if max_words > 0 and n_words > max_words:
            removed += 1
            continue
        kept.append(text)

    return kept, {"kept": len(kept), "removed": removed, "total": len(examples)}


def dedup_exact(examples: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """Remove exact duplicate texts.

    Returns (deduplicated_examples, stats_dict).
    """
    seen: set = set()
    kept: List[str] = []
    n_dupes = 0

    for text in examples:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in seen:
            n_dupes += 1
            continue
        seen.add(h)
        kept.append(text)

    return kept, {"kept": len(kept), "duplicates_removed": n_dupes, "total": len(examples)}


def dedup_ngram(
    examples: List[str],
    n: int = 5,
    threshold: float = 0.8,
) -> Tuple[List[str], Dict[str, int]]:
    """Remove near-duplicates using n-gram overlap (Jaccard similarity).

    Args:
        examples: Text examples.
        n: N-gram size.
        threshold: Jaccard similarity threshold for duplicate detection.
    """
    def _ngrams(text: str, n: int) -> set:
        words = text.lower().split()
        if len(words) < n:
            return {tuple(words)}
        return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

    kept: List[str] = []
    kept_ngrams: List[set] = []
    n_dupes = 0

    for text in examples:
        text_ng = _ngrams(text, n)
        is_dupe = False

        for existing_ng in kept_ngrams:
            if not text_ng or not existing_ng:
                continue
            intersection = len(text_ng & existing_ng)
            union = len(text_ng | existing_ng)
            if union > 0 and intersection / union >= threshold:
                is_dupe = True
                break

        if is_dupe:
            n_dupes += 1
        else:
            kept.append(text)
            kept_ngrams.append(text_ng)

    return kept, {"kept": len(kept), "near_duplicates_removed": n_dupes, "total": len(examples)}


def quality_score(
    text: str,
    min_length: int = 50,
    max_repetition_ratio: float = 0.3,
) -> float:
    """Compute a heuristic quality score for a text (0-1).

    Checks:
    - Length (penalize very short/very long)
    - Repetition (high word repetition = low quality)
    - Alphanumeric ratio (mostly non-text = low quality)
    - Sentence structure (has punctuation)
    """
    if not text:
        return 0.0

    score = 1.0

    # Length penalty
    n = len(text)
    if n < min_length:
        score *= n / min_length

    # Word repetition
    words = text.lower().split()
    if len(words) > 5:
        counts = Counter(words)
        most_common_frac = counts.most_common(1)[0][1] / len(words)
        if most_common_frac > max_repetition_ratio:
            score *= 1.0 - (most_common_frac - max_repetition_ratio)

    # Alphanumeric ratio
    alnum = sum(1 for c in text if c.isalnum())
    if n > 0:
        alnum_ratio = alnum / n
        if alnum_ratio < 0.3:
            score *= alnum_ratio / 0.3

    # Punctuation presence
    has_punct = any(c in text for c in ".!?;:")
    if not has_punct and len(words) > 10:
        score *= 0.8

    return max(0.0, min(1.0, round(score, 4)))
