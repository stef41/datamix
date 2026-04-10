"""Dataset diversity metrics."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DiversityMetrics:
    """Computed diversity metrics for a corpus."""

    vocabulary_richness: float
    length_variance: float
    topic_entropy: float
    unique_ngram_ratio: float
    redundancy_score: float


class DiversityAnalyzer:
    """Analyze diversity of a text dataset."""

    def __init__(self, ngram_size: int = 3) -> None:
        self._ngram_size = ngram_size

    # ── public API ───────────────────────────────────────────────────────

    def analyze(self, texts: List[str]) -> DiversityMetrics:
        """Compute all diversity metrics at once."""
        return DiversityMetrics(
            vocabulary_richness=self.vocabulary_richness(texts),
            length_variance=self.length_distribution(texts)["std"],
            topic_entropy=self.topic_entropy(texts),
            unique_ngram_ratio=self.ngram_diversity(texts),
            redundancy_score=self.redundancy_score(texts),
        )

    def vocabulary_richness(self, texts: List[str]) -> float:
        """Type-token ratio: unique words / total words."""
        tokens = self._tokenize_all(texts)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def length_distribution(self, texts: List[str]) -> Dict[str, float]:
        """Statistics on text lengths (in characters)."""
        if not texts:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        lengths = [len(t) for t in texts]
        n = len(lengths)
        mean = sum(lengths) / n
        variance = sum((x - mean) ** 2 for x in lengths) / n
        std = math.sqrt(variance)
        sorted_lengths = sorted(lengths)
        if n % 2 == 1:
            median = float(sorted_lengths[n // 2])
        else:
            median = (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2.0
        return {
            "mean": mean,
            "std": std,
            "min": float(min(lengths)),
            "max": float(max(lengths)),
            "median": median,
        }

    def ngram_diversity(self, texts: List[str], n: Optional[int] = None) -> float:
        """Ratio of unique n-grams to total n-grams across all texts."""
        if n is None:
            n = self._ngram_size
        all_ngrams: List[tuple] = []
        for text in texts:
            tokens = self._tokenize(text)
            for i in range(len(tokens) - n + 1):
                all_ngrams.append(tuple(tokens[i : i + n]))
        if not all_ngrams:
            return 0.0
        return len(set(all_ngrams)) / len(all_ngrams)

    def topic_entropy(self, texts: List[str]) -> float:
        """Shannon entropy of keyword distribution.

        Uses word frequency as a proxy for topic distribution.
        Filters to words appearing at least twice to approximate keywords.
        """
        tokens = self._tokenize_all(texts)
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def redundancy_score(self, texts: List[str]) -> float:
        """Score indicating content overlap (0=none, 1=fully redundant).

        Based on the proportion of shared n-grams between pairs,
        estimated efficiently via the complement of n-gram diversity.
        """
        if len(texts) < 2:
            return 0.0
        diversity = self.ngram_diversity(texts)
        return max(0.0, 1.0 - diversity)

    def compare_diversity(
        self, texts_a: List[str], texts_b: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compare diversity metrics between two text sets."""
        metrics_a = self.analyze(texts_a)
        metrics_b = self.analyze(texts_b)
        return {
            "a": {
                "vocabulary_richness": metrics_a.vocabulary_richness,
                "length_variance": metrics_a.length_variance,
                "topic_entropy": metrics_a.topic_entropy,
                "unique_ngram_ratio": metrics_a.unique_ngram_ratio,
                "redundancy_score": metrics_a.redundancy_score,
            },
            "b": {
                "vocabulary_richness": metrics_b.vocabulary_richness,
                "length_variance": metrics_b.length_variance,
                "topic_entropy": metrics_b.topic_entropy,
                "unique_ngram_ratio": metrics_b.unique_ngram_ratio,
                "redundancy_score": metrics_b.redundancy_score,
            },
            "delta": {
                "vocabulary_richness": metrics_a.vocabulary_richness - metrics_b.vocabulary_richness,
                "length_variance": metrics_a.length_variance - metrics_b.length_variance,
                "topic_entropy": metrics_a.topic_entropy - metrics_b.topic_entropy,
                "unique_ngram_ratio": metrics_a.unique_ngram_ratio - metrics_b.unique_ngram_ratio,
                "redundancy_score": metrics_a.redundancy_score - metrics_b.redundancy_score,
            },
        }

    # ── internal helpers ─────────────────────────────────────────────────

    _WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?")

    def _tokenize(self, text: str) -> List[str]:
        return [m.lower() for m in self._WORD_RE.findall(text)]

    def _tokenize_all(self, texts: List[str]) -> List[str]:
        tokens: List[str] = []
        for text in texts:
            tokens.extend(self._tokenize(text))
        return tokens


def format_diversity_report(metrics: DiversityMetrics) -> str:
    """Render a DiversityMetrics as a human-readable report."""
    lines = [
        "=== Diversity Report ===",
        f"  Vocabulary richness : {metrics.vocabulary_richness:.4f}",
        f"  Length variance      : {metrics.length_variance:.2f}",
        f"  Topic entropy        : {metrics.topic_entropy:.4f} bits",
        f"  Unique n-gram ratio  : {metrics.unique_ngram_ratio:.4f}",
        f"  Redundancy score     : {metrics.redundancy_score:.4f}",
    ]
    return "\n".join(lines)
