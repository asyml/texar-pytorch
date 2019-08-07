import collections
import math
from typing import List, Counter, Tuple

from texar.torch.run.metric.base_metric import StreamingMetric
from texar.torch.utils.types import MaybeList

__all__ = [
    "BLEU",
]


def _maybe_str_to_list(list_or_str: MaybeList[str]) -> List[str]:
    if isinstance(list_or_str, str):
        return list_or_str.split()
    return list_or_str


def _get_ngrams(segment: MaybeList[str],
                max_order: int) -> Counter[Tuple[str, ...]]:
    r"""Extracts all n-grams up to a given maximum order from an
    input segment.

    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned
            by this methods.

    Returns:
        The Counter containing all n-grams upto :attr:`max_order`
        in segment with a count of how many times each n-gram occurred.
    """
    ngram_counts: Counter[Tuple[str, ...]] = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


class BLEU(StreamingMetric[MaybeList[str], float]):
    reference_length: int
    hypothesis_length: int
    matches_by_order: List[int]
    possible_matches_by_order: List[int]

    def __init__(self, max_order: int = 4, lowercase: bool = False,
                 smooth: bool = False, use_bp: bool = True, **kwargs):
        self.max_order = max_order
        self.lowercase = lowercase
        self.smooth = smooth
        self.use_bp = use_bp
        super().__init__(**kwargs)

    def reset(self) -> None:
        self.reference_length = 0
        self.hypothesis_length = 0
        self.matches_by_order = [0] * self.max_order
        self.possible_matches_by_order = [0] * self.max_order

    def add(self, predicted: List[MaybeList[str]],
            labels: List[MaybeList[str]]) -> None:
        for (reference, hypothesis) in zip(labels, predicted):
            reference = _maybe_str_to_list(reference)
            if self.lowercase:
                reference = [x.lower() for x in reference]
            reference_ngram_counts = _get_ngrams(reference, self.max_order)

            hypothesis = _maybe_str_to_list(hypothesis)
            if self.lowercase:
                hypothesis = [x.lower() for x in hypothesis]
            hypothesis_ngram_counts = _get_ngrams(hypothesis, self.max_order)

            self.reference_length += len(reference)
            self.hypothesis_length += len(hypothesis)

            overlap = hypothesis_ngram_counts & reference_ngram_counts
            for ngram in overlap:
                self.matches_by_order[len(ngram) - 1] += overlap[ngram]
            for order in range(self.max_order):
                possible_matches = len(hypothesis) - order
                self.possible_matches_by_order[order] += possible_matches

    def value(self) -> float:
        if self.reference_length == 0:
            return 0.0
        precisions = []
        for i in range(self.max_order):
            if self.smooth:
                precision = ((self.matches_by_order[i] + 1.0) /
                             (self.possible_matches_by_order[i] + 1.0))
            elif self.possible_matches_by_order[i] > 0:
                precision = (self.matches_by_order[i] /
                             self.possible_matches_by_order[i])
            else:
                precision = 0.0
            precisions.append(precision)

        geo_mean = 0.0
        if min(precisions) > 0:
            p_log_sum = sum(math.log(p) for p in precisions) / self.max_order
            geo_mean = math.exp(p_log_sum)

        bp = 1.0
        if self.use_bp:
            ratio = self.hypothesis_length / self.reference_length
            if ratio > 1.0:
                bp = 1.0
            elif abs(ratio) < 1e-8:
                bp = 0.0
            else:
                bp = math.exp(1 - 1.0 / ratio)

        bleu = geo_mean * bp
        return bleu * 100.0
