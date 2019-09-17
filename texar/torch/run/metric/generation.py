# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Executor metrics for generation tasks.
"""

import collections
import math
from typing import Counter, List, Sequence, Tuple

from texar.torch.run.metric.base_metric import StreamingMetric
from texar.torch.utils.types import MaybeList

__all__ = [
    "BLEU",
]


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
    r"""The BLEU metric for evaluating translation tasks. BLEU stands for
    bilingual evaluation understudy, and measure the percentage of overlapping
    n-grams between the translation (hypothesis) and references.

    This metric also
    supports Smooth BLEU, computed following the method outlined in the paper:

        (Lin et al. 2004) ORANGE: a method for evaluating automatic evaluation
        metrics for machine translation.
        Chin-Yew Lin, Franz Josef Och. COLING 2004.

    BLEU is a :class:`~texar.torch.run.metric.StreamingMetric`, requires both
    predicted values and labels. BLEU values are :class:`float` numbers between
    0 and 100, with higher values being better.

    Args:
        max_order (int): Maximum n-gram order to use when computing BLEU score.
            Defaults to 4.
        lowercase (bool): Whether to lowercase all text before computing. If
            enabled, the metric is also known as "uncased BLEU". Defaults to
            `False`.
        smooth (bool): Whether or not to apply `(Lin et al. 2004)` smoothing.
            Defaults to `False`.
        brevity_penalty (bool): Whether to apply brevity penalty. Defaults to
            `True`.

    Keyword Args:
        pred_name (str): Name of the predicted value. This will be used as the
            key to the dictionary returned by the model.
        label_name (str): Name of the label. This will be used as the key to the
            batch object returned by the dataset.
    """
    reference_length: int
    hypothesis_length: int
    matches_by_order: List[int]
    possible_matches_by_order: List[int]

    def __init__(self, max_order: int = 4, lowercase: bool = False,
                 smooth: bool = False, brevity_penalty: bool = True,
                 *, pred_name: str, label_name: str):
        self.max_order = max_order
        self.lowercase = lowercase
        self.smooth = smooth
        self.brevity_penalty = brevity_penalty
        super().__init__(pred_name=pred_name, label_name=label_name)

    def reset(self) -> None:
        self.reference_length = 0
        self.hypothesis_length = 0
        self.matches_by_order = [0] * self.max_order
        self.possible_matches_by_order = [0] * self.max_order

    def add(self, predicted: Sequence[MaybeList[str]],
            labels: Sequence[MaybeList[str]]) -> None:
        for (reference, hypothesis) in zip(labels, predicted):
            if isinstance(reference, str):
                reference = reference.split()
            if self.lowercase:
                reference = [x.lower() for x in reference]
            reference_ngram_counts = _get_ngrams(reference, self.max_order)

            if isinstance(hypothesis, str):
                hypothesis = hypothesis.split()
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
        if self.brevity_penalty:
            ratio = self.hypothesis_length / self.reference_length
            if ratio > 1.0:
                bp = 1.0
            elif abs(ratio) < 1e-8:
                bp = 0.0
            else:
                bp = math.exp(1 - 1.0 / ratio)

        bleu = geo_mean * bp
        return bleu * 100.0
