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
Python implementation of BLEU adapted from:
    `https://github.com/tensorflow/models/blob/master/official/transformer/compute_bleu.py`
"""

from typing import Callable, Counter, List, Tuple

import re
import sys
import unicodedata
import collections
import math
import numpy as np

from texar.torch.evals.bleu import corpus_bleu
from texar.torch.evals.bleu_moses import corpus_bleu_moses
from texar.torch.utils.types import MaybeList

__all__ = [
    "corpus_bleu_transformer",
    "bleu_transformer_tokenize",
    "file_bleu",
]


def _get_ngrams(segment: List[str],
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
    ngram_counts: collections.Counter = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def corpus_bleu_transformer(reference_corpus: List[List[str]],
                            translation_corpus: List[List[str]],
                            max_order: int = 4,
                            use_bp: bool = True) -> float:
    r"""Computes BLEU score of translated segments against references.

    This BLEU has been used in evaluating Transformer (Vaswani et al.)
    "Attention is all you need" for machine translation. The resulting BLEU
    score are usually a bit higher than that in
    `texar.torch.evals.corpus_bleu` and `texar.torch.evals.corpus_bleu_moses`.

    Args:
        reference_corpus: list of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        use_bp: boolean, whether to apply brevity penalty.

    Returns:
        BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0.0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    for (references, translations) in zip(reference_corpus, translation_corpus):
        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams(references, max_order)
        translation_ngram_counts = _get_ngrams(translations, max_order)

        overlap = dict(
            (ngram, min(count, translation_ngram_counts[ngram]))
            for ngram, count in ref_ngram_counts.items()
        )

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += \
                translation_ngram_counts[ngram]

    precisions = [0.0] * max_order
    smooth = 1.0

    for i in range(max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = (matches_by_order[i] /
                                 possible_matches_by_order[i])
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        ratio = translation_length / reference_length
        if ratio <= 0:
            bp = 0
        elif ratio < 1.0:
            bp = math.exp(1 - 1.0 / ratio)
        else:
            bp = 1.0

    bleu = geo_mean * bp

    return np.float32(bleu) * 100


class UnicodeRegex:
    """Ad-hoc hack to recognize all punctuation and symbols."""

    def __init__(self):
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    @staticmethod
    def property_chars(prefix):
        return "".join(
            chr(x)
            for x in range(sys.maxunicode)
            if unicodedata.category(chr(x)).startswith(prefix)
        )


uregex = UnicodeRegex()


def bleu_transformer_tokenize(string: str) -> List[str]:
    r"""Tokenize a string following the official BLEU implementation.

    The BLEU scores from `multi-bleu.perl` depend on your `tokenizer`, which is
    unlikely to be reproducible from your experiment or consistent across
    different users. This function provides a standard tokenization following
    `mteval-v14.pl`.

    See
    `https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983`.
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).

    Note that a number (e.g. a year) followed by a dot at the end of sentence
    is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original `mteval-v14.pl`
    and we want to be consistent with it.

    Args:
        string: the input string

    Returns:
        a list of tokens
    """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()


def file_bleu(ref_filename: str,
              hyp_filename: str,
              bleu_version: str = "corpus_bleu_transformer",
              case_sensitive: bool = False) -> float:
    r"""Compute BLEU for two files (reference and hypothesis translation).

    Args:
        ref_filename: Reference file path.
        hyp_filename: Hypothesis file path.
        bleu_version: A str with the name of a BLEU computing method selected
            in the list of: `corpus_bleu`, `corpus_bleu_moses`,
            `corpus_bleu_transformer`.
        case_sensitive: If `False`, lowercase reference and hypothesis
            tokens.

    Returns:
        BLEU score.
    """
    with open(ref_filename, encoding="utf-8") as f:
        ref_lines = f.read().splitlines()

    with open(hyp_filename, encoding="utf-8") as f:
        hyp_lines = f.read().splitlines()

    if len(ref_lines) != len(hyp_lines):
        raise ValueError(
            "Reference and translation files have different number of lines")

    if not case_sensitive:
        ref_lines = [x.lower() for x in ref_lines]
        hyp_lines = [x.lower() for x in hyp_lines]

    ref_tokens: List[MaybeList[List[str]]]
    if bleu_version == "corpus_bleu_transformer":
        ref_tokens = [bleu_transformer_tokenize(x) for x in ref_lines]
    else:
        ref_tokens = [[bleu_transformer_tokenize(x)] for x in ref_lines]
    hyp_tokens = [bleu_transformer_tokenize(x) for x in hyp_lines]

    bleu_dict = {
        "corpus_bleu": corpus_bleu,
        "corpus_bleu_moses": corpus_bleu_moses,
        "corpus_bleu_transformer": corpus_bleu_transformer
    }
    fn: Callable[[List[MaybeList[List[str]]], List[List[str]]], float]
    fn = bleu_dict[bleu_version]  # type: ignore

    return fn(ref_tokens, hyp_tokens)
