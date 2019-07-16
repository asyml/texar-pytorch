# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2018 Texar
# ==============================================================================
"""BLEU metric utililities used for MT eval.

Usage: python bleu_tool.py --translation=my-wmt13.de --reference=wmt13_deen.de
"""
# This also:
# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
# BLEU score will be similar to the one obtained using: mteval-v14.pl
# Note:compound splitting is not implemented in this module

import re
import sys
import unicodedata
from argparse import ArgumentParser

from texar.evals.bleu import corpus_bleu


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


def bleu_tokenize(string):
    r"""Tokenize a string following the official BLEU implementation.

    See
    `https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983`_
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).

    Note that a number (e.g. a year) followed by a dot at the end of sentence
    is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
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


def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
    """Compute BLEU for two files (reference and hypothesis translation)."""
    ref_lines = open(ref_filename, encoding="utf-8").read().splitlines()
    hyp_lines = open(hyp_filename, encoding="utf-8").read().splitlines()
    assert len(ref_lines) == len(hyp_lines)
    ref_tokens = [[bleu_tokenize(x)] for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
    return corpus_bleu(list_of_references=ref_tokens,
                       hypotheses=hyp_tokens,
                       lowercase=(not case_sensitive))


def main():
    parser = ArgumentParser(
        description="Compute BLEU score. \
        Usage: t2t-bleu --translation=my-wmt13.de --reference=wmt13_deen.de"
    )

    parser.add_argument("--translation", type=str)
    parser.add_argument("--reference", type=str)
    args = parser.parse_args()

    bleu = bleu_wrapper(args.reference, args.translation, case_sensitive=False)
    print("BLEU_uncased = %6.2f" % bleu)
    bleu = bleu_wrapper(args.reference, args.translation, case_sensitive=True)
    print("BLEU_cased = %6.2f" % bleu)


if __name__ == "__main__":
    main()
