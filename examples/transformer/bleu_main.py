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

Usage: python bleu_main.py --translation=my-wmt13.de --reference=wmt13_deen.de
"""
# This also:
# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
# BLEU score will be similar to the one obtained using: mteval-v14.pl
# Note:compound splitting is not implemented in this module


from argparse import ArgumentParser

import texar.torch as tx


def main() -> None:
    parser = ArgumentParser(
        description="Compute BLEU score. \
        Usage: t2t-bleu --translation=my-wmt13.de --reference=wmt13_deen.de"
    )

    parser.add_argument("--translation", type=str)
    parser.add_argument("--reference", type=str)
    args = parser.parse_args()

    bleu = tx.evals.file_bleu(
        args.reference, args.translation, case_sensitive=False)
    print(f"BLEU_uncased = {bleu:6.2f}")
    bleu = tx.evals.file_bleu(
        args.reference, args.translation, case_sensitive=True)
    print(f"BLEU_cased = {bleu:6.2f}")


if __name__ == "__main__":
    main()
