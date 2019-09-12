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
Utils of BioBERT Modules.
"""

from abc import ABC

from texar.torch.modules.pretrained.bert import PretrainedBERTMixin

__all__ = [
    "PretrainedBioBERTMixin",
]

_BIOBERT_PATH = "https://github.com/naver/biobert-pretrained/releases/download/"


class PretrainedBioBERTMixin(PretrainedBERTMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the BioBERT model.

    The BioBERT model was proposed in (`Lee et al`. 2019)
    `BioBERT: a pre-trained biomedical language representation model for biomedical text mining`_
    . A domain specific language representation model pre-trained on
    large-scale biomedical corpora. Based on the BERT architecture, BioBERT
    effectively transfers the knowledge from a large amount of biomedical
    texts to biomedical text mining models with minimal task-specific
    architecture modifications. Available model names include:

      * ``biobert-v1.0-pmc``: BioBERT v1.0 (+ PMC 270K) - based on
        BERT-base-Cased (same vocabulary)
      * ``biobert-v1.0-pubmed-pmc``: BioBERT v1.0 (+ PubMed 200K + PMC 270K) -
        based on BERT-base-Cased (same vocabulary)
      * ``biobert-v1.0-pubmed``: BioBERT v1.0 (+ PubMed 200K) - based on
        BERT-base-Cased (same vocabulary)
      * ``biobert-v1.1-pubmed``: BioBERT v1.1 (+ PubMed 1M) - based on
        BERT-base-Cased (same vocabulary)

    We provide the following BERT classes:

      * :class:`~texar.torch.modules.BioBERTEncoder` for text encoding.
      * :class:`~texar.torch.modules.BioBERTClassifier` for text
        classification and sequence tagging.

    .. _`BioBERT: a pre-trained biomedical language representation model for biomedical text mining`:
        https://arxiv.org/abs/1901.08746

    """

    _MODEL_NAME = "BioBERT"
    _MODEL2URL = {
        'biobert-v1.0-pmc':
            _BIOBERT_PATH + 'v1.0-pmc/biobert_v1.0_pmc.tar.gz',
        'biobert-v1.0-pubmed-pmc':
            _BIOBERT_PATH + 'v1.0-pubmed-pmc/biobert_v1.0_pubmed_pmc.tar.gz',
        'biobert-v1.0-pubmed':
            _BIOBERT_PATH + 'v1.0-pubmed/biobert_v1.0_pubmed.tar.gz',
        'biobert-v1.1-pubmed':
            _BIOBERT_PATH + 'v1.1-pubmed/biobert_v1.1_pubmed.tar.gz',
    }
    _MODEL2CKPT = {
        'biobert-v1.0-pmc': 'biobert_model.ckpt',
        'biobert-v1.0-pubmed-pmc': 'biobert_model.ckpt',
        'biobert-v1.0-pubmed': 'biobert_model.ckpt',
        'biobert-v1.1-pubmed': 'model.ckpt-1000000'
    }
