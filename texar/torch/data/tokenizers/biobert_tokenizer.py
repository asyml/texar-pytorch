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
Pre-trained BioBERT tokenizer.
"""

from texar.torch.data.tokenizers.bert_tokenizer import BERTTokenizer

__all__ = [
    'BioBERTTokenizer',
]

_BIOBERT_PATH = "https://github.com/naver/biobert-pretrained/releases/download/"


class BioBERTTokenizer(BERTTokenizer):
    r"""Pre-trained BioBERT Tokenizer.
    :class:`~texar.data.BioBERTTokenizer` is identical to
    :class:`~texar.data.BERTTokenizer`.

    Args:
        pretrained_model_name (optional): a `str`, the name of
            pre-trained model (e.g., `biobert-v1.0-pmc`). Please refer to
            :class:`~texar.torch.modules.PretrainedBioBERTMixin` for
            all supported models.
            If None, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
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
    _MAX_INPUT_SIZE = {
        'biobert-v1.0-pmc': 512,
        'biobert-v1.0-pubmed-pmc': 512,
        'biobert-v1.0-pubmed': 512,
        'biobert-v1.1-pubmed': 512,
    }

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        * The tokenizer is determined by the constructor argument
          :attr:`pretrained_model_name` if it's specified. In this case,
          `hparams` are ignored.
        * Otherwise, the tokenizer is determined by
          `hparams['pretrained_model_name']` if it's specified. All other
          configurations in `hparams` are ignored.
        * If the above two are `None`, the tokenizer is defined by the
          configurations in `hparams`.

        .. code-block:: python

            {
                "pretrained_model_name": "biobert-v1.0-pmc",
                "vocab_file": None,
                "max_len": 512,
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "tokenize_chinese_chars": True,
                "do_lower_case": True,
                "do_basic_tokenize": True,
                "non_split_tokens": None,
                "name": "biobert_tokenizer",
            }

        Here:

        `"pretrained_model_name"`: str or None
            The name of the pre-trained BioBERT model.

        `"vocab_file"`: str or None
            The path to a one-wordpiece-per-line vocabulary file.

        `"max_len"`: int
            The maximum sequence length that this model might ever be used with.

        `"unk_token"`: str
            Unknown token.

        `"sep_token"`: str
            Separation token.

        `"pad_token"`: str
            Padding token.

        `"cls_token"`: str
            Classification token.

        `"mask_token"`: str
            Masking token.

        `"tokenize_chinese_chars"`: bool
            Whether to tokenize Chinese characters.

        `"do_lower_case"`: bool
            Whether to lower case the input
            Only has an effect when `do_basic_tokenize=True`

        `"do_basic_tokenize"`: bool
            Whether to do basic tokenization before wordpiece.

        `"non_split_tokens"`: list
            List of tokens which will never be split during tokenization.
            Only has an effect when `do_basic_tokenize=True`

        `"name"`: str
            Name of the tokenizer.

        """
        default_ = BERTTokenizer.default_hparams()
        default_['pretrained_model_name'] = 'biobert-v1.0-pmc'
        default_['name'] = 'biobert_tokenizer'
        return default_
