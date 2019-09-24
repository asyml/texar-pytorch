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
Mono text data class that define data reading, parsing, batching, and other
preprocessing operations.
"""
from enum import Enum
from typing import List, Optional

import torch

from texar.torch.data.data.data_base import DataSource
from texar.torch.data.data.dataset_utils import Batch, padded_batch
from texar.torch.data.data.text_data_base import (
    TextDataBase, TextLineDataSource)
from texar.torch.data.embedding import Embedding
from texar.torch.data.vocabulary import SpecialTokens, Vocab
from texar.torch.hyperparams import HParams
from texar.torch.utils import utils

__all__ = [
    "_default_mono_text_dataset_hparams",
    "MonoTextData",
]


class _LengthFilterMode(Enum):
    r"""Options of length filter mode.
    """
    TRUNC = "truncate"
    DISCARD = "discard"


def _default_mono_text_dataset_hparams():
    r"""Returns hyperparameters of a mono text dataset with default values.

    See :meth:`texar.torch.MonoTextData.default_hparams` for details.
    """
    return {
        "files": [],
        "compression_type": None,
        "vocab_file": "",
        "embedding_init": Embedding.default_hparams(),
        "delimiter": None,
        "max_seq_length": None,
        "length_filter_mode": "truncate",
        "pad_to_max_seq_length": False,
        "bos_token": SpecialTokens.BOS,
        "eos_token": SpecialTokens.EOS,
        "other_transformations": [],
        "variable_utterance": False,
        "utterance_delimiter": "|||",
        "max_utterance_cnt": 5,
        "data_name": None,
        "@no_typecheck": ["files"]
    }


# todo(avinash): Add variable utterance logic
class MonoTextData(TextDataBase[List[str], List[str]]):
    r"""Text data processor that reads single set of text files. This can be
    used for, e.g., language models, auto-encoders, etc.

    Args:
        hparams: A `dict` or instance of :class:`~texar.torch.HParams`
            containing hyperparameters. See :meth:`default_hparams` for the
            defaults.
        device: The device of the produced batches. For GPU training, set to
            current CUDA device.

    By default, the processor reads raw data files, performs tokenization,
    batching and other pre-processing steps, and results in a Dataset
    whose element is a python `dict` including three fields:

    "text":
        A list of ``[batch_size]`` elements each containing a list of
        **raw** text tokens of the sequences. Short sequences in the batch
        are padded with **empty string**. By default only ``EOS`` token is
        appended to each sequence. Out-of-vocabulary tokens are **NOT**
        replaced with ``UNK``.
    "text_ids":
        A list of ``[batch_size]`` elements each containing a list of token
        indexes of source sequences in the batch.
    "length":
        A list of ``[batch_size]`` elements of integers containing the length
        of each source sequence in the batch (including ``BOS`` and ``EOS``
        if added).

    The above field names can be accessed through :attr:`text_name`,
    :attr:`text_id_name`, :attr:`length_name`.

    Example:

        .. code-block:: python

            hparams={
                'dataset': { 'files': 'data.txt', 'vocab_file': 'vocab.txt' },
                'batch_size': 1
            }
            data = MonoTextData(hparams)
            iterator = DataIterator(data)
            for batch in iterator:
                # batch contains the following
                # batch_ == {
                #    'text': [['<BOS>', 'example', 'sequence', '<EOS>']],
                #    'text_ids': [[1, 5, 10, 2]],
                #    'length': [4]
                # }
    """

    _delimiter: Optional[str]
    _bos: Optional[str]
    _eos: Optional[str]
    _max_seq_length: Optional[int]
    _should_pad: bool

    def __init__(self, hparams, device: Optional[torch.device] = None,
                 vocab: Optional[Vocab] = None,
                 embedding: Optional[Embedding] = None,
                 data_source: Optional[DataSource] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        if self._hparams.dataset.variable_utterance:
            raise NotImplementedError

        # Create vocabulary
        self._bos_token = self._hparams.dataset.bos_token
        self._eos_token = self._hparams.dataset.eos_token
        self._other_transforms = self._hparams.dataset.other_transformations
        bos = utils.default_str(self._bos_token, SpecialTokens.BOS)
        eos = utils.default_str(self._eos_token, SpecialTokens.EOS)
        if vocab is None:
            self._vocab = Vocab(self._hparams.dataset.vocab_file,
                                bos_token=bos, eos_token=eos)
        else:
            self._vocab = vocab

        # Create embedding
        if embedding is not None:
            self._embedding = self.make_embedding(
                self._hparams.dataset.embedding_init,
                self._vocab.token_to_id_map_py)
        else:
            self._embedding = embedding

        self._delimiter = self._hparams.dataset.delimiter
        self._max_seq_length = self._hparams.dataset.max_seq_length
        self._length_filter_mode = _LengthFilterMode(
            self._hparams.dataset.length_filter_mode)
        self._pad_length = self._max_seq_length
        if self._pad_length is not None:
            self._pad_length += sum(int(x != '')
                                    for x in [self._bos_token, self._eos_token])

        if data_source is None:
            if (self._length_filter_mode is _LengthFilterMode.DISCARD and
                    self._max_seq_length is not None):
                data_source = TextLineDataSource(
                    self._hparams.dataset.files,
                    compression_type=self._hparams.dataset.compression_type,
                    delimiter=self._delimiter,
                    max_length=self._max_seq_length)
            else:
                data_source = TextLineDataSource(
                    self._hparams.dataset.files,
                    compression_type=self._hparams.dataset.compression_type)

        super().__init__(data_source, hparams, device=device)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters:

        .. code-block:: python

            {
                # (1) Hyperparameters specific to text dataset
                "dataset": {
                    "files": [],
                    "compression_type": None,
                    "vocab_file": "",
                    "embedding_init": {},
                    "delimiter": None,
                    "max_seq_length": None,
                    "length_filter_mode": "truncate",
                    "pad_to_max_seq_length": False,
                    "bos_token": "<BOS>"
                    "eos_token": "<EOS>"
                    "other_transformations": [],
                    "variable_utterance": False,
                    "utterance_delimiter": "|||",
                    "max_utterance_cnt": 5,
                    "data_name": None,
                }
                # (2) General hyperparameters
                "num_epochs": 1,
                "batch_size": 64,
                "allow_smaller_final_batch": True,
                "shuffle": True,
                "shuffle_buffer_size": None,
                "shard_and_shuffle": False,
                "num_parallel_calls": 1,
                "prefetch_buffer_size": 0,
                "max_dataset_size": -1,
                "seed": None,
                "name": "mono_text_data",
                # (3) Bucketing
                "bucket_boundaries": [],
                "bucket_batch_sizes": None,
                "bucket_length_fn": None,
            }

        Here:

        1. For the hyperparameters in the :attr:`"dataset"` field:

          `"files"`: str or list
              A (list of) text file path(s).

              Each line contains a single text sequence.

          `"compression_type"`: str, optional
              One of `None` (no compression), ``"ZLIB"``, or ``"GZIP"``.

          `"vocab_file"`: str
              Path to vocabulary file. Each line of the file should contain
              one vocabulary token.

              Used to create an instance of :class:`~texar.torch.data.Vocab`.

          `"embedding_init"`: dict
              The hyperparameters for pre-trained embedding loading and
              initialization.

              The structure and default values are defined in
              :meth:`texar.torch.data.Embedding.default_hparams`.

          `"delimiter"`: str, optional
              The delimiter to split each line of the text files into tokens.
              If `None` (default), behavior will be equivalent to `str.split()`,
              i.e. split on any blank character.

          `"max_seq_length"`: int, optional
              Maximum length of output sequences. Data samples exceeding the
              length will be truncated or discarded according to
              :attr:`"length_filter_mode"`. The length does not include
              any added :attr:`"bos_token"` or :attr:`"eos_token"`. If
              `None` (default), no filtering is performed.

          `"length_filter_mode"`: str
              Either ``"truncate"`` or ``"discard"``. If ``"truncate"``
              (default), tokens exceeding :attr:`"max_seq_length"` will be
              truncated.
              If ``"discard"``, data samples longer than
              :attr:`"max_seq_length"` will be discarded.

          `"pad_to_max_seq_length"`: bool
              If `True`, pad all data instances to length
              :attr:`"max_seq_length"`.
              Raises error if :attr:`"max_seq_length"` is not provided.

          `"bos_token"`: str
              The Begin-Of-Sequence token prepended to each sequence.

              Set to an empty string to avoid prepending.

          `"eos_token"`: str
              The End-Of-Sequence token appended to each sequence.

              Set to an empty string to avoid appending.

          `"other_transformations"`: list
              A list of transformation functions or function names/paths to
              further transform each single data instance.

              (More documentations to be added.)

          `"variable_utterance"`: bool
              If `True`, each line of the text file is considered to contain
              multiple sequences (utterances) separated by
              :attr:`"utterance_delimiter"`.

              For example, in dialog data, each line can contain a series of
              dialog history utterances. See the example in
              `examples/hierarchical_dialog` for a use case.

              .. warning::
                  Variable utterances is not yet supported. This option (and
                  related ones below) will be ignored.

          `"utterance_delimiter"`: str
              The delimiter to split over utterance level. Should not be the
              same with :attr:`"delimiter"`. Used only when
              :attr:`"variable_utterance"` is `True`.

          `"max_utterance_cnt"`: int
              Maximally allowed number of utterances in a data instance.
              Extra utterances are truncated out.

          `"data_name"`: str
              Name of the dataset.

        2. For the **general** hyperparameters, see
        :meth:`texar.torch.data.DatasetBase.default_hparams` for details.

        3. **Bucketing** is to group elements of the dataset
        together by length and then pad and batch. For bucketing
        hyperparameters:

          `"bucket_boundaries"`: list
              An int list containing the upper length boundaries of the
              buckets.

              Set to an empty list (default) to disable bucketing.

          `"bucket_batch_sizes"`: list
              An int list containing batch size per bucket. Length should be
              `len(bucket_boundaries) + 1`.

              If `None`, every bucket will have the same batch size specified
              in :attr:`batch_size`.

          `"bucket_length_fn"`: str or callable
              Function maps dataset element to ``int``, determines
              the length of the element.

              This can be a function, or the name or full module path to the
              function. If function name is given, the function must be in the
              :mod:`texar.torch.custom` module.

              If `None` (default), length is determined by the number of
              tokens (including BOS and EOS if added) of the element.

          .. warning::
              Bucketing is not yet supported. These options will be ignored.

        """
        hparams = TextDataBase.default_hparams()
        hparams["name"] = "mono_text_data"
        hparams.update({
            "dataset": _default_mono_text_dataset_hparams()
        })
        return hparams

    @staticmethod
    def make_embedding(emb_hparams, token_to_id_map):
        r"""Optionally loads embedding from file (if provided), and returns
        an instance of :class:`texar.torch.data.Embedding`.
        """
        embedding = None
        if emb_hparams["file"] is not None and len(emb_hparams["file"]) > 0:
            embedding = Embedding(token_to_id_map, emb_hparams)
        return embedding

    def process(self, raw_example: List[str]) -> List[str]:
        # Truncates sentences and appends BOS/EOS tokens.
        words = raw_example
        if (self._max_seq_length is not None and
                len(words) > self._max_seq_length):
            if self._length_filter_mode is _LengthFilterMode.TRUNC:
                words = words[:self._max_seq_length]

        if self._hparams.dataset["bos_token"] != '':
            words.insert(0, self._hparams.dataset["bos_token"])
        if self._hparams.dataset["eos_token"] != '':
            words.append(self._hparams.dataset["eos_token"])

        # Apply the "other transformations".
        for transform in self._other_transforms:
            words = transform(words)

        return words

    def collate(self, examples: List[List[str]]) -> Batch:
        # For `MonoTextData`, each example is represented as a list of strings.
        # `_collate` takes care of padding and numericalization.

        # If `pad_length` is `None`, pad to the longest sentence in the batch.
        text_ids = [self._vocab.map_tokens_to_ids_py(sent) for sent in examples]
        text_ids, lengths = padded_batch(text_ids, self._pad_length,
                                         pad_value=self._vocab.pad_token_id)
        # Also pad the examples
        pad_length = self._pad_length or max(lengths)
        examples = [
            sent + [''] * (pad_length - len(sent))
            if len(sent) < pad_length else sent
            for sent in examples
        ]

        text_ids = torch.from_numpy(text_ids)
        lengths = torch.tensor(lengths, dtype=torch.long)
        batch = {self.text_name: examples, self.text_id_name: text_ids,
                 self.length_name: lengths}
        return Batch(len(examples), batch=batch)

    def list_items(self) -> List[str]:
        r"""Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        items = ['text', 'text_ids', 'length']
        data_name = self._hparams.dataset.data_name
        if data_name is not None:
            items = [data_name + '_' + item for item in items]
        return items

    @property
    def vocab(self) -> Vocab:
        r"""The vocabulary, an instance of :class:`~texar.torch.data.Vocab`.
        """
        return self._vocab

    @property
    def text_name(self):
        r"""The name for the text field"""
        if self.hparams.dataset["data_name"]:
            name = "{}_text".format(self.hparams.dataset["data_name"])
        else:
            name = "text"
        return name

    @property
    def text_id_name(self):
        r"""The name for text ids"""
        if self.hparams.dataset["data_name"]:
            name = "{}_text_ids".format(self.hparams.dataset["data_name"])
        else:
            name = "text_ids"
        return name

    @property
    def length_name(self):
        r"""The name for text length"""
        if self.hparams.dataset["data_name"]:
            name = "{}_length".format(self.hparams.dataset["data_name"])
        else:
            name = "length"
        return name

    @property
    def embedding_init_value(self):
        r"""The `Tensor` containing the embedding value loaded from file.
        `None` if embedding is not specified.
        """
        if self._embedding is None:
            return None
        return self._embedding.word_vecs
