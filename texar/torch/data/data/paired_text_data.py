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
Paired text data that consists of source text and target text.
"""
import math
from typing import List, Optional, Tuple

import torch

from texar.torch.data.data.data_base import (
    DataSource, FilterDataSource, ZipDataSource)
from texar.torch.data.data.dataset_utils import Batch, padded_batch
from texar.torch.data.data.mono_text_data import (
    MonoTextData, _LengthFilterMode, _default_mono_text_dataset_hparams)
from texar.torch.data.data.text_data_base import (
    TextDataBase, TextLineDataSource)
from texar.torch.data.embedding import Embedding
from texar.torch.data.vocabulary import SpecialTokens, Vocab
from texar.torch.hyperparams import HParams
from texar.torch.utils import utils

__all__ = [
    "_default_paired_text_dataset_hparams",
    "PairedTextData",
]


def _default_paired_text_dataset_hparams():
    r"""Returns hyperparameters of a paired text dataset with default values.

    See :meth:`texar.torch.data.PairedTextData.default_hparams` for details.
    """
    source_hparams = _default_mono_text_dataset_hparams()
    source_hparams["bos_token"] = None
    source_hparams["data_name"] = "source"
    target_hparams = _default_mono_text_dataset_hparams()
    target_hparams.update(
        {
            "vocab_share": False,
            "embedding_init_share": False,
            "processing_share": False,
            "data_name": "target"
        }
    )
    return {
        "source_dataset": source_hparams,
        "target_dataset": target_hparams
    }


class PairedTextData(TextDataBase[Tuple[List[str], List[str]],
                                  Tuple[List[str], List[str]]]):
    r"""Text data processor that reads parallel source and target text.
    This can be used in, e.g., seq2seq models.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.
        device: The device of the produced batches. For GPU training, set to
            current CUDA device.

    By default, the processor reads raw data files, performs tokenization,
    batching and other pre-processing steps, and results in a Dataset
    whose element is a python `dict` including six fields:

    "source_text":
        A list of ``[batch_size]`` elements each containing a list of
        **raw** text tokens of source sequences. Short sequences in the
        batch are padded with **empty string**. By default only ``EOS``
        token is appended to each sequence. Out-of-vocabulary tokens are
        **NOT** replaced with ``UNK``.
    "source_text_ids":
        A list of ``[batch_size]`` elements each containing a list of token
        indexes of source sequences in the batch.
    "source_length":
        A list of ``[batch_size]`` elements of integers containing the length
        of each source sequence in the batch.
    "target_text":
        A list same as "source_text" but for target sequences. By default
        both BOS and EOS are added.
    "target_text_ids":
        A list same as "source_text_ids" but for target sequences.
    "target_length":
        An list same as "source_length" but for target sequences.

    The above field names can be accessed through :attr:`source_text_name`,
    :attr:`source_text_id_name`, :attr:`source_length_name`, and those prefixed
    with ``target_``, respectively.

    Example:

    .. code-block:: python

        hparams={
            'source_dataset': {'files': 's', 'vocab_file': 'vs'},
            'target_dataset': {'files': ['t1', 't2'], 'vocab_file': 'vt'},
            'batch_size': 1
        }
        data = PairedTextData(hparams)
        iterator = DataIterator(data)

        for batch in iterator:
            # batch contains the following
            # batch_ == {
            #    'source_text': [['source', 'sequence', '<EOS>']],
            #    'source_text_ids': [[5, 10, 2]],
            #    'source_length': [3]
            #    'target_text': [['<BOS>', 'target', 'sequence', '1',
                                '<EOS>']],
            #    'target_text_ids': [[1, 6, 10, 20, 2]],
            #    'target_length': [5]
            # }

    """

    def __init__(self, hparams, device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())

        src_hparams = self.hparams.source_dataset
        tgt_hparams = self.hparams.target_dataset

        # create vocabulary
        self._src_bos_token = src_hparams["bos_token"]
        self._src_eos_token = src_hparams["eos_token"]
        self._src_transforms = src_hparams["other_transformations"]
        self._src_vocab = Vocab(src_hparams.vocab_file,
                                bos_token=src_hparams.bos_token,
                                eos_token=src_hparams.eos_token)

        if tgt_hparams["processing_share"]:
            self._tgt_bos_token = src_hparams["bos_token"]
            self._tgt_eos_token = src_hparams["eos_token"]
        else:
            self._tgt_bos_token = tgt_hparams["bos_token"]
            self._tgt_eos_token = tgt_hparams["eos_token"]
        tgt_bos_token = utils.default_str(self._tgt_bos_token,
                                          SpecialTokens.BOS)
        tgt_eos_token = utils.default_str(self._tgt_eos_token,
                                          SpecialTokens.EOS)
        if tgt_hparams["vocab_share"]:
            if tgt_bos_token == self._src_vocab.bos_token and \
                    tgt_eos_token == self._src_vocab.eos_token:
                self._tgt_vocab = self._src_vocab
            else:
                self._tgt_vocab = Vocab(src_hparams["vocab_file"],
                                        bos_token=tgt_bos_token,
                                        eos_token=tgt_eos_token)
        else:
            self._tgt_vocab = Vocab(tgt_hparams["vocab_file"],
                                    bos_token=tgt_bos_token,
                                    eos_token=tgt_eos_token)

        # create embeddings
        self._src_embedding = MonoTextData.make_embedding(
            src_hparams.embedding_init, self._src_vocab.token_to_id_map_py)

        if self._hparams.target_dataset.embedding_init_share:
            self._tgt_embedding = self._src_embedding
        else:
            tgt_emb_file = tgt_hparams.embedding_init["file"]
            self._tgt_embedding = None
            if tgt_emb_file is not None and tgt_emb_file != "":
                self._tgt_embedding = MonoTextData.make_embedding(
                    self._tgt_vocab.token_to_id_map_py,
                    tgt_hparams.embedding_init)

        # create data source
        self._src_delimiter = src_hparams.delimiter
        self._src_max_seq_length = src_hparams.max_seq_length
        self._src_length_filter_mode = _LengthFilterMode(
            src_hparams.length_filter_mode)
        self._src_pad_length = self._src_max_seq_length
        if self._src_pad_length is not None:
            self._src_pad_length += sum(int(x is not None and x != '')
                                        for x in [src_hparams.bos_token,
                                                  src_hparams.eos_token])

        src_data_source = TextLineDataSource(
            src_hparams.files, compression_type=src_hparams.compression_type)

        self._tgt_transforms = tgt_hparams["other_transformations"]
        self._tgt_delimiter = tgt_hparams.delimiter
        self._tgt_max_seq_length = tgt_hparams.max_seq_length
        self._tgt_length_filter_mode = _LengthFilterMode(
            tgt_hparams.length_filter_mode)
        self._tgt_pad_length = self._tgt_max_seq_length
        if self._tgt_pad_length is not None:
            self._tgt_pad_length += sum(int(x is not None and x != '')
                                        for x in [tgt_hparams.bos_token,
                                                  tgt_hparams.eos_token])

        tgt_data_source = TextLineDataSource(
            tgt_hparams.files, compression_type=tgt_hparams.compression_type)

        data_source: DataSource[Tuple[List[str], List[str]]]
        data_source = ZipDataSource(  # type: ignore
            src_data_source, tgt_data_source)
        if ((self._src_length_filter_mode is _LengthFilterMode.DISCARD and
             self._src_max_seq_length is not None) or
                (self._tgt_length_filter_mode is _LengthFilterMode.DISCARD and
                 self._tgt_length_filter_mode is not None)):
            max_source_length = self._src_max_seq_length or math.inf
            max_tgt_length = self._tgt_max_seq_length or math.inf

            def filter_fn(raw_example):
                return (len(raw_example[0]) <= max_source_length and
                        len(raw_example[1]) <= max_tgt_length)

            data_source = FilterDataSource(data_source, filter_fn)

        super().__init__(data_source, hparams, device=device)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                # (1) Hyperparams specific to text dataset
                "source_dataset": {
                    "files": [],
                    "compression_type": None,
                    "vocab_file": "",
                    "embedding_init": {},
                    "delimiter": None,
                    "max_seq_length": None,
                    "length_filter_mode": "truncate",
                    "pad_to_max_seq_length": False,
                    "bos_token": None,
                    "eos_token": "<EOS>",
                    "other_transformations": [],
                    "variable_utterance": False,
                    "utterance_delimiter": "|||",
                    "max_utterance_cnt": 5,
                    "data_name": "source",
                },
                "target_dataset": {
                    # ...
                    # Same fields are allowed as in "source_dataset" with the
                    # same default values, except the
                    # following new fields/values:
                    "bos_token": "<BOS>"
                    "vocab_share": False,
                    "embedding_init_share": False,
                    "processing_share": False,
                    "data_name": "target"
                }
                # (2) General hyperparams
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
                "name": "paired_text_data",
                # (3) Bucketing
                "bucket_boundaries": [],
                "bucket_batch_sizes": None,
                "bucket_length_fn": None,
            }

        Here:

        1. Hyperparameters in the :attr:`"source_dataset"` and
           :attr:`"target_dataset"` fields have the same definition as those
           in :meth:`texar.torch.data.MonoTextData.default_hparams`, for source
           and target text, respectively.

           For the new hyperparameters in :attr:`"target_dataset"`:

           `"vocab_share"`: bool
               Whether to share the vocabulary of source.
               If `True`, the vocab file of target is ignored.

           `"embedding_init_share"`: bool
               Whether to share the embedding initial value of source. If
               `True`, :attr:`"embedding_init"` of target is ignored.

              :attr:`"vocab_share"` must be true to share the embedding
              initial value.

           `"processing_share"`: bool
               Whether to share the processing configurations of source,
               including `"delimiter"`, `"bos_token"`, `"eos_token"`, and
               `"other_transformations"`.

        2. For the **general** hyperparameters, see
           :meth:`texar.torch.data.DatasetBase.default_hparams` for details.

        3. For **bucketing** hyperparameters, see
           :meth:`texar.torch.data.MonoTextData.default_hparams` for details,
           except that the default `"bucket_length_fn"` is the maximum sequence
           length of source and target sequences.

           .. warning::
               Bucketing is not yet supported. These options will be ignored.

        """
        hparams = TextDataBase.default_hparams()
        hparams["name"] = "paired_text_data"
        hparams.update(_default_paired_text_dataset_hparams())
        return hparams

    @staticmethod
    def make_embedding(src_emb_hparams, src_token_to_id_map,
                       tgt_emb_hparams=None, tgt_token_to_id_map=None,
                       emb_init_share=False):
        r"""Optionally loads source and target embeddings from files (if
        provided), and returns respective :class:`texar.torch.data.Embedding`
        instances.
        """
        src_embedding = MonoTextData.make_embedding(
            src_emb_hparams, src_token_to_id_map)

        if emb_init_share:
            tgt_embedding = src_embedding
        else:
            tgt_emb_file = tgt_emb_hparams["file"]
            tgt_embedding = None
            if tgt_emb_file is not None and tgt_emb_file != "":
                tgt_embedding = Embedding(tgt_token_to_id_map, tgt_emb_hparams)

        return src_embedding, tgt_embedding

    def process(self, raw_example: Tuple[List[str], List[str]]) -> \
            Tuple[List[str], List[str]]:
        # `_process` truncates sentences and appends BOS/EOS tokens.
        src_words = raw_example[0]
        if (self._src_max_seq_length is not None and
                len(src_words) > self._src_max_seq_length):
            if self._src_length_filter_mode is _LengthFilterMode.TRUNC:
                src_words = src_words[:self._src_max_seq_length]

        if self._src_bos_token is not None and self._src_bos_token != '':
            src_words.insert(0, self._src_bos_token)
        if self._src_eos_token is not None and self._src_eos_token != '':
            src_words.append(self._src_eos_token)

        # apply the transformations to source
        for transform in self._src_transforms:
            src_words = transform(src_words)

        tgt_words = raw_example[1]
        if (self._tgt_max_seq_length is not None and
                len(tgt_words) > self._tgt_max_seq_length):
            if self._tgt_length_filter_mode is _LengthFilterMode.TRUNC:
                tgt_words = tgt_words[:self._tgt_max_seq_length]

        if self._tgt_bos_token is not None and self._tgt_bos_token != '':
            tgt_words.insert(0, self._tgt_bos_token)
        if self._tgt_eos_token is not None and self._tgt_eos_token != '':
            tgt_words.append(self._tgt_eos_token)

        # apply the transformations to target
        for transform in self._tgt_transforms:
            tgt_words = transform(tgt_words)

        return src_words, tgt_words

    @staticmethod
    def _get_name_prefix(src_hparams, tgt_hparams):
        name_prefix = [
            src_hparams["data_name"], tgt_hparams["data_name"]]
        if name_prefix[0] == name_prefix[1]:
            raise ValueError("'data_name' of source and target "
                             "datasets cannot be the same.")
        return name_prefix

    def collate(self, examples: List[Tuple[List[str], List[str]]]) -> Batch:
        # For `PairedTextData`, each example is represented as a tuple of list
        # of strings.
        # `_collate` takes care of padding and numericalization.

        # If `pad_length` is `None`, pad to the longest sentence in the batch.
        src_examples = [example[0] for example in examples]
        source_ids = [self._src_vocab.map_tokens_to_ids_py(sent) for sent
                      in src_examples]
        source_ids, source_lengths = \
            padded_batch(source_ids,
                         self._src_pad_length,
                         pad_value=self._src_vocab.pad_token_id)
        src_pad_length = self._src_pad_length or max(source_lengths)
        src_examples = [
            sent + [''] * (src_pad_length - len(sent))
            if len(sent) < src_pad_length else sent
            for sent in src_examples
        ]

        source_ids = torch.from_numpy(source_ids)
        source_lengths = torch.tensor(source_lengths, dtype=torch.long)

        tgt_examples = [example[1] for example in examples]
        target_ids = [self._tgt_vocab.map_tokens_to_ids_py(sent) for sent
                      in tgt_examples]
        target_ids, target_lengths = \
            padded_batch(target_ids,
                         self._tgt_pad_length,
                         pad_value=self._tgt_vocab.pad_token_id)
        tgt_pad_length = self._tgt_pad_length or max(target_lengths)
        tgt_examples = [
            sent + [''] * (tgt_pad_length - len(sent))
            if len(sent) < tgt_pad_length else sent
            for sent in tgt_examples
        ]

        target_ids = torch.from_numpy(target_ids)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)

        return Batch(len(examples), source_text=src_examples,
                     source_text_ids=source_ids, source_length=source_lengths,
                     target_text=tgt_examples, target_text_ids=target_ids,
                     target_length=target_lengths)

    def list_items(self) -> List[str]:
        r"""Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        items = ['text', 'text_ids', 'length']
        src_name = self._hparams.source_dataset['data_name']
        tgt_name = self._hparams.target_dataset['data_name']

        if src_name is not None:
            src_items = [src_name + '_' + item for item in items]
        else:
            src_items = items

        if tgt_name is not None:
            tgt_items = [tgt_name + '_' + item for item in items]
        else:
            tgt_items = items

        return src_items + tgt_items

    @property
    def vocab(self):
        r"""A pair instances of :class:`~texar.torch.data.Vocab` that are source
        and target vocabs, respectively.
        """
        return self._src_vocab, self._tgt_vocab

    @property
    def source_vocab(self):
        r"""The source vocab, an instance of :class:`~texar.torch.data.Vocab`.
        """
        return self._src_vocab

    @property
    def target_vocab(self):
        r"""The target vocab, an instance of :class:`~texar.torch.data.Vocab`.
        """
        return self._tgt_vocab

    @property
    def source_text_name(self):
        r"""The name for source text"""
        name = "{}_text".format(self.hparams.source_dataset["data_name"])
        return name

    @property
    def source_text_id_name(self):
        r"""The name for source text id"""
        name = "{}_text_ids".format(self.hparams.source_dataset["data_name"])
        return name

    @property
    def source_length_name(self):
        r"""The name for source length"""
        name = "{}_length".format(self.hparams.source_dataset["data_name"])
        return name

    @property
    def target_text_name(self):
        r"""The name for target text"""
        name = "{}_text".format(self.hparams.target_dataset["data_name"])
        return name

    @property
    def target_text_id_name(self):
        r"""The name for target text id"""
        name = "{}_text_ids".format(self.hparams.target_dataset["data_name"])
        return name

    @property
    def target_length_name(self):
        r"""The name for target length"""
        name = "{}_length".format(self.hparams.target_dataset["data_name"])
        return name

    def embedding_init_value(self):
        r"""A pair of `Tensor` containing the embedding values of source and
        target data loaded from file.
        """
        src_emb = self.hparams.source_dataset["embedding_init"]
        tgt_emb = self.hparams.target_dataser["embedding_init"]
        return src_emb, tgt_emb
