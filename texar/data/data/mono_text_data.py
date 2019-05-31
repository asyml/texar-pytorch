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

import numpy as np

import torch
from texar.data.data import dataset_utils as dsutils
from texar.data.data.dataset_utils import Batch
from texar.data.data.text_data_base import TextDataBase, TextLineDataSource
from texar.data.embedding import Embedding
from texar.data.vocabulary import SpecialTokens, Vocab
from texar.hyperparams import HParams
from texar.utils import utils
from texar.utils.dtypes import is_callable

# pylint: disable=invalid-name, arguments-differ, protected-access, no-member

__all__ = [
    "_default_mono_text_dataset_hparams",
    "MonoTextData"
]


class _LengthFilterMode(Enum):  # pylint: disable=no-init, too-few-public-methods
    r"""Options of length filter mode.
    """
    TRUNC = "truncate"
    DISCARD = "discard"


def _default_mono_text_dataset_hparams():
    r"""Returns hyperparameters of a mono text dataset with default values.

    See :meth:`texar.MonoTextData.default_hparams` for details.
    """
    return {
        "files": [],
        "compression_type": None,
        "vocab_file": "",
        "embedding_init": Embedding.default_hparams(),
        "delimiter": " ",
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


class MonoTextData(TextDataBase[str, List[str]]):
    r"""Text data processor that reads single set of text files. This can be
    used for, e.g., language models, auto-encoders, etc.

    Args:
        hparams: A `dict` or instance of :class:`~texar.HParams` containing
            hyperparameters. See :meth:`default_hparams` for the defaults.

    By default, the processor reads raw data files, performs tokenization,
    batching and other pre-processing steps, and results in a TF Dataset
    whose element is a python `dict` including three fields:

        - "text":
            A string Tensor of shape `[batch_size, max_time]` containing
            the **raw** text toknes. `max_time` is the length of the longest
            sequence in the batch.
            Short sequences in the batch are padded with **empty string**.
            BOS and EOS tokens are added as per
            :attr:`hparams`. Out-of-vocabulary tokens are **NOT** replaced
            with UNK.
        - "text_ids":
            An `int64` Tensor of shape `[batch_size, max_time]`
            containing the token indexes.
        - "length":
            An `int` Tensor of shape `[batch_size]` containing the
            length of each sequence in the batch (including BOS and
            EOS if added).

    If :attr:`'variable_utterance'` is set to `True` in :attr:`hparams`, the
    resulting dataset has elements with four fields:

        - "text":
            A string Tensor of shape
            `[batch_size, max_utterance, max_time]`, where *max_utterance* is
            either the maximum number of utterances in each elements of the
            batch, or :attr:`max_utterance_cnt` as specified in :attr:`hparams`.
        - "text_ids":
            An `int64` Tensor of shape
            `[batch_size, max_utterance, max_time]` containing the token
            indexes.
        - "length":
            An `int` Tensor of shape `[batch_size, max_utterance]`
            containing the length of each sequence in the batch.
        - "utterance_cnt":
            An `int` Tensor of shape `[batch_size]` containing
            the number of utterances of each element in the batch.

    The above field names can be accessed through :attr:`text_name`,
    :attr:`text_id_name`, :attr:`length_name`, and
    :attr:`utterance_cnt_name`, respectively.

    Example:

        .. code-block:: python

            hparams={
                'dataset': { 'files': 'data.txt', 'vocab_file': 'vocab.txt' },
                'batch_size': 1
            }
            data = MonoTextData(hparams)
            iterator = DataIterator(data)
            batch = iterator.get_next()

            iterator.switch_to_dataset(sess) # initializes the dataset
            batch_ = sess.run(batch)
            # batch_ == {
            #    'text': [['<BOS>', 'example', 'sequence', '<EOS>']],
            #    'text_ids': [[1, 5, 10, 2]],
            #    'length': [4]
            # }
    """

    _delimiter: str
    _bos: Optional[str]
    _eos: Optional[str]
    _max_seq_length: Optional[int]
    _should_pad: bool

    def __init__(self, hparams):
        dataset_hparams = HParams(hparams, self.default_hparams()).dataset
        if dataset_hparams.variable_utterance:
            raise NotImplementedError

        data_source = TextLineDataSource(
            dataset_hparams["files"],
            compression_type=dataset_hparams["compression_type"])

        super().__init__(data_source, hparams)

        # Create vocab and embedding
        self._vocab = self.make_vocab(dataset_hparams)
        self._embedding = self.make_embedding(
            dataset_hparams["embedding_init"], self._vocab.token_to_id_map_py)

        self._delimiter = self._hparams.dataset.delimiter
        self._bos = self._hparams.dataset.bos_token
        if self._bos == '':
            self._bos = None
        self._eos = self._hparams.dataset.eos_token
        if self._eos == '':
            self._eos = None
        self._max_seq_length = self._hparams.dataset.max_seq_length
        self._length_filter_mode = _LengthFilterMode(
            self._hparams.dataset.length_filter_mode)
        self._should_pad = self._hparams.dataset.pad_to_max_seq_length
        self._pad_length = self._max_seq_length
        if self._max_seq_length is not None:
            self._pad_length += sum(int(x is not None)
                                    for x in [self._bos, self._eos])

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
                    "delimiter": " ",
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

            "files" : str or list
                A (list of) text file path(s).

                Each line contains a single text sequence.

            "compression_type" : str, optional
                One of "" (no compression), "ZLIB", or "GZIP".

            "vocab_file": str
                Path to vocabulary file. Each line of the file should contain
                one vocabulary token.

                Used to create an instance of :class:`~texar.data.Vocab`.

            "embedding_init" : dict
                The hyperparameters for pre-trained embedding loading and
                initialization.

                The structure and default values are defined in
                :meth:`texar.data.Embedding.default_hparams`.

            "delimiter" : str
                The delimiter to split each line of the text files into tokens.

            "max_seq_length" : int, optional
                Maximum length of output sequences. Data samples exceeding the
                length will be truncated or discarded according to
                :attr:`"length_filter_mode"`. The length does not include
                any added
                :attr:`"bos_token"` or :attr:`"eos_token"`. If `None` (default),
                no filtering is performed.

            "length_filter_mode" : str
                Either "truncate" or "discard". If "truncate" (default),
                tokens exceeding the :attr:`"max_seq_length"` will be truncated.
                If "discard", data samples longer than the
                :attr:`"max_seq_length"`
                will be discarded.

            "pad_to_max_seq_length" : bool
                If `True`, pad all data instances to length
                :attr:`"max_seq_length"`.
                Raises error if :attr:`"max_seq_length"` is not provided.

            "bos_token" : str
                The Begin-Of-Sequence token prepended to each sequence.

                Set to an empty string to avoid prepending.

            "eos_token" : str
                The End-Of-Sequence token appended to each sequence.

                Set to an empty string to avoid appending.

            "other_transformations" : list
                A list of transformation functions or function names/paths to
                further transform each single data instance.

                (More documentations to be added.)

            "variable_utterance" : bool
                If `True`, each line of the text file is considered to contain
                multiple sequences (utterances) separated by
                :attr:`"utterance_delimiter"`.

                For example, in dialog data, each line can contain a series of
                dialog history utterances. See the example in
                `examples/hierarchical_dialog` for a use case.

            "utterance_delimiter" : str
                The delimiter to split over utterance level. Should not be the
                same with :attr:`"delimiter"`. Used only when
                :attr:`"variable_utterance"``==True`.

            "max_utterance_cnt" : int
                Maximally allowed number of utterances in a data instance.
                Extra utterances are truncated out.

            "data_name" : str
                Name of the dataset.

        2. For the **general** hyperparameters, see
        :meth:`texar.data.DataBase.default_hparams` for details.

        3. **Bucketing** is to group elements of the dataset together by length
        and then pad and batch. (See more at
        :tf_main:`bucket_by_sequence_length
        <contrib/data/bucket_by_sequence_length>`). For bucketing
        hyperparameters:

            "bucket_boundaries" : list
                An int list containing the upper length boundaries of the
                buckets.

                Set to an empty list (default) to disable bucketing.

            "bucket_batch_sizes" : list
                An int list containing batch size per bucket. Length should be
                `len(bucket_boundaries) + 1`.

                If `None`, every bucket will have the same batch size specified
                in :attr:`batch_size`.

            "bucket_length_fn" : str or callable
                Function maps dataset element to `tf.int32` scalar, determines
                the length of the element.

                This can be a function, or the name or full module path to the
                function. If function name is given, the function must be in the
                :mod:`texar.custom` module.

                If `None` (default), length is determined by the number of
                tokens (including BOS and EOS if added) of the element.

        """
        hparams = TextDataBase.default_hparams()
        hparams["name"] = "mono_text_data"
        hparams.update({
            "dataset": _default_mono_text_dataset_hparams()
        })
        return hparams

    @staticmethod
    def make_vocab(hparams):
        """Reads vocab file and returns an instance of
        :class:`texar.data.Vocab`.
        """
        bos_token = utils.default_str(
            hparams["bos_token"], SpecialTokens.BOS)
        eos_token = utils.default_str(
            hparams["eos_token"], SpecialTokens.EOS)
        vocab = Vocab(hparams["vocab_file"],
                      bos_token=bos_token, eos_token=eos_token)
        return vocab

    @staticmethod
    def make_embedding(emb_hparams, token_to_id_map):
        """Optionally loads embedding from file (if provided), and returns
        an instance of :class:`texar.data.Embedding`.
        """
        embedding = None
        if emb_hparams["file"] is not None and len(emb_hparams["file"]) > 0:
            embedding = Embedding(token_to_id_map, emb_hparams)
        return embedding

    @staticmethod
    def _make_other_transformations(other_trans_hparams, data_spec):
        """Creates a list of transformation functions based on the
        hyperparameters.

        Args:
            other_trans_hparams (list): A list of transformation functions,
                names, or full paths.
            data_spec: An instance of :class:`texar.data._DataSpec` to
                be passed to transformation functions.

        Returns:
            A list of transformation functions.
        """
        other_trans = []
        for tran in other_trans_hparams:
            if not is_callable(tran):
                tran = utils.get_function(tran, ["texar.custom"])
            other_trans.append(dsutils.make_partial(tran, data_spec))
        return other_trans

    @staticmethod
    def _make_length_filter(dataset_hparams, length_name, decoder):
        filter_mode = dataset_hparams["length_filter_mode"]
        max_length = dataset_hparams["max_seq_length"]
        filter_fn = None
        if filter_mode == _LengthFilterMode.DISCARD and max_length is not None:
            max_length += decoder.added_length
            filter_fn = dsutils._make_length_filter_fn(length_name,
                                                       max_length)
        return filter_fn

    def _process_dataset(self, dataset, hparams, data_spec):
        chained_tran, data_spec = self._make_processor(
            hparams["dataset"], data_spec,
            name_prefix=hparams["dataset"]["data_name"])
        num_parallel_calls = hparams["num_parallel_calls"]
        dataset = dataset.map(
            lambda *args: chained_tran(dsutils.maybe_tuple(args)),
            num_parallel_calls=num_parallel_calls)

        # Filters by length
        length_name = dsutils._connect_name(
            data_spec.name_prefix,
            data_spec.decoder.length_tensor_name)
        filter_fn = self._make_length_filter(
            hparams["dataset"], length_name, data_spec.decoder)
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        # Truncates data count
        dataset = dataset.take(hparams["max_dataset_size"])

        return dataset, data_spec

    def _make_bucket_length_fn(self):
        length_fn = self._hparams.bucket_length_fn
        if not length_fn:
            length_fn = lambda x: x[self.length_name]
        elif not is_callable(length_fn):
            # pylint: disable=redefined-variable-type
            length_fn = utils.get_function(length_fn, ["texar.custom"])
        return length_fn

    def _process(self, raw_example: str) -> List[str]:
        words = raw_example.split(self._delimiter)
        if (self._max_seq_length is not None and
                len(words) > self._max_seq_length):
            if self._length_filter_mode is _LengthFilterMode.TRUNC:
                words = words[:self._max_seq_length]
        if self._bos != '':
            words.insert(0, self._bos)
        if self._eos != '':
            words.append(self._eos)

        return words

    def _collate(self, examples: List[List[str]]) -> Batch:
        # For `MonoTextData`, each example is represented as a list of strings.
        # `collate_fn` takes care of padding and numericalization.
        lengths = torch.tensor([len(sent) for sent in examples],
                               dtype=torch.long)
        if self._should_pad:
            # Here in PyTorch we support `max_seq_length` of `None`,
            # which means to pad to the longest sentence in batch.
            pad_length = self._pad_length or lengths.max()
            examples = [
                sent + [''] * (pad_length - len(sent))
                if len(sent) < pad_length else sent
                for sent in examples
            ]
        # If `_should_pad` is `False`, sentences must be of the same length
        # or the following code will raise exceptions.
        ids = np.zeros((len(examples), len(examples[0])), dtype=np.long)
        for b_idx, sent in enumerate(examples):
            length = lengths[b_idx]
            ids[b_idx, :length] = \
                self._vocab.map_tokens_to_ids_py(sent[:length])
        return Batch(len(examples), text=examples,
                     text_ids=ids, length=lengths)

    def list_items(self) -> List[str]:
        r"""Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        items = ['text', 'text_ids', 'length']
        data_name = self._hparams['dataset']['data_name']
        if data_name is not None:
            items = [data_name + '_' + item for item in items]
        return items

    @property
    def vocab(self):
        r"""The vocabulary, an instance of :class:`~texar.data.Vocab`.
        """
        return self._vocab

    @property
    def embedding_init_value(self):
        r"""The `Tensor` containing the embedding value loaded from file.
        `None` if embedding is not specified.
        """
        if self._embedding is None:
            return None
        return self._embedding.word_vecs
