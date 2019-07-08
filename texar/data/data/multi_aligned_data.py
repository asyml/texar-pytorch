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
Data consisting of multiple aligned parts.
"""
from typing import (Optional, Tuple, List, Any, Dict, Callable, Union)

import torch

from texar.hyperparams import HParams
from texar.utils import utils
from texar.utils.dtypes import is_str
from texar.data.data.dataset_utils import Batch, connect_name
from texar.data.data.data_base import (DataSource, DataBase,
                                       ZipDataSource, FilterDataSource)
from texar.data.data.text_data_base import TextDataBase, TextLineDataSource
from texar.data.data.record_data import PickleDataSource
from texar.data.data.mono_text_data import _default_mono_text_dataset_hparams
from texar.data.data.scalar_data import (ScalarData,
                                         _default_scalar_dataset_hparams)
from texar.data.data.record_data import (RecordData,
                                         _default_record_dataset_hparams)
from texar.data.data.mono_text_data import MonoTextData, _LengthFilterMode
from texar.data.vocabulary import Vocab, SpecialTokens
from texar.data.embedding import Embedding

# pylint: disable=invalid-name, arguments-differ
# pylint: disable=protected-access, too-many-instance-attributes

__all__ = [
    "_default_dataset_hparams",
    "MultiAlignedData"
]


class _DataTypes:  # pylint: disable=no-init, too-few-public-methods
    r"""Enumeration of data types.
    """
    TEXT = "text"
    INT = "int"
    FLOAT = "float"
    RECORD = "record"


def _is_text_data(data_type):
    return data_type == _DataTypes.TEXT


def _is_scalar_data(data_type):
    return data_type in [_DataTypes.INT, _DataTypes.FLOAT]


def _is_record_data(data_type):
    return data_type == _DataTypes.RECORD


def _default_dataset_hparams(data_type=None):
    r"""Returns hyperparameters of a dataset with default values.

    See :meth:`texar.data.MultiAlignedData.default_hparams` for details.
    """
    if data_type is None or _is_text_data(data_type):
        hparams = _default_mono_text_dataset_hparams()
        hparams.update({
            "data_type": _DataTypes.TEXT,
            "vocab_share_with": None,
            "embedding_init_share_with": None,
            "processing_share_with": None,
        })
    elif _is_scalar_data(data_type):
        hparams = _default_scalar_dataset_hparams()
    elif _is_record_data(data_type):
        hparams = _default_record_dataset_hparams()
        hparams.update({
            "data_type": _DataTypes.RECORD,
        })
    return hparams


class MultiAlignedData(
    TextDataBase[Tuple[Union[str, Dict[str, Any]], ...],
                 Tuple[Union[List[str], int, float, Dict[str, Any]], ...]]):
    r"""Data consisting of multiple aligned parts.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.
        device: The device of the produced batches. For GPU training, set to
            current CUDA device.

    The processor can read any number of parallel fields as specified in
    the "datasets" list of :attr:`hparams`, and result in a Dataset whose
    element is a python `dict` containing data fields from each of the
    specified datasets. Fields from a text dataset or Record dataset have
    names prefixed by its "data_name". Fields from a scalar dataset are
    specified by its "data_name".

    Example:

        .. code-block:: python

            hparams={
                'datasets': [
                    {'files': 'a.txt', 'vocab_file': 'v.a', 'data_name': 'x'},
                    {'files': 'b.txt', 'vocab_file': 'v.b', 'data_name': 'y'},
                    {'files': 'c.txt', 'data_type': 'int', 'data_name': 'z'}
                ]
                'batch_size': 1
            }
            data = MultiAlignedData(hparams)
            iterator = DataIterator(data)

            for batch in iterator:
                # batch contains the following
                # batch == {
                #    'x_text': [['<BOS>', 'x', 'sequence', '<EOS>']],
                #    'x_text_ids': [['1', '5', '10', '2']],
                #    'x_length': [4]
                #    'y_text': [['<BOS>', 'y', 'sequence', '1', '<EOS>']],
                #    'y_text_ids': [['1', '6', '10', '20', '2']],
                #    'y_length': [5],
                #    'z': [1000],
                # }

            ...

            hparams={
                'datasets': [
                    {'files': 'd.txt', 'vocab_file': 'v.d', 'data_name': 'm'},
                    {
                        'files': 'd.tfrecord',
                        'data_type': 'tf_record',
                        "feature_original_types": {
                            'image': ['tf.string', 'FixedLenFeature']
                        },
                        'image_options': {
                            'image_feature_name': 'image',
                            'resize_height': 512,
                            'resize_width': 512,
                        },
                        'data_name': 't',
                    }
                ]
                'batch_size': 1
            }
            data = MultiAlignedData(hparams)
            iterator = DataIterator(data)
            for batch in iterator:
                # batch contains the following
                # batch_ == {
                #    'x_text': [['<BOS>', 'NewYork', 'City', 'Map', '<EOS>']],
                #    'x_text_ids': [['1', '100', '80', '65', '2']],
                #    'x_length': [5],
                #
                #    # "t_image" is a list of a "numpy.ndarray" image
                #    # in this example. Its width equals to 512 and
                #    # its height equals to 512.
                #    't_image': [...]
                # }

    """

    def __init__(self, hparams, device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        # Defaultizes hyperparameters of each dataset
        datasets_hparams = self._hparams.datasets
        defaultized_datasets_hparams = []
        for hparams_i in datasets_hparams:
            data_type = hparams_i.get("data_type", None)
            defaultized_ds_hpms = HParams(hparams_i,
                                          _default_dataset_hparams(data_type))
            defaultized_datasets_hparams.append(defaultized_ds_hpms)
        self._hparams.datasets = defaultized_datasets_hparams

        self._vocab = self.make_vocab(self._hparams.datasets)
        self._embedding = self.make_embedding(self._hparams.datasets,
                                              self._vocab)

        name_prefix: List[str] = []
        self._names: List[Dict[str, Any]] = []
        datasources: List[DataSource] = []
        filters: List[Optional[Callable[[str], bool]]] = []
        self._databases: List[DataBase] = []
        for idx, hparams_i in enumerate(self._hparams.datasets):
            dtype = hparams_i.data_type
            datasource_i: DataSource

            if _is_text_data(dtype):
                datasource_i = TextLineDataSource(
                    hparams_i.files,
                    compression_type=hparams_i.compression_type)
                datasources.append(datasource_i)
                if (hparams_i["length_filter_mode"] ==
                    _LengthFilterMode.DISCARD.value) and \
                        (hparams_i["max_seq_length"] is not None):

                    def _get_filter(delimiter, max_seq_length):
                        return lambda x: len(x.split(delimiter)) <= \
                               max_seq_length

                    filters.append(_get_filter(hparams_i["delimiter"],
                                               hparams_i["max_seq_length"]))
                else:
                    filters.append(None)

                self._names.append(
                    {"text": connect_name(hparams_i["data_name"], "text"),
                     "text_ids": connect_name(hparams_i["data_name"],
                                              "text_ids"),
                     "length": connect_name(hparams_i["data_name"],
                                            "length")})

                text_hparams = MonoTextData.default_hparams()
                for key in text_hparams["dataset"].keys():
                    if key in hparams_i:
                        text_hparams["dataset"][key] = hparams_i[key]
                # handle prepend logic in MultiAlignedData collate function
                text_hparams["dataset"]["data_name"] = None

                self._databases.append(
                    MonoTextData._construct(hparams=text_hparams,
                                            device=device,
                                            vocab=self._vocab[idx],
                                            embedding=self._embedding[idx]))
            elif _is_scalar_data(dtype):
                datasource_i = TextLineDataSource(
                    hparams_i.files,
                    compression_type=hparams_i.compression_type)
                datasources.append(datasource_i)
                filters.append(None)
                self._names.append({"label": hparams_i["data_name"]})
                scalar_hparams = ScalarData.default_hparams()
                scalar_hparams.update({"dataset": hparams_i.todict()})
                self._databases.append(ScalarData._construct(
                    hparams=scalar_hparams, device=device))
            elif _is_record_data(dtype):
                datasource_i = PickleDataSource(file_paths=hparams_i.files)
                datasources.append(datasource_i)
                self._names.append(
                    {feature_type: connect_name(hparams_i["data_name"],
                                                feature_type)
                     for feature_type in
                     hparams_i.feature_original_types.keys()})
                filters.append(None)
                record_hparams = RecordData.default_hparams()
                for key in record_hparams["dataset"].keys():
                    if key in hparams_i:
                        record_hparams["dataset"][key] = hparams_i[key]
                self._databases.append(RecordData._construct(
                    hparams=record_hparams))
            else:
                raise ValueError("Unknown data type: %s" % hparams_i.data_type)

            # check for duplicate names
            for i in range(1, len(name_prefix)):
                if name_prefix[i] in name_prefix[:i - 1]:
                    raise ValueError("Data name duplicated: %s"
                                     % name_prefix[i])

            name_prefix.append(hparams_i["data_name"])

        self._name_to_id = {v: k for k, v in enumerate(name_prefix)}

        datasource: DataSource
        datasource = ZipDataSource(*datasources)

        if any(filters):
            def filter_fn(data):
                return all(filters[idx](data_i) for idx, data_i in
                            enumerate(data) if filters[idx] is not None)
            datasource = FilterDataSource(datasource, filter_fn=filter_fn)
        super().__init__(datasource, self._hparams, device)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters:

        .. code-block:: python

            {
                # (1) Hyperparams specific to text dataset
                "datasets": []
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
                "name": "multi_aligned_data",
            }

        Here:

        1. "datasets" is a list of `dict` each of which specifies a
           dataset which can be text, scalar or Record. The :attr:`"data_name"`
           field of each dataset is used as the name prefix of the data fields
           from the respective dataset. The :attr:`"data_name"` field of each
           dataset should not be the same.

           i) For scalar dataset, the allowed hyperparameters and default
              values are the same as the "dataset" field of
              :meth:`texar.data.ScalarData.default_hparams`. Note that
              :attr:`"data_type"` must be explicitly specified
              (either "int" or "float").

           ii) For Record dataset, the allowed hyperparameters and default
               values are the same as the "dataset" field of
               :meth:`texar.data.RecordData.default_hparams`. Note that
               :attr:`"data_type"` must be explicitly specified ("record").

           iii) For text dataset, the allowed hyperparameters and default
                values are the same as the "dataset" filed of
                :meth:`texar.data.MonoTextData.default_hparams`, with several
                extra hyperparameters:

                `"data_type"`: str
                    The type of the dataset, one of {"text", "int", "float",
                    "record"}. If set to "int" or "float", the dataset is
                    considered to be a scalar dataset. If set to
                    "record", the dataset is considered to be a Record
                    dataset.

                    If not specified or set to "text", the dataset is
                    considered to be a text dataset.

                `"vocab_share_with"`: int, optional
                    Share the vocabulary of a preceding text dataset with
                    the specified index in the list (starting from 0). The
                    specified dataset must be a text dataset, and must have
                    an index smaller than the current dataset.

                    If specified, the vocab file of current dataset is
                    ignored. Default is `None` which disables the vocab
                    sharing.

                `"embedding_init_share_with"`: int, optional
                    Share the embedding initial value of a preceding text
                    dataset with the specified index in the list (starting
                    from 0). The specified dataset must be a text dataset,
                    and must have an index smaller than the current dataset.

                    If specified, the :attr:`"embedding_init"` field of the
                    current dataset is ignored. Default is `None` which
                    disables the initial value sharing.

                `"processing_share_with"`: int, optional
                    Share the processing configurations of a preceding text
                    dataset with the specified index in the list (starting
                    from 0). The specified dataset must be a text dataset,
                    and must have an index smaller than the current dataset.

                    If specified, relevant field of the current dataset are
                    ignored, including `delimiter`, `bos_token`,
                    `eos_token`, and "other_transformations". Default is
                    `None` which disables the processing sharing.

        2. For the **general** hyperparameters, see
        :meth:`texar.data.DataBase.default_hparams` for details.

        """
        hparams = TextDataBase.default_hparams()
        hparams["name"] = "multi_aligned_data"
        hparams["datasets"] = []
        return hparams

    def to(self, device: torch.device):
        for dataset in self._databases:
            dataset.to(device)
        return super().to(device)

    @staticmethod
    def _raise_sharing_error(err_data, share_data, hparam_name):
        raise ValueError(
            "Must only share specifications with a preceding dataset. "
            "Dataset %d has '%s=%d'" % (err_data, hparam_name, share_data))

    @staticmethod
    def make_vocab(hparams):
        r"""Makes a list of vocabs based on the hyperparameters.

        Args:
            hparams (list): A list of dataset hyperparameters.

        Returns:
            A list of :class:`texar.data.Vocab` instances. Some instances
            may be the same objects if they are set to be shared and have
            the same other configurations.
        """
        if not isinstance(hparams, (list, tuple)):
            hparams = [hparams]

        vocabs = []
        for i, hparams_i in enumerate(hparams):
            if not _is_text_data(hparams_i["data_type"]):
                vocabs.append(None)
                continue

            proc_share = hparams_i["processing_share_with"]
            if proc_share is not None:
                bos_token = hparams[proc_share]["bos_token"]
                eos_token = hparams[proc_share]["eos_token"]
            else:
                bos_token = hparams_i["bos_token"]
                eos_token = hparams_i["eos_token"]
            bos_token = utils.default_str(
                bos_token, SpecialTokens.BOS)
            eos_token = utils.default_str(
                eos_token, SpecialTokens.EOS)

            vocab_share = hparams_i["vocab_share_with"]
            if vocab_share is not None:
                if vocab_share >= i:
                    MultiAlignedData._raise_sharing_error(
                        i, vocab_share, "vocab_share_with")
                if vocabs[vocab_share] is None:
                    raise ValueError("Cannot share vocab with dataset %d which "
                                     "does not have a vocab." % vocab_share)
                if bos_token == vocabs[vocab_share].bos_token and \
                        eos_token == vocabs[vocab_share].eos_token:
                    vocab = vocabs[vocab_share]
                else:
                    vocab = Vocab(hparams[vocab_share]["vocab_file"],
                                  bos_token=bos_token,
                                  eos_token=eos_token)
            else:
                vocab = Vocab(hparams_i["vocab_file"],
                              bos_token=bos_token,
                              eos_token=eos_token)
            vocabs.append(vocab)

        return vocabs

    @staticmethod
    def make_embedding(hparams, vocabs):
        r"""Optionally loads embeddings from files (if provided), and
        returns respective :class:`texar.data.Embedding` instances.
        """
        if not isinstance(hparams, (list, tuple)):
            hparams = [hparams]

        embs = []
        for i, hparams_i in enumerate(hparams):
            if not _is_text_data(hparams_i["data_type"]):
                embs.append(None)
                continue

            emb_share = hparams_i["embedding_init_share_with"]
            if emb_share is not None:
                if emb_share >= i:
                    MultiAlignedData._raise_sharing_error(
                        i, emb_share, "embedding_init_share_with")
                if not embs[emb_share]:
                    raise ValueError("Cannot share embedding with dataset %d "
                                     "which does not have an embedding." %
                                     emb_share)
                if emb_share != hparams_i["vocab_share_with"]:
                    raise ValueError("'embedding_init_share_with' != "
                                     "vocab_share_with. embedding_init can "
                                     "be shared only when vocab is shared.")
                emb = embs[emb_share]
            else:
                emb = None
                emb_file = hparams_i["embedding_init"]["file"]
                if emb_file and emb_file != "":
                    emb = Embedding(vocabs[i].token_to_id_map_py,
                                    hparams_i["embedding_init"])
            embs.append(emb)

        return embs

    def process(self, raw_example: Tuple[Union[str, Dict[str, Any]], ...]) \
            -> Tuple[Union[List[str], int, float, Dict[str, Any]], ...]:
        processed_examples = []
        for i, raw_example_i in enumerate(raw_example):
            example_i = self._databases[i].process(raw_example_i)
            processed_examples.append(example_i)
        return tuple(processed_examples)

    def collate(self, examples) -> Batch:
        transposed_examples = map(list, zip(*examples))
        batch: Dict[str, Any] = {}
        for i, transposed_example in enumerate(transposed_examples):
            kth_batch = self._databases[i].collate(transposed_example)
            for key, name in self._names[i].items():
                batch.update({name: kth_batch[key]})
        return Batch(len(examples), batch=batch)

    def list_items(self):
        r"""Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        return [value for name in self._names for value in name.values()]

    def _maybe_name_to_id(self, name_or_id):
        if is_str(name_or_id):
            if name_or_id not in self._name_to_id:
                raise ValueError("Unknown data name: {}".format(name_or_id))
            return self._name_to_id[name_or_id]
        return name_or_id

    def vocab(self, name_or_id):
        r"""Returns the :class:`~texar.data.Vocab` of text dataset by its name
        or id. `None` if the dataset is not of text type.

        Args:
            name_or_id (str or int): Data name or the index of text dataset.
        """
        i = self._maybe_name_to_id(name_or_id)
        return self._vocab[i]

    def embedding_init_value(self, name_or_id):
        r"""Returns the `Tensor` of embedding initial value of the
        dataset by its name or id. `None` if the dataset is not of text type.
        """
        i = self._maybe_name_to_id(name_or_id)
        return self._embedding[i]

    def text_name(self, name_or_id):
        r"""The name of text tensor of text dataset by its name or id. If the
        dataset is not of text type, returns `None`.
        """
        i = self._maybe_name_to_id(name_or_id)
        if not _is_text_data(self._hparams.datasets[i]["data_type"]):
            return None
        return self._names[i]["text"]

    def length_name(self, name_or_id):
        r"""The name of length tensor of text dataset by its name or id. If the
        dataset is not of text type, returns `None`.
        """
        i = self._maybe_name_to_id(name_or_id)
        if not _is_text_data(self._hparams.datasets[i]["data_type"]):
            return None
        return self._names[i]["length"]

    def text_id_name(self, name_or_id):
        r"""The name of length tensor of text dataset by its name or id. If the
        dataset is not of text type, returns `None`.
        """
        i = self._maybe_name_to_id(name_or_id)
        if not _is_text_data(self._hparams.datasets[i]["data_type"]):
            return None
        return self._names[i]["text_ids"]

    def data_name(self, name_or_id):
        r"""The name of the data tensor of scalar dataset by its name or id..
        If the dataset is not a scalar data, returns `None`.
        """
        i = self._maybe_name_to_id(name_or_id)
        if not _is_scalar_data(self._hparams.datasets[i]["data_type"]):
            return None
        return self._names[i]["label"]
