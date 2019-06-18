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
from typing import Optional, Tuple, List, Any, Dict, Callable, TypeVar, Union
import numpy as np

import torch

from texar.hyperparams import HParams
from texar.utils import utils
from texar.utils.dtypes import is_str, get_numpy_dtype
from texar.data.data.dataset_utils import Batch, padded_batch, connect_name
from texar.data.data.data_base import (DataSource,
                                       ZipDataSource, FilterDataSource)
from texar.data.data.text_data_base import TextDataBase, TextLineDataSource
from texar.data.data.record_data import PickleDataSource
from texar.data.data.mono_text_data import _default_mono_text_dataset_hparams
from texar.data.data.scalar_data import _default_scalar_dataset_hparams
from texar.data.data.record_data import (_default_record_dataset_hparams,
                                         _convert_feature_hparams,
                                         _create_image_transform)
from texar.data.data.mono_text_data import _LengthFilterMode
from texar.data.data_utils import count_file_lines
from texar.data.data import dataset_utils as dsutils
from texar.data.vocabulary import Vocab, SpecialTokens
from texar.data.embedding import Embedding

# pylint: disable=invalid-name, arguments-differ
# pylint: disable=protected-access, too-many-instance-attributes

__all__ = [
    "_default_dataset_hparams",
    "MultiAlignedData"
]


RawExample = TypeVar('RawExample')
Example = TypeVar('Example')


class _DataTypes(object):  # pylint: disable=no-init, too-few-public-methods
    """Enumeration of data types.
    """
    TEXT = "text"
    INT = "int"
    FLOAT = "float"
    RECORD = "record"


def _is_text_data(data_type):
    return data_type == _DataTypes.TEXT


def _is_scalar_data(data_type):
    return data_type == _DataTypes.INT or data_type == _DataTypes.FLOAT


def _is_tfrecord_data(data_type):
    return data_type == _DataTypes.RECORD


def _default_dataset_hparams(data_type=None):
    """Returns hyperparameters of a dataset with default values.

    See :meth:`texar.data.MultiAlignedData.default_hparams` for details.
    """
    if not data_type or _is_text_data(data_type):
        hparams = _default_mono_text_dataset_hparams()
        hparams.update({
            "data_type": _DataTypes.TEXT,
            "vocab_share_with": None,
            "embedding_init_share_with": None,
            "processing_share_with": None,
        })
    elif _is_scalar_data(data_type):
        hparams = _default_scalar_dataset_hparams()
    elif _is_tfrecord_data(data_type):
        hparams = _default_record_dataset_hparams()
        hparams.update({
            "data_type": _DataTypes.RECORD,
        })
    return hparams


class MultiAlignedData(
    TextDataBase[Tuple[Union[str, Dict[str, Any]], ...],
                 Tuple[Union[List[str], int, float, Dict[str, Any]], ...]]):
    """Data consisting of multiple aligned parts.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.

    The processor can read any number of parallel fields as specified in
    the "datasets" list of :attr:`hparams`, and result in a TF Dataset whose
    element is a python `dict` containing data fields from each of the
    specified datasets. Fields from a text dataset or TFRecord dataset have
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
            batch = iterator.get_next()

            iterator.switch_to_dataset(sess) # initializes the dataset
            batch_ = sess.run(batch)
            # batch_ == {
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
            batch = iterator.get_next()

            iterator.switch_to_dataset(sess) # initializes the dataset
            batch_ = sess.run(batch)
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
        self._other_transforms: List[
            List[Callable[[Union[List[str], int, RawExample]], Example]]] = []
        # Defaultizes hparams of each dataset
        datasets_hparams = self._hparams.datasets
        defaultized_datasets_hparams = []
        for ds_hpms in datasets_hparams:
            data_type = ds_hpms.get("data_type", None)
            defaultized_ds_hpms = HParams(ds_hpms,
                                          _default_dataset_hparams(data_type))
            defaultized_datasets_hparams.append(defaultized_ds_hpms)
            self._other_transforms.append(
                defaultized_ds_hpms.other_transformations)
        self._hparams.datasets = defaultized_datasets_hparams

        self._vocab = self.make_vocab(self._hparams.datasets)
        self._embedding = self.make_embedding(self._hparams.datasets,
                                              self._vocab)

        self._dataset_features: List[Optional[Dict[str, Any]]] = []
        self._name_prefix: List[str] = []
        datasources: List[DataSource] = []
        for _, hparams_i in enumerate(self._hparams.datasets):
            dtype = hparams_i.data_type
            datasource_i: DataSource
            if _is_text_data(dtype) or _is_scalar_data(dtype):
                datasource_i = TextLineDataSource(hparams_i.files,
                                             compression_type=
                                             hparams_i.compression_type)
                datasources.append(datasource_i)
                self._dataset_features.append(None)
            elif _is_tfrecord_data(dtype):
                datasource_i = PickleDataSource(file_paths=hparams_i.files)
                datasources.append(datasource_i)
                feature_types = hparams_i.feature_original_types
                feature_types = _convert_feature_hparams(feature_types)

                convert_types = hparams_i.feature_convert_types
                convert_types = {key: get_numpy_dtype(value)
                                       for key, value in convert_types.items()}
                for key, dtype in convert_types.items():
                    feature_types[key] = feature_types[key].\
                        _replace(dtype=dtype)
                image_options = hparams_i.image_options
                if isinstance(image_options, HParams):
                    image_options = [image_options]
                image_transforms = {}
                for options in image_options:
                    key = options.get('image_feature_name')
                    if key is None or key not in feature_types:
                        continue
                    image_transforms[key] = _create_image_transform(
                        options.get('resize_height'),
                        options.get('resize_width'),
                        options.get('resize_method') or 'bilinear')
                self._dataset_features.append({"feature_types": feature_types,
                                               "convert_types": convert_types,
                                               "image_transforms":
                                                   image_transforms})
            else:
                raise ValueError("Unknown data type: %s" % hparams_i.data_type)

            # check for duplicate names
            for i in range(1, len(self._name_prefix)):
                if self._name_prefix[i] in self._name_prefix[:i - 1]:
                    raise ValueError("Data name duplicated: %s"
                                     % self._name_prefix[i])

            self._name_prefix.append(hparams_i["data_name"])

        self._name_to_id = {v: k for k, v in enumerate(self._name_prefix)}

        datasource: DataSource
        datasource = ZipDataSource(*datasources)

        filters: List[Optional[Callable[[str], bool]]] = []
        for i, ds_hpms in enumerate(self._hparams.datasets):
            if _is_text_data(ds_hpms["data_type"]) and \
                ds_hpms["length_filter_mode"] is\
                    _LengthFilterMode.DISCARD.value and \
                    ds_hpms["max_seq_length"] is not None:
                delimiter = ds_hpms["delimiter"]
                max_seq_length = ds_hpms["max_seq_length"]
                filters.append(lambda x: len(x.split(delimiter)) <=
                               max_seq_length)
            else:
                filters.append(None)

        if any(filters):
            def filter_fn(data):
                return all([filters[idx](data_i) for
                            idx, data_i in enumerate(data) if filters[idx]
                            is not None])
            datasource = FilterDataSource(datasource, filter_fn=filter_fn)
        super().__init__(datasource, self._hparams, device)

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.

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
        dataset which can be text, scalar or TFRecord. The
        :attr:`"data_name"` field of each dataset is used as the name
        prefix of the data fields from the respective dataset. The
        :attr:`"data_name"` field of each dataset should not be the same.

            - For scalar dataset, the allowed hyperparameters and default \
            values are the same as the "dataset" field of \
            :meth:`texar.data.ScalarData.default_hparams`. Note that \
            :attr:`"data_type"` must be explicily specified \
            (either "int" or "float"). \

            - For TFRecord dataset, the allowed hyperparameters and default \
            values are the same as the "dataset" field of \
            :meth:`texar.data.TFRecordData.default_hparams`. Note that \
            :attr:`"data_type"` must be explicily specified \
            (tf_record"). \

            - For text dataset, the allowed hyperparameters and default values\
            are the same as the "dataset" filed of \
            :meth:`texar.data.MonoTextData.default_hparams`, with several \
            extra hyperparameters:

                "data_type" : str
                    The type of the dataset, one of {"text", "int", "float",
                    "tf_record"}. If set to "int" or "float", the dataset is
                    considered to be a scalar dataset. If set to "tf_record",
                    the dataset is considered to be a TFRecord dataset.
                    If not specified or set to "text", the dataset is
                    considered to be a text dataset.

                "vocab_share_with" : int, optional
                    Share the vocabulary of a preceding text dataset with the
                    specified index in the list (starting from 0). The
                    specified dataset must be a text dataset, and must have
                    an index smaller than the current dataset.

                    If specified, the vocab file of current dataset is ignored.
                    Default is `None` which disables the vocab sharing.

                "embedding_init_share_with": int, optional
                    Share the embedding initial value of a preceding text
                    dataset with the specified index in the list (starting
                    from 0).
                    The specified dataset must be a text dataset, and must have
                    an index smaller than the current dataset.

                    If specified, the :attr:`"embedding_init"` field of
                    the current dataset is ignored. Default is `None` which
                    disables the initial value sharing.

                "processing_share_with" : int, optional
                    Share the processing configurations of a preceding text
                    dataset with the specified index in the list (starting
                    from 0).
                    The specified dataset must be a text dataset, and must have
                    an index smaller than the current dataset.

                    If specified, relevant field of the current dataset are
                    ignored, including "delimiter", "bos_token", "eos_token",
                    and "other_transformations". Default is `None` which
                    disables the processing sharing.

        2. For the **general** hyperparameters, see
        :meth:`texar.data.DataBase.default_hparams` for details.
        """
        hparams = TextDataBase.default_hparams()
        hparams["name"] = "multi_aligned_data"
        hparams["datasets"] = []
        return hparams

    @staticmethod
    def _raise_sharing_error(err_data, shr_data, hparam_name):
        raise ValueError(
            "Must only share specifications with a preceding dataset. "
            "Dataset %d has '%s=%d'" % (err_data, hparam_name, shr_data))

    @staticmethod
    def make_vocab(hparams):
        """Makes a list of vocabs based on the hparams.

        Args:
            hparams (list): A list of dataset hyperparameters.

        Returns:
            A list of :class:`texar.data.Vocab` instances. Some instances
            may be the same objects if they are set to be shared and have
            the same other configs.
        """
        if not isinstance(hparams, (list, tuple)):
            hparams = [hparams]

        vocabs = []
        for i, hparams_i in enumerate(hparams):
            if not _is_text_data(hparams_i["data_type"]):
                vocabs.append(None)
                continue

            proc_shr = hparams_i["processing_share_with"]
            if proc_shr is not None:
                bos_token = hparams[proc_shr]["bos_token"]
                eos_token = hparams[proc_shr]["eos_token"]
            else:
                bos_token = hparams_i["bos_token"]
                eos_token = hparams_i["eos_token"]
            bos_token = utils.default_str(
                bos_token, SpecialTokens.BOS)
            eos_token = utils.default_str(
                eos_token, SpecialTokens.EOS)

            vocab_shr = hparams_i["vocab_share_with"]
            if vocab_shr is not None:
                if vocab_shr >= i:
                    MultiAlignedData._raise_sharing_error(
                        i, vocab_shr, "vocab_share_with")
                if not vocabs[vocab_shr]:
                    raise ValueError("Cannot share vocab with dataset %d which "
                                     "does not have a vocab." % vocab_shr)
                if bos_token == vocabs[vocab_shr].bos_token and \
                        eos_token == vocabs[vocab_shr].eos_token:
                    vocab = vocabs[vocab_shr]
                else:
                    vocab = Vocab(hparams[vocab_shr]["vocab_file"],
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
        """Optionally loads embeddings from files (if provided), and
        returns respective :class:`texar.data.Embedding` instances.
        """
        if not isinstance(hparams, (list, tuple)):
            hparams = [hparams]

        embs = []
        for i, hparams_i in enumerate(hparams):
            if not _is_text_data(hparams_i["data_type"]):
                embs.append(None)
                continue

            emb_shr = hparams_i["embedding_init_share_with"]
            if emb_shr is not None:
                if emb_shr >= i:
                    MultiAlignedData._raise_sharing_error(
                        i, emb_shr, "embedding_init_share_with")
                if not embs[emb_shr]:
                    raise ValueError("Cannot share embedding with dataset %d "
                                     "which does not have an embedding." %
                                     emb_shr)
                if emb_shr != hparams_i["vocab_share_with"]:
                    raise ValueError("'embedding_init_share_with' != "
                                     "vocab_share_with. embedding_init can "
                                     "be shared only when vocab is shared.")
                emb = embs[emb_shr]
            else:
                emb = None
                emb_file = hparams_i["embedding_init"]["file"]
                if emb_file and emb_file != "":
                    emb = Embedding(vocabs[i].token_to_id_map_py,
                                    hparams_i["embedding_init"])
            embs.append(emb)

        return embs

    def _process(self, raw_example: Tuple[Union[str, Dict[str, Any]], ...]) \
            -> Tuple[Union[List[str], int, float, Dict[str, Any]], ...]:
        dataset_hparams = self.hparams.datasets
        processed_examples = []
        for i, raw_example_i in enumerate(raw_example):
            if _is_text_data(dataset_hparams[i]["data_type"]):
                delimiter = dataset_hparams[i]["delimiter"]
                raw_tokens = raw_example_i.split(delimiter)  # type: ignore
                max_seq_length = dataset_hparams[i]["max_seq_length"]
                if (max_seq_length is not None and
                        len(raw_tokens) > max_seq_length):
                    if dataset_hparams[i]["length_filter_mode"] is \
                            _LengthFilterMode.TRUNC.value:
                        raw_tokens = raw_tokens[:max_seq_length]

                if (dataset_hparams[i].bos_token is not None and
                        dataset_hparams[i].bos_token != ''):
                    raw_tokens.insert(0, dataset_hparams[i].bos_token)
                if (dataset_hparams[i].eos_token is not None and
                        dataset_hparams[i].eos_token != ''):
                    raw_tokens.append(dataset_hparams[i].eos_token)
                transforms = self._other_transforms[i]
                if len(transforms) > 0:
                    for transform in transforms:
                        raw_tokens = transform(raw_tokens)
                processed_examples.append(raw_tokens)
            elif _is_scalar_data(dataset_hparams[i]["data_type"]):
                scalar_ex: Union[str, int, float] = str(raw_example_i)
                data_type = dataset_hparams[i]["data_type"]
                if data_type == "int":
                    scalar_ex = int(scalar_ex)
                elif data_type == "float":
                    scalar_ex = float(scalar_ex)
                transforms = self._other_transforms[i]
                if len(transforms) > 0:
                    for transform in transforms:
                        scalar_ex = transform(scalar_ex)
                processed_examples.append(scalar_ex)
            elif _is_tfrecord_data(dataset_hparams[i]["data_type"]):
                example: Dict[str, Any] = raw_example_i  # type: ignore
                # for some reasons even after the not None check here, mypy
                # thinks self._dataset_features[i] may be None, hence adding
                # ignore to the lines below
                if self._dataset_features[i] is not None:
                    for key, dtype in \
                            self._dataset_features[i]["convert_types"].items():  # type: ignore # noqa
                        example[key] = np.asarray(example[key], dtype=dtype)
                    for key, transform in \
                            self._dataset_features[i]["image_transforms"].items():  # type: ignore # noqa
                        example[key] = transform(example[key])
                for transform in self._other_transforms[i]:
                    example = transform(example)
                processed_examples.append(example)
        return tuple(processed_examples)

    def _collate(self, examples) -> Batch:
        dataset_hparams = self.hparams["datasets"]
        transposed_examples: List[Tuple] = list(map(tuple, zip(*examples)))
        batch: Dict[str, Any] = {}
        for i, transposed_example in enumerate(transposed_examples):
            if _is_text_data(dataset_hparams[i]["data_type"]):
                # Treat it as MonoTextData and collate
                pad_length = dataset_hparams[i]["max_seq_length"]
                if pad_length is not None:
                    pad_length += sum(x is not None and x != '' for x in
                                      [self.vocab(i).bos_token,
                                       self.vocab(i).eos_token])
                text_ids = [self.vocab(i).map_tokens_to_ids_py(sent) for sent in
                            transposed_example]
                text_ids, lengths = padded_batch(text_ids, pad_length,
                                                 pad_value=
                                                 self.vocab(i).pad_token_id)
                # also pad the examples
                pad_length = pad_length or max(lengths)
                transposed_example = tuple(
                    sent + [''] * (pad_length - len(sent))
                    if len(sent) < pad_length else sent
                    for sent in transposed_example
                )

                text_ids = torch.from_numpy(text_ids).to(device=self.device)
                lengths = torch.tensor(lengths, dtype=torch.long,
                                       device=self.device)

                batch.update(
                    {connect_name(self._name_prefix[i], "text"):
                     transposed_example,
                     connect_name(self._name_prefix[i], "text_ids"):
                     text_ids,
                     connect_name(self._name_prefix[i], "length"):
                     lengths})
            elif _is_scalar_data(dataset_hparams[i]["data_type"]):
                # convert the list of strings into appropriate tensors here
                data_type = dataset_hparams[i]["data_type"]
                if data_type == "int":
                    example_np = np.array(transposed_example, dtype=np.int32)
                    example = torch.from_numpy(example_np). \
                        to(device=self.device)
                elif data_type == "float":
                    example_np = np.array(transposed_example, dtype=np.float32)
                    example = torch.from_numpy(example_np). \
                        to(device=self.device)
                else:
                    raise ValueError(
                        "Incorrect \'data_type\'. Currently \'int\' and "
                        "\'float\' are supported. Received {}"
                        .format(data_type))
                batch.update({self._name_prefix[i]: example})
            elif _is_tfrecord_data(dataset_hparams[i]["data_type"]):
                for key, descriptor in \
                        self._dataset_features[i]["feature_types"].items():  # type: ignore # noqa
                    values = [ex[key] for ex in transposed_example]
                    if descriptor.shape is not None:
                        # FixedLenFeature, do not pad.
                        # NumPy functions work on PyTorch tensors too.
                        if len(descriptor.shape) > 0 \
                                and descriptor.shape[0] is None:
                            values, _ = padded_batch(values)
                        else:
                            values = np.stack(values, axis=0)
                        if (not torch.is_tensor(values) and
                                descriptor.dtype not in [np.str_, np.bytes_]):
                            values = torch.from_numpy(values)
                    else:
                        # VarLenFeature, just put everything in a Python list.
                        pass
                    batch.update({connect_name(self._name_prefix[i], key):
                                  values})
        return Batch(len(examples),
                     **batch)

    @staticmethod
    def _get_name_prefix(dataset_hparams):
        name_prefix = [hpms["data_name"] for hpms in dataset_hparams]
        for i in range(1, len(name_prefix)):
            if name_prefix[i] in name_prefix[:i - 1]:
                raise ValueError("Data name duplicated: %s" % name_prefix[i])
        return name_prefix

    def list_items(self):
        """Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        dataset_hparams = self.hparams.datasets
        items = []
        for i, ds_hpms in enumerate(dataset_hparams):
            if _is_text_data(ds_hpms["data_type"]):
                items.extend([connect_name(self._name_prefix[i], "text"),
                              connect_name(self._name_prefix[i], "text_ids"),
                              connect_name(self._name_prefix[i], "length")])
            elif _is_scalar_data(ds_hpms["data_type"]):
                items.append(self._name_prefix[i])
            elif _is_tfrecord_data(ds_hpms["data_type"]):
                for feature_name in \
                        self._dataset_features[i]["feature_types"].keys():
                    items.append(connect_name(self._name_prefix[i],
                                 feature_name))
        return items

    @property
    def dataset(self):
        """The dataset.
        """
        return self._source

    def dataset_size(self):
        """Returns the number of data instances in the dataset.

        Note that this is the total data count in the raw files, before any
        filtering and truncation.
        """
        if not self._dataset_size:
            # pylint: disable=attribute-defined-outside-init
            self._dataset_size = count_file_lines(
                self._hparams.datasets[0].files)
        return self._dataset_size

    def _maybe_name_to_id(self, name_or_id):
        if is_str(name_or_id):
            if name_or_id not in self._name_to_id:
                raise ValueError("Unknown data name: {}".format(name_or_id))
            return self._name_to_id[name_or_id]
        return name_or_id

    def vocab(self, name_or_id):
        """Returns the :class:`~texar.data.Vocab` of text dataset by its name
        or id. `None` if the dataset is not of text type.

        Args:
            name_or_id (str or int): Data name or the index of text dataset.
        """
        i = self._maybe_name_to_id(name_or_id)
        return self._vocab[i]

    def embedding_init_value(self, name_or_id):
        """Returns the `Tensor` of embedding init value of the
        dataset by its name or id. `None` if the dataset is not of text type.
        """
        i = self._maybe_name_to_id(name_or_id)
        return self._embedding[i]

    def text_name(self, name_or_id):
        """The name of text tensor of text dataset by its name or id. If the
        dataset is not of text type, returns `None`.
        """
        i = self._maybe_name_to_id(name_or_id)
        if not _is_text_data(self._hparams.datasets[i]["data_type"]):
            return None
        name = dsutils.connect_name(self._name_prefix[i], "text")
        return name

    def length_name(self, name_or_id):
        """The name of length tensor of text dataset by its name or id. If the
        dataset is not of text type, returns `None`.
        """
        i = self._maybe_name_to_id(name_or_id)
        if not _is_text_data(self._hparams.datasets[i]["data_type"]):
            return None
        name = dsutils.connect_name(self._name_prefix[i], "length")
        return name

    def text_id_name(self, name_or_id):
        """The name of length tensor of text dataset by its name or id. If the
        dataset is not of text type, returns `None`.
        """
        i = self._maybe_name_to_id(name_or_id)
        if not _is_text_data(self._hparams.datasets[i]["data_type"]):
            return None
        name = dsutils.connect_name(
            self._name_prefix[i], "text_ids")
        return name

    def utterance_cnt_name(self, name_or_id):
        """The name of utterance count tensor of text dataset by its name or id.
        If the dataset is not variable utterance text data, returns `None`.
        """
        i = self._maybe_name_to_id(name_or_id)
        if not _is_text_data(self._hparams.datasets[i]["data_type"]) or \
                not self._hparams.datasets[i]["variable_utterance"]:
            return None
        name = dsutils.connect_name(
            self._name_prefix[i], "utterance_cnt")
        return name

    # @property
    def data_name(self, name_or_id):
        """The name of the data tensor of scalar dataset by its name or id..
        If the dataset is not a scalar data, returns `None`.
        """
        i = self._maybe_name_to_id(name_or_id)
        if not _is_scalar_data(self._hparams.datasets[i]["data_type"]):
            return None
        return self._name_prefix[i]
