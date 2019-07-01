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
Data class that supports reading TFRecord data and data type converting.
"""
import io
import pickle
from typing import (
    Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union)

import numpy as np
import torch

from texar.data.data.data_base import DataBase, DataSource, SequenceDataSource
from texar.data.data.dataset_utils import Batch, padded_batch, connect_name
from texar.hyperparams import HParams
from texar.utils.dtypes import get_numpy_dtype
from texar.utils.types import MaybeList

# pylint: disable=protected-access

__all__ = [
    "_default_record_dataset_hparams",
    "PickleDataSource",
    "RecordData",
]


def _default_record_dataset_hparams():
    r"""Returns hyperparameters of a TFRecord dataset with default values.

    See :meth:`texar.data.TFRecordData.default_hparams` for details.
    """
    return {
        "files": [],
        "feature_original_types": {},
        "feature_convert_types": {},
        "image_options": {},
        "compression_type": None,
        "other_transformations": [],
        "num_shards": None,
        "shard_id": None,
        "data_name": None,
        "@no_typecheck": [
            "files",
            "feature_original_types",
            "feature_convert_types",
            "image_options"
        ],
    }


RawExample = TypeVar('RawExample')


class PickleDataSource(DataSource[RawExample]):
    r"""Data source for reading from (multiple) pickled binary files. Each file
    could contain multiple pickled objects, and each object is yielded as an
    example.

    This data source does not support indexing.

    Args:
        file_paths (str or list[str]): Paths to pickled binary files.
        lists_are_examples (bool): If `True`, lists will be treated as
            a single example; if `False`, each element in the list will be
            treated as separate examples. Default is `True`. Set this to
            `False` if the entire pickled binary file is a list.

            .. note::
                It is recommended against storing all examples as a list,
                because in this case, all examples can only be accessed
                after the whole list is parsed.

        pickle_kwargs: Additional keyword arguments to pass to
            :meth:`pickle.load`.
    """

    def __init__(self, file_paths: MaybeList[str],
                 lists_are_examples: bool = True, **pickle_kwargs):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self._file_paths = file_paths
        self._lists_are_examples = lists_are_examples
        self._pickle_kwargs = pickle_kwargs

    def __iter__(self):
        for path in self._file_paths:
            with open(path, 'rb') as f:
                if self._lists_are_examples:
                    while True:
                        try:
                            ex = pickle.load(f, **self._pickle_kwargs)
                            if isinstance(ex, list):
                                yield from ex
                            else:
                                yield ex
                        except EOFError:
                            break
                else:
                    while True:
                        try:
                            yield pickle.load(f, **self._pickle_kwargs)
                        except EOFError:
                            break


TransformFn = Callable[[bytes], torch.ByteTensor]


def _create_image_transform(height: Optional[int], width: Optional[int],
                            resize_method: Union[str, int] = 'bilinear') \
        -> TransformFn:
    r"""Create a function based on `Pillow image transforms
    <https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize>`
    that performs resizing with desired resize method (interpolation).

    Args:
        height (int, optional): Height of the transformed image. Set to `None`
            to not perform resizing.
        width (int, optional): Width of the transformed image. Set to `None`
            to not perform resizing.
        resize_method (str or int, optional): Interpolation method to use.
            Supported values are ``"nearest"`` (nearest neighbor),
            ``"bilinear"``, ``"bicubic"``, and ``"lanczos"``. Enum values from
            PIL (e.g., ``PIL.Image.BILINEAR``) are also supported. Defaults to
            ``"bilinear"``.

    Returns:
        The created transformation function.
    """
    try:
        import PIL.Image
    except ImportError:
        raise ImportError(
            "To use image resizing with RecordData, the Pillow library must be "
            "installed. Please see "
            "https://pillow.readthedocs.io/en/stable/installation.html.")

    # We take the final part of a possibly dot-separated string for
    # compatibility reasons, because in texar-TF `resize_method` could take the
    # form of "tf.image.ResizeMethod.BILINEAR".
    if isinstance(resize_method, int):
        interpolation = resize_method
    else:
        method = resize_method.lower().split('.')[-1]
        if method in ["nearest_neighbor", "nearest"]:
            interpolation = PIL.Image.NEAREST
        elif method == "bilinear":
            interpolation = PIL.Image.BILINEAR
        elif method == "bicubic":
            interpolation = PIL.Image.BICUBIC
        elif method == "lanczos":
            interpolation = PIL.Image.LANCZOS
        else:
            raise ValueError(f"Unsupported resize method '{resize_method}'")
    if height is None or width is None:
        size = None
    else:
        size = (height, width)

    def transform(raw_bytes):
        image = PIL.Image.open(io.BytesIO(raw_bytes))
        if size is not None:
            image = image.resize(size, interpolation)

        # Convert to torch Tensor. Adapted from
        # torchvision.transform.functional.to_tensor.
        if image.mode == '1':
            tensor = 255 * torch.from_numpy(
                np.array(image, np.uint8, copy=False))
        else:
            tensor = torch.ByteTensor(
                torch.ByteStorage.from_buffer(image.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK.
        if image.mode == 'YCbCr':
            n_channel = 3
        elif image.mode == 'I;16':
            n_channel = 1
        else:
            n_channel = len(image.mode)
        tensor = tensor.view(image.size[1], image.size[0], n_channel)
        return tensor

    return transform


class FeatureDescription(NamedTuple):
    r"""Description of a feature."""
    dtype: np.dtype
    shape: Optional[Tuple[int, ...]]


def _convert_feature_hparams(feature_types: Union[Dict[str, Any], HParams]) \
        -> Dict[str, FeatureDescription]:
    features = {}
    for key, value in feature_types.items():
        if len(value) == 3:
            if isinstance(value[-1], int):
                shape = (value[-1],)
            else:
                shape = tuple(value[-1])  # type: ignore
        else:
            shape = tuple()  # type: ignore
        dtype = get_numpy_dtype(value[0])
        if len(value) < 2 or value[1] == 'FixedLenFeature':
            features[key] = FeatureDescription(dtype, shape)
        elif value[1] == 'FixedLenSequenceFeature':
            if shape[0] is not None:
                raise ValueError("'FixedLenSequenceFeature should have "
                                 "None as first dimension in shape.")
            features[key] = FeatureDescription(dtype, shape)
        elif value[1] == 'VarLenFeature':
            features[key] = FeatureDescription(dtype, None)
        else:
            raise ValueError(
                f"Unsupported feature type '{value[1]}' for key '{key}', "
                f"only 'FixedLenFeature', 'FixedLenSequenceFeature', and "
                f"'VarLenFeature' are supported as of now.")
    return features


class RecordData(DataBase[Dict[str, Any], Dict[str, Any]]):
    r"""TFRecord data which loads and processes TFRecord files.

    This module can be used to process image data, features, etc.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams`
            for the defaults.
        device: The device of the produces batches. For GPU training, set to
            current CUDA device.

    The module reads and restores data from TFRecord files and
    results in a TF Dataset whose element is a Python `dict` that maps feature
    names to feature values. The features names and dtypes are specified in
    :attr:`hparams["dataset"]["feature_original_types"]`.

    The module also provides simple processing options for image data, such
    as image resize.

    Example:

        .. code-block:: python

            # Read data from TFRecord file
            hparams={
                'dataset': {
                    'files': 'image1.tfrecord',
                    'feature_original_types': {
                        'height': ['tf.int64', 'FixedLenFeature'],
                        'width': ['tf.int64', 'FixedLenFeature'],
                        'label': ['tf.int64', 'FixedLenFeature'],
                        'image_raw': ['tf.string', 'FixedLenFeature']
                    }
                },
                'batch_size': 1
            }
            data = TFRecordData(hparams)
            iterator = DataIterator(data)
            batch = iterator.get_next()

            iterator.switch_to_dataset(sess) # initializes the dataset
            batch_ = sess.run(batch)
            # batch_ == {
            #    'data': {
            #        'height': [239],
            #        'width': [149],
            #        'label': [1],
            #
            #        # 'image_raw' is a list of image data bytes in this
            #        # example.
            #        'image_raw': [...],
            #    }
            # }

        .. code-block:: python

            # Read image data from TFRecord file and do resizing
            hparams={
                'dataset': {
                    'files': 'image2.tfrecord',
                    'feature_original_types': {
                        'label': ['tf.int64', 'FixedLenFeature'],
                        'image_raw': ['tf.string', 'FixedLenFeature']
                    },
                    'image_options': {
                        'image_feature_name': 'image_raw',
                        'resize_height': 512,
                        'resize_width': 512,
                    }
                },
                'batch_size': 1
            }
            data = TFRecordData(hparams)
            iterator = DataIterator(data)
            batch = iterator.get_next()

            iterator.switch_to_dataset(sess) # initializes the dataset
            batch_ = sess.run(batch)
            # batch_ == {
            #    'data': {
            #        'label': [1],
            #
            #        # "image_raw" is a list of a "numpy.ndarray" image
            #        # in this example. Each image has a width of 512 and
            #        # height of 512.
            #        'image_raw': [...]
            #    }
            # }

    """

    def __init__(self, hparams=None, device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())

        feature_types = self._hparams.dataset.feature_original_types
        self._features = _convert_feature_hparams(feature_types)

        convert_types = self._hparams.dataset.feature_convert_types
        self._convert_types = {key: get_numpy_dtype(value)
                               for key, value in convert_types.items()}
        for key, dtype in self._convert_types.items():
            self._features[key] = self._features[key]._replace(dtype=dtype)

        image_options = self._hparams.dataset.image_options
        if isinstance(image_options, HParams):
            image_options = [image_options]
        self._image_transforms: Dict[str, TransformFn] = {}
        for options in image_options:
            key = options.get('image_feature_name')
            if key is None or key not in self._features:
                continue
            self._image_transforms[key] = _create_image_transform(
                options.get('resize_height'), options.get('resize_width'),
                options.get('resize_method') or 'bilinear')

        self._other_transforms = self._hparams.dataset.other_transformations

        data_source = PickleDataSource[Dict[str, Any]](
            self._hparams.dataset.files)

        super().__init__(data_source, hparams, device)

    @classmethod
    def _construct(cls, hparams):
        record_data = cls.__new__(cls)
        record_data._hparams = HParams(hparams, record_data.default_hparams())

        feature_types = record_data._hparams.dataset.feature_original_types
        record_data._features = _convert_feature_hparams(feature_types)

        convert_types = record_data._hparams.dataset.feature_convert_types
        record_data._convert_types = {key: get_numpy_dtype(value)
                                      for key, value in convert_types.items()}
        for key, dtype in record_data._convert_types.items():
            record_data._features[key] = record_data._features[key]. \
                _replace(dtype=dtype)

        image_options = record_data._hparams.dataset.image_options
        if isinstance(image_options, HParams):
            image_options = [image_options]
        record_data._image_transforms = {}
        for options in image_options:
            key = options.get('image_feature_name')
            if key is None or key not in record_data._features:
                continue
            record_data._image_transforms[key] = _create_image_transform(
                options.get('resize_height'), options.get('resize_width'),
                options.get('resize_method') or 'bilinear')

        record_data._other_transforms = \
            record_data._hparams.dataset.other_transformations

        data_name = record_data._hparams.dataset.data_name
        record_data._items = {key: connect_name(data_name, key)
                              for key, _ in record_data._features.items()}

        data_source = SequenceDataSource([])

        super(RecordData, record_data).__init__(data_source, hparams)
        return record_data

    class _RecordWriter(io.BytesIO):
        def __init__(self, file_path: str,
                     features: Dict[str, FeatureDescription]):
            super().__init__()
            self._file_path = file_path
            self._features = features
            self._file_handle = open(self._file_path, 'wb')

        def close(self) -> None:
            self._file_handle.close()

        def write(self, example: Dict[str, Any]):  # type: ignore
            converted = {}
            for key, descriptor in self._features.items():
                value = example[key]
                if descriptor.shape is not None:
                    # FixedLenFeature, convert into NumPy array.
                    value = np.asarray(value, dtype=descriptor.dtype)
                converted[key] = value
            pickle.dump(converted, self._file_handle)

    @classmethod
    def writer(cls, file_path: str,
               feature_original_types: Dict[str, Tuple[Any, ...]]) \
            -> '_RecordWriter':
        r"""Construct a file writer object that saves records in pickled format.

        Example:
            .. code-block:: python

                output_file = "data/train.record"
                feature_original_types = {
                    "input_ids": ["int64", "FixedLenFeature", 128],
                    "label_ids": ["int64", "FixedLenFeature"],
                }
                with tx.data.RecordData.writer(
                        output_file, feature_original_types) as writer:
                    writer.write({
                        "input_ids": np.randint(0, 100, size=128),
                        "label_ids": np.randint(0, 100),
                    })

        Args:
            file_path (str): Path to save the dataset.
            feature_original_types: Feature names and types. Please refer to
                :meth:`default_hparams` for details.

        Returns:
            A file writer object.
        """
        feature_types = _convert_feature_hparams(feature_original_types)
        return cls._RecordWriter(file_path, feature_types)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                # (1) Hyperparameters specific to TFRecord dataset
                'dataset': {
                    'files': [],
                    'feature_original_types': {},
                    'feature_convert_types': {},
                    'image_options': {},
                    "num_shards": None,
                    "shard_id": None,
                    "other_transformations": [],
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
                "name": "tfrecord_data",
            }

        Here:

        1. For the hyperparameters in the :attr:`"dataset"` field:

            `"files"`: str or list
                A (list of) TFRecord file path(s).

            `"feature_original_types"`: dict
                The feature names (str) with their data types and length types,
                key and value in pair
                `feature_name: [dtype, feature_len_type, len]`,

                - `dtype` is a Python type (`int`, `str`), dtype instance from
                  PyTorch (``torch.float``), NumPy (``np.int64``),
                  or TensorFlow (``tf.string``), or their stringified names such
                  as ``"torch.float"`` and ``"np.int64"``. The feature will be
                  read from the files and parsed into this dtype.

                - `feature_len_type` is of type `str`, and can be either
                  'FixedLenFeature' or 'VarLenFeature' for fixed length
                  features and non-fixed length features, respectively.

                - `len` is an `int` and is optional. It is the length for
                  'FixedLenFeature'. Ignored if 'VarLenFeature' is used.

                Example:

                .. code-block:: python

                    feature_original_types = {
                        "input_ids": ["tf.int64", "FixedLenFeature", 128],
                        "label_ids": ["tf.int64", "FixedLenFeature"],
                        "name_lists": ["tf.string", "VarLenFeature"],
                    }

            `"feature_convert_types"`: dict, optional
                Specifies dtype converting after reading the data files. This
                `dict` maps feature names to desired data dtypes. For example,
                you can first read a feature into dtype ``torch.int32`` by
                specifying in "feature_original_types" above, and convert
                the feature to dtype ``"torch.long"`` by specifying here.
                Features not specified here will not do dtype-convert.

                - `dtype` is a Python type (`int`, `str`), dtype instance from
                  PyTorch (``torch.float``), NumPy (``np.int64``),
                  or TensorFlow (``tf.string``), or their stringified names such
                  as ``"torch.float"`` and ``"np.int64"``.

                Be noticed that this converting process is after all the data
                are restored, `feature_original_types` has to be set firstly.

                Example:

                .. code-block:: python

                    feature_convert_types = {
                        "input_ids": "tf.int32",
                        "label_ids": "tf.int32",
                    }

            `"image_options"`: dict, optional
                Specifies the image feature name and performs image resizing,
                includes three fields:

                - "image_feature_name":
                    A `str`, the name of the feature which contains
                    the image data. If set, the image data
                    will be restored in format `numpy.ndarray`.
                - "resize_height":
                    A `int`, the height of the image after resizing.
                - "resize_width":
                    A `int`, the width of the image after resizing

                If either `resize_height` or `resize_width` is not set,
                image data will be restored with original shape.

            .. warning::
                  Sharding is not yet supported. This option (and
                  related ones below) will be ignored.

            "num_shards": int, optional
                The number of data shards in distributed mode. Usually set to
                the number of processes in distributed computing.
                Used in combination with :attr:`"shard_id"`.

            `"shard_id"`: int, optional
                Sets the unique id to identify a shard. The module will
                processes only the corresponding shard of the whole data.
                Used in combination with :attr:`"num_shards"`.

                E.g., in a case of distributed computing on 2 GPUs, the hparams
                of the data module for the two processes can be as below,
                respectively.

                For gpu 0:

                .. code-block:: python

                    dataset: {
                        ...
                        "num_shards": 2,
                        "shard_id": 0
                    }

                For gpu 1:

                .. code-block:: python

                    dataset: {
                        ...
                        "num_shards": 2,
                        "shard_id": 1
                    }

                Also refer to `examples/bert` for a use case.

            `"other_transformations"`: list
                A list of transformation functions or function names/paths to
                further transform each single data instance.

            `"data_name"`: str
                Name of the dataset.

        2. For the **general** hyperparameters, see
        :meth:`texar.data.DataBase.default_hparams` for details.
        """
        hparams = DataBase.default_hparams()
        hparams["name"] = "record_data"
        hparams.update({
            "dataset": _default_record_dataset_hparams()
        })
        return hparams

    def process(self, raw_example: Dict[str, Any]) -> Dict[str, Any]:
        example = raw_example
        for key, dtype in self._convert_types.items():
            example[key] = np.asarray(example[key], dtype=dtype)
        for key, transform in self._image_transforms.items():
            example[key] = transform(example[key])
        for transform in self._other_transforms:
            example = transform(example)
        return example

    def collate(self, examples: List[Dict[str, Any]]) -> Batch:
        batch = {}
        for key, descriptor in self._features.items():
            values = [ex[key] for ex in examples]
            if descriptor.shape is not None:
                # FixedLenFeature, do not pad.
                # NumPy functions work on PyTorch tensors too.
                if len(descriptor.shape) > 0 and descriptor.shape[0] is None:
                    values, _ = padded_batch(values)
                else:
                    values = np.stack(values, axis=0)
                if (not isinstance(values, torch.Tensor) and
                        descriptor.dtype not in [np.str_, np.bytes_]):
                    values = torch.from_numpy(values).to(device=self.device)
                elif isinstance(values, torch.Tensor):
                    values = values.to(device=self.device)
            else:
                # VarLenFeature, just put everything in a Python list.
                pass
            batch[key] = values
        return Batch(len(examples), batch)

    def list_items(self) -> List[str]:
        r"""Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        return list(self._features.keys())

    @property
    def feature_names(self):
        r"""A list of feature names.
        """
        return self.list_items()
