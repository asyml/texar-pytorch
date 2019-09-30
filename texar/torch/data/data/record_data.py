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
Data class that supports reading pickled data as record structures.
"""
import copy
import io
import pickle
import warnings
from enum import Enum
from typing import (
    Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union)

import numpy as np
import torch

from texar.torch.data.data.data_base import DatasetBase, DataSource
from texar.torch.data.data.dataset_utils import Batch, padded_batch
from texar.torch.hyperparams import HParams
from texar.torch.utils.dtypes import get_numpy_dtype
from texar.torch.utils.types import MaybeList

__all__ = [
    "_default_record_dataset_hparams",
    "PickleDataSource",
    "RecordData",
]


def _default_record_dataset_hparams():
    r"""Returns hyperparameters of a record dataset with default values.

    See :meth:`texar.torch.data.RecordData.default_hparams` for details.
    """
    return {
        "files": [],
        "feature_types": None,
        "feature_original_types": None,
        "feature_convert_types": {},
        "image_options": {},
        "compression_type": None,
        "other_transformations": [],
        "num_shards": None,
        "shard_id": None,
        "data_name": None,
        "@no_typecheck": [
            "files",
            "feature_types",
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
    # compatibility reasons, because in Texar-TF `resize_method` could take the
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


class CollateMethod(Enum):
    StackedTensor = "stacked_tensor"
    PaddedTensor = "padded_tensor"
    List = "list"


class FeatureDescription(NamedTuple):
    r"""Description of a feature."""
    collate_method: CollateMethod
    dtype: Optional[np.dtype]
    shape: Optional[Tuple[int, ...]]


_DEPRECATED_NAME_MAPPING = {
    "FixedLenFeature": "stacked_tensor",
    "FixedLenSequenceFeature": "padded_tensor",
    "VarLenFeature": "list",
}


def _convert_feature_hparams(feature_types: Union[Dict[str, Any], HParams]) \
        -> Dict[str, FeatureDescription]:
    if isinstance(feature_types, HParams):
        feature_types = feature_types.todict()
    else:
        feature_types = copy.deepcopy(feature_types)
    show_deprecation_warning = False
    for key, value in feature_types.items():
        if len(value) > 1 and value[1] in _DEPRECATED_NAME_MAPPING:
            feature_types[key] = (
                value[0], _DEPRECATED_NAME_MAPPING[value[1]], *value[2:])
            show_deprecation_warning = True
    if show_deprecation_warning:
        warnings.warn(f"RecordData feature types "
                      f"{', '.join(repr(x) for x in _DEPRECATED_NAME_MAPPING)} "
                      f"are deprecated. Please see RecordData.default_hparams "
                      f"for update instructions.", UserWarning)

    features = {}
    for key, value in feature_types.items():
        shape: Optional[Tuple[int, ...]] = None
        if len(value) == 3:
            if isinstance(value[-1], int):
                shape = (value[-1],)
            elif all(isinstance(x, int) for x in value[-1]):
                shape = tuple(value[-1])
                if len(shape) == 0:
                    shape = (1,)  # scalar tensor
            else:
                raise ValueError(f"'shape' of feature '{key}' is not of type "
                                 f"int, tuple, or torch.Size")
        if len(value) < 2:
            collate_method = CollateMethod.StackedTensor
        else:
            try:
                collate_method = CollateMethod(value[1])
            except ValueError:
                values = [x.value for x in CollateMethod.__members__.values()]
                raise ValueError(
                    f"Unsupported feature collate method '{value[1]}' for "
                    f"feature '{key}', only "
                    f"{', '.join(repr(x) for x in values)} are "
                    f"supported as of now.")

        dtype = None
        if value[0] is not None:
            dtype = get_numpy_dtype(value[0])
        elif collate_method is not CollateMethod.List:
            raise ValueError(f"'dtype' for feature '{key}' must not be None "
                             f"unless collate method is 'list'")

        features[key] = FeatureDescription(collate_method, dtype, shape)

    return features


def _check_shape(tensor: np.ndarray, key: str, descriptor: FeatureDescription):
    if descriptor.shape is None:
        return
    # Check whether shape matches.
    if descriptor.collate_method is CollateMethod.PaddedTensor:
        shape = tensor.shape[1:]
    else:
        shape = tensor.shape
    if len(shape) == 0:
        shape = (1,)  # scalar tensor

    if shape != descriptor.shape:
        if descriptor.collate_method is CollateMethod.PaddedTensor:
            raise ValueError(
                f"Expected tensor of shape {('any', *descriptor.shape)} for "
                f"feature {key}, but received tensor of shape {tensor.shape}")
        else:
            raise ValueError(
                f"Expected tensor of shape {descriptor.shape} for "
                f"feature {key}, but received tensor of shape {tensor.shape}")


class RecordData(DatasetBase[Dict[str, Any], Dict[str, Any]]):
    r"""Record data which loads and processes pickled files.

    This module can be used to process image data, features, etc.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams`
            for the defaults.
        device: The device of the produced batches. For GPU training, set to
            current CUDA device.

    The module reads and restores data from pickled files and results in a
    dataset whose element is a Python `dict` that maps feature names to feature
    values. The features names and dtypes are specified in
    :attr:`hparams.dataset.feature_types`.

    The module also provides simple processing options for image data, such
    as image resize.

    Example:

        .. code-block:: python

            # Read data from pickled file
            hparams={
                'dataset': {
                    'files': 'image1.pkl',
                    'feature_types': {
                        'height': ['int64', 'list'],  # or 'stacked_tensor'
                        'width': ['int64', 'list'],   # or 'stacked_tensor'
                        'label': ['int64', 'stacked_tensor'],
                        'image_raw': ['bytes', 'stacked_tensor'],
                    }
                },
                'batch_size': 1
            }
            data = RecordData(hparams)
            iterator = DataIterator(data)

            batch = next(iter(iterator))  # get the first batch in dataset
            # batch == {
            #    'data': {
            #        'height': [239],
            #        'width': [149],
            #        'label': tensor([1]),
            #
            #        # 'image_raw' is a NumPy ndarray of raw image bytes in this
            #        # example.
            #        'image_raw': [...],
            #    }
            # }

        .. code-block:: python

            # Read image data from pickled file and do resizing
            hparams={
                'dataset': {
                    'files': 'image2.pkl',
                    'feature_types': {
                        'label': ['int64', 'stacked_tensor'],
                        'image_raw': ['bytes', 'stacked_tensor'],
                    },
                    'image_options': {
                        'image_feature_name': 'image_raw',
                        'resize_height': 512,
                        'resize_width': 512,
                    }
                },
                'batch_size': 1
            }
            data = RecordData(hparams)
            iterator = DataIterator(data)

            batch = next(iter(iterator))  # get the first batch in dataset
            # batch == {
            #    'data': {
            #        'label': tensor([1]),
            #
            #        # "image_raw" is a tensor of image pixel data in this
            #        # example. Each image has a width of 512 and height of 512.
            #        'image_raw': tensor([...])
            #    }
            # }

    """

    def __init__(self, hparams=None, device: Optional[torch.device] = None,
                 data_source: Optional[DataSource] = None):
        self._hparams = HParams(hparams, self.default_hparams())

        feature_types = self._hparams.dataset.feature_original_types
        if feature_types is not None:
            warnings.warn(
                "'feature_original_types' of RecordData is deprecated. Please "
                "see default_hparams of RecordData for update instructions")
        if self._hparams.dataset.feature_types is not None:
            feature_types = self._hparams.dataset.feature_types
        elif feature_types is None:
            raise ValueError("'feature_types' must be specified")
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

        if data_source is None:
            data_source = PickleDataSource[Dict[str, Any]](
                self._hparams.dataset.files)

        super().__init__(data_source, hparams, device)

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
                if descriptor.collate_method is CollateMethod.List:
                    converted[key] = value
                    continue
                # Convert to NumPy array.
                value = np.asarray(value, dtype=descriptor.dtype)
                _check_shape(value, key, descriptor)
                converted[key] = value
            pickle.dump(converted, self._file_handle)

    @classmethod
    def writer(cls, file_path: str,
               feature_types: Dict[str, Tuple[Any, ...]]) \
            -> '_RecordWriter':
        r"""Construct a file writer object that saves records in pickled format.

        Example:

        .. code-block:: python

            file_path = "data/train.pkl"
            feature_types = {
                "input_ids": ["int64", "stacked_tensor", 128],
                "label_ids": ["int64", "stacked_tensor"],
            }
            with tx.data.RecordData.writer(file_path, feature_types) as writer:
                writer.write({
                    "input_ids": np.random.randint(0, 100, size=128),
                    "label_ids": np.random.randint(0, 100),
                })

        Args:
            file_path (str): Path to save the dataset.
            feature_types: Feature names and types. Please refer to
                :meth:`default_hparams` for details.

        Returns:
            A file writer object.
        """
        feature_types = _convert_feature_hparams(feature_types)
        return cls._RecordWriter(file_path, feature_types)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                # (1) Hyperparameters specific to the record data
                'dataset': {
                    'files': [],
                    'feature_types': {},
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
               A (list of) pickled file path(s).

           `"feature_types"`: dict
               The feature names (`str`) with their descriptions in the form of
               ``feature_name: [dtype, feature_collate_method, shape]``:

               - ``dtype`` is a Python type (``int``, ``str``), dtype instance
                 from PyTorch (``torch.float``), NumPy (``np.int64``),
                 or TensorFlow (``tf.string``), or their stringified names such
                 as ``"torch.float"`` and ``"np.int64"``. The feature will be
                 read from the files and parsed into this dtype.

               - ``feature_collate_method`` is of type ``str``, and describes
                 how features are collated in the batch. Available values are:

                 - ``"stacked_tensor"``: Features are assumed to be tensors of a
                   fixed shape (or scalars). When collating, features are
                   stacked, with the batch dimension being the first dimension.
                   This is the default value if ``feature_collate_method`` is
                   not specified. For example:

                   - 5 scalar features -> a tensor of shape [5].
                   - 4 tensor features, each of shape [6, 5] -> a tensor of
                     shape [4, 6, 5].

                 - ``"padded_tensor"``: Features are assumed to be tensors, with
                   all dimensions except the first having the same size. When
                   collating, features are padded with zero values along the
                   end of the first dimension so that every tensor has the same
                   size, and then stacked, with the batch dimension being the
                   first dimension. For example:

                   - 3 tensor features, with shapes [4, 7, 8], [5, 7, 8], and
                     [4, 7, 8] -> a tensor of shape [3, 5, 7, 8].

                 - ``"list"``: Features can be any objects. When collating, the
                   features are stored in a Python list.

               - ``shape`` is optional, and can be of type ``int``, `tuple``, or
                 ``torch.Size``. If specified, shapes of tensor features will be
                 checked, depending on the ``feature_collate_method``:

                 - ``"stacked_tensor"``: The shape of every feature tensor must
                   be ``shape``.
                 - ``"padded_tensor"``: The shape (excluding first dimension)
                   of every feature tensor must be ``shape``.
                 - ``"list"``: ``shape`` is ignored.

                 .. note::
                    Shape check is performed before any transformations are
                    applied.

               Example:

               .. code-block:: python

                   feature_types = {
                       "input_ids": ["int64", "stacked_tensor", 128],
                       "label_ids": ["int64", "stacked_tensor"],
                       "name_lists": ["string", "list"],
                   }

               .. note::
                   This field is named `"feature_original_types"` in Texar-TF.
                   This name is still supported, but is deprecated in favor of
                   `"feature_types"`.

                   Texar-TF also uses different names for feature types:

                   - ``"FixedLenFeature"`` corresponds to ``"stacked_tensor"``.
                   - ``"FixedLenSequenceFeature"`` corresponds to
                     ``"padded_tensor"``.
                   - ``"VarLenFeature"`` corresponds to ``"list"``.

                   These names are also accepted in Texar-PyTorch, but are
                   deprecated in favor of the new names.

           `"feature_convert_types"`: dict, optional
               Specifies dtype converting after reading the data files. This
               `dict` maps feature names to desired data dtypes. For example,
               you can first read a feature into dtype ``torch.int32`` by
               specifying in :attr:`"feature_types"` above, and convert
               the feature to dtype ``"torch.long"`` by specifying here.
               Features not specified here will not do dtype-convert.

               - ``dtype`` is a Python type (`int`, `str`), dtype instance from
                 PyTorch (``torch.float``), NumPy (``np.int64``),
                 or TensorFlow (``tf.string``), or their stringified names such
                 as ``"torch.float"`` and ``"np.int64"``.

               Note that this converting process happens after all the data
               are restored.

               Example:

               .. code-block:: python

                   feature_convert_types = {
                       "input_ids": "int32",
                       "label_ids": "int32",
                   }

           `"image_options"`: dict, optional
               Specifies the image feature name and performs image resizing,
               includes three fields:

               - `"image_feature_name"`: str
                   The name of the feature which contains the image data. If
                   set, the image data will be restored in a `numpy.ndarray`.
               - `"resize_height"`: int
                   The height of the image after resizing.
               - `"resize_width"`: int
                   The width of the image after resizing.

               If any of :attr:`"resize_height"` or :attr:`"resize_width"` is
               not set, image data will be restored with original shape.

           `"num_shards"`: int, optional
               The number of data shards in distributed mode. Usually set to
               the number of processes in distributed computing.
               Used in combination with :attr:`"shard_id"`.

               .. warning::
                   Sharding is not yet supported. This option (and
                   related ones below) will be ignored.

           `"shard_id"`: int, optional
               Sets the unique id to identify a shard. The module will
               processes only the corresponding shard of the whole data.
               Used in combination with :attr:`"num_shards"`.

               For example, in a case of distributed computing on 2 GPUs, the
               hyperparameters of the data module for the two processes can be
               configured as below, respectively.

               For GPU 0:

               .. code-block:: python

                   dataset: {
                       ...
                       "num_shards": 2,
                       "shard_id": 0
                   }

               For GPU 1:

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
           :meth:`texar.torch.data.DatasetBase.default_hparams` for details.
        """
        hparams = DatasetBase.default_hparams()
        hparams["name"] = "record_data"
        hparams.update({
            "dataset": _default_record_dataset_hparams()
        })
        return hparams

    def process(self, raw_example: Dict[str, Any]) -> Dict[str, Any]:
        for key, descriptor in self._features.items():
            _check_shape(raw_example[key], key, descriptor)
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
            if descriptor.collate_method is not CollateMethod.List:
                # NumPy functions work on PyTorch tensors too.
                if descriptor.collate_method is CollateMethod.StackedTensor:
                    values = np.stack(values, axis=0)
                else:  # padded_tensor
                    values, _ = padded_batch(values)
                if (not isinstance(values, torch.Tensor) and
                        descriptor.dtype not in [np.str_, np.bytes_]):
                    values = torch.from_numpy(values)
            else:
                # Just put everything in a Python list.
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
