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
Various data classes that define data reading, parsing, batching, and other
preprocessing operations.
"""
from typing import (Optional, List, Union, Type)

import torch
import numpy as np

from texar.hyperparams import HParams
from texar.data.data.dataset_utils import Batch
from texar.data.data.data_base import DataBase
from texar.data.data.text_data_base import TextLineDataSource

__all__ = [
    "_default_scalar_dataset_hparams",
    "ScalarData"
]


def _default_scalar_dataset_hparams():
    """Returns hyperparameters of a scalar dataset with default values.

    See :meth:`texar.data.ScalarData.default_hparams` for details.
    """
    return {
        "files": [],
        "compression_type": None,
        "data_type": "int",
        "data_name": "data",
        "other_transformations": [],
        "@no_typecheck": ["files"]
    }


class ScalarData(DataBase[str, Union[int, float]]):
    r"""Scalar data where each line of the files is a scalar (int or float),
    e.g., a data label.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.

    The processor reads and processes raw data and results in a dataset
    whose element is a python `dict` including one field. The field name is
    specified in :attr:`hparams["dataset"]["data_name"]`. If not specified,
    the default name is `"data"`. The field name can be accessed through
    :attr:`data_name`.

    This field is a Tensor of shape `[batch_size]` containing a batch of
    scalars, of either int or float type as specified in :attr:`hparams`.

    Example:

        .. code-block:: python

            hparams={
                'dataset': { 'files': 'data.txt', 'data_name': 'label' },
                'batch_size': 2
            }
            data = ScalarData(hparams)
            iterator = DataIterator(data)
            for batch in iterator:
                # batch contains the following
                # batch == {
                #     'label': [2, 9]
                # }
    """

    def __init__(self, hparams, device: Optional[torch.device] = None):
        self._hparams = HParams(hparams, self.default_hparams())
        self._other_transforms = self._hparams.dataset.other_transformations
        data_type = self._hparams.dataset["data_type"]
        self._typecast_func: Union[Type[int], Type[float]]
        if data_type == "int":
            self._typecast_func = int
            self._to_data_type = np.int32
        elif data_type == "float":
            self._typecast_func = float
            self._to_data_type = np.float32
        else:
            raise ValueError("Incorrect 'data_type'. Currently 'int' and "
                             "'float' are supported. Received {}"
                             .format(data_type))
        data_source = TextLineDataSource(
            self._hparams.dataset.files,
            compression_type=self._hparams.dataset.compression_type)
        super().__init__(data_source, hparams, device=device)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                # (1) Hyperparams specific to scalar dataset
                "dataset": {
                    "files": [],
                    "compression_type": None,
                    "data_type": "int",
                    "other_transformations": [],
                    "data_name": "data",
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
                "name": "scalar_data",
            }

        Here:

        1. For the hyperparameters in the :attr:`"dataset"` field:

            `"files"`: str or list
                A (list of) file path(s).

                Each line contains a single scalar number.

            `"compression_type"`: str, optional
                One of "" (no compression), "ZLIB", or "GZIP".

            `"data_type"`: str
                The scalar type. Currently supports "int" and "float".

            `"other_transformations"`: list
                A list of transformation functions or function names/paths to
                further transform each single data instance.

                (More documentations to be added.)

            `"data_name"`: str
                Name of the dataset.

        2. For the **general** hyperparameters, see
        :meth:`texar.data.DataBase.default_hparams` for details.

        """
        hparams = DataBase.default_hparams()
        hparams["name"] = "scalar_data"
        hparams.update({
            "dataset": _default_scalar_dataset_hparams()
        })
        return hparams

    def process(self, raw_example: str) -> Union[int, float]:
        # Apply the "other transformations".
        example: Union[int, float] = self._typecast_func(raw_example)
        for transform in self._other_transforms:
            example = transform(example)
        return example

    def collate(self, examples: List[Union[int, float]]) -> Batch:
        # convert the list of strings into appropriate tensors here
        examples_np = np.array(examples, dtype=self._to_data_type)
        collated_examples = torch.from_numpy(examples_np).to(device=self.device)
        return Batch(len(examples),
                     batch={self.data_name: collated_examples})

    def list_items(self):
        r"""Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        return [self.hparams.dataset["data_name"]]

    @property
    def dataset(self):
        r"""The dataset.
        """
        return self._source

    @property
    def data_name(self):
        r"""The name of the data tensor, "data" by default if not specified in
        :attr:`hparams`.
        """
        return self.hparams.dataset["data_name"]
