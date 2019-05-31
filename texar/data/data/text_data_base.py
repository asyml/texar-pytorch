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
Base text data class that is inherited by all text data classes.
"""
from abc import ABC
from typing import Iterable, Optional, TypeVar

from texar.data.data.data_base import DataBase, DataSource
from texar.utils.types import MaybeList

__all__ = [
    "TextLineDataSource",
    "TextDataBase",
]

RawExample = TypeVar('RawExample')
Example = TypeVar('Example')


class TextLineDataSource(DataSource[str]):
    def __init__(self, file_paths: MaybeList[str],
                 compression_type: Optional[str] = None):
        if compression_type is not None:
            raise NotImplementedError
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self._file_paths = file_paths

    def __iter__(self) -> Iterable[str]:
        for path in self._file_paths:
            with open(path, 'r') as f:
                for line in f:
                    yield line.rstrip('\n')


class TextDataBase(DataBase[RawExample, Example], ABC):  # pylint: disable=too-few-public-methods
    """Base class inherited by all text data classes.
    """

    def __init__(self, source: DataSource[RawExample], hparams):
        super().__init__(source, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.

        See the specific subclasses for the details.
        """
        hparams = DataBase.default_hparams()
        hparams.update({
            "bucket_boundaries": [],
            "bucket_batch_sizes": None,
            "bucket_length_fn": None})
        return hparams
