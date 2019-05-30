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
from typing import Iterable, List, Optional, TypeVar

from torch.utils.data import Dataset

from texar.data.data.data_base import DataBase
from texar.utils.types import MaybeList

__all__ = [
    "TextLineDataset",
    "TextDataBase",
]

Example = TypeVar('Example')


class TextLineDataset(Dataset):
    def __init__(self, file_paths: MaybeList[str],
                 compression_type: Optional[str] = None):
        if compression_type is not None:
            raise NotImplementedError
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        lines: List[str] = []
        for path in file_paths:
            with open(path, 'r') as f:
                lines.extend(line.rstrip('\n') for line in f)
        self._lines = lines

    def __getitem__(self, index) -> str:
        return self._lines[index]

    def __iter__(self) -> Iterable[str]:
        return iter(self._lines)

    def __len__(self) -> int:
        return len(self._lines)


class TextDataBase(DataBase[Example], ABC):  # pylint: disable=too-few-public-methods
    """Base class inherited by all text data classes.
    """

    def __init__(self, hparams):
        super().__init__(hparams)

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
