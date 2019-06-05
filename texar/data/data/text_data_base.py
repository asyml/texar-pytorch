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
import io
from abc import ABC
from typing import IO, Iterator, Optional, TypeVar

import torch
from texar.data.data.data_base import DataBase, DataSource
from texar.utils.types import MaybeList

__all__ = [
    "TextLineDataSource",
    "TextDataBase",
]

RawExample = TypeVar('RawExample')
Example = TypeVar('Example')


class TextLineDataSource(DataSource[str]):
    r"""Data source for reading from (multiple) text files. Each line is yielded
    as an example.

    This data source does not support indexing.
    """

    def __init__(self, file_paths: MaybeList[str],
                 compression_type: Optional[str] = None,
                 delimiter: Optional[str] = None,
                 max_length: Optional[int] = None):
        r"""Construct a :class:`TextLineDataSource` instance.

        Args:
            file_paths (str or list[str]): Paths to the text files.
            compression_type (str): The compression type for the text files,
                ``"gzip"`` and ``"zlib"`` are supported. Default is ``None``,
                in which case files are treated as plain text files.
            delimiter (str, optional): Delimiter for tokenization purposes. This
                is used in combination with ``max_length``.
            max_length (int, optional): Maximum length for data examples. Length
                is measured as the number of tokens in a line after being
                tokenized using the provided ``delimiter``. Lines with more than
                ``max_length`` tokens will be dropped.

                .. note::
                    ``delimiter`` and ``max_length`` should both be ``None`` or
                    not ``None``.
        """
        if compression_type is not None:
            compression_type = compression_type.lower()
            if compression_type not in ['gzip', 'zlib']:
                raise ValueError(
                    f"Unsupported compression type: {compression_type}")
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self._compression_type = compression_type
        self._file_paths = file_paths
        self._max_length = max_length
        self._delimiter = delimiter
        if (self._max_length is not None) ^ (self._delimiter is not None):
            raise ValueError("'max_length' and 'delimiter' should both be"
                             "None or not None")

    class _ZlibWrapper(IO[bytes]):
        def __init__(self, file: io.BufferedIOBase):
            import zlib
            self.file = file
            self.zlib = zlib.decompressobj()
            self.buffer = b''

        def read(self, n) -> bytes:
            if len(self.buffer) > 0:
                b = self.buffer[:n]
                self.buffer = self.buffer[n:]
                return b
            while True:
                b = self.zlib.decompress(self.file.read(n))
                if len(b) > 0:
                    break
            if len(b) > n:
                self.buffer = b[n:]
                return b[:n]

        def __getattr__(self, item):
            return getattr(self.file, item)

    def _open_file(self, path: str) -> IO[str]:
        if self._compression_type == 'zlib':
            f = io.TextIOWrapper(self._ZlibWrapper(path))
        elif self._compression_type == 'gzip':
            import gzip
            f = gzip.open(path, 'rt')
        else:
            f = open(path, 'r')
        return f

    def __iter__(self) -> Iterator[str]:
        for path in self._file_paths:
            with self._open_file(path) as f:
                for line in f:
                    line = line.rstrip('\n')
                    if self._max_length is not None:
                        # A very brute way to filter out overly long lines at
                        # this stage, but still better than actually performing
                        # tokenization.
                        if (line.count(self._delimiter) + 1  # type: ignore
                                > self._max_length):
                            continue
                    yield line.rstrip('\n')


class TextDataBase(DataBase[RawExample, Example], ABC):  # pylint: disable=too-few-public-methods
    """Base class inherited by all text data classes.
    """

    def __init__(self, source: DataSource[RawExample], hparams,
                 device: Optional[torch.device] = None):
        super().__init__(source, hparams, device=device)

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
