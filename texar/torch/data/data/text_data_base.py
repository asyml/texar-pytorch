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
import locale
from abc import ABC
from typing import IO, Iterator, List, Optional, TypeVar

import torch
from texar.torch.data.data.data_base import DatasetBase, DataSource
from texar.torch.utils.types import MaybeList

__all__ = [
    "TextLineDataSource",
    "TextDataBase",
]

RawExample = TypeVar('RawExample')
Example = TypeVar('Example')


class TextLineDataSource(DataSource[List[str]]):
    r"""Data source for reading from (multiple) text files. Each line is
    tokenized and yielded as an example.

    This data source does not support indexing.

    Args:
        file_paths (str or list[str]): Paths to the text files.
        compression_type (str, optional): The compression type for the text
            files, ``"gzip"`` and ``"zlib"`` are supported. Default is
            `None`, in which case files are treated as plain text files.
        encoding (str, optional): Encoding for the files. By default uses
            the default locale of the system (usually UTF-8).
        delimiter (str, optional): Delimiter for tokenization purposes. This
            is used in combination with ``max_length``. If `None`, text is split
            on any blank character.
        max_length (int, optional): Maximum length for data examples. Length
            is measured as the number of tokens in a line after being
            tokenized using the provided ``delimiter``. Lines with more than
            ``max_length`` tokens will be dropped.
    """

    def __init__(self, file_paths: MaybeList[str],
                 compression_type: Optional[str] = None,
                 encoding: Optional[str] = None,
                 delimiter: Optional[str] = None,
                 max_length: Optional[int] = None):
        if compression_type is not None:
            compression_type = compression_type.lower()
            if compression_type not in ['gzip', 'zlib']:
                raise ValueError(
                    f"Unsupported compression type: {compression_type}")
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self._compression_type = compression_type
        self._encoding = encoding or locale.getpreferredencoding()
        self._file_paths = file_paths
        self._max_length = max_length
        self._delimiter = delimiter

    class _ZlibWrapper(io.BufferedReader):
        def __init__(self, raw: IO[bytes]):
            super().__init__(raw)  # type: ignore
            import zlib
            self.file = raw
            self.zlib = zlib.decompressobj()
            self.buffer = b''

        @property
        def closed(self) -> bool:
            return self.file.closed

        def readable(self) -> bool:
            return True

        def close(self) -> None:
            self.file.close()

        def read1(self, n: int = -1) -> bytes:
            # Our implementation does not really satisfy the definition for
            # `read1`, but whatever, it seems to work.
            if n == -1:
                raw = self.file.read(n)
                b = self.buffer
                self.buffer = b''
                if raw:
                    cur = self.zlib.decompress(raw)
                    if len(cur) > 0:
                        if len(b) > 0:
                            b = b + cur
                        else:
                            b = cur
            else:
                if len(self.buffer) > 0:
                    b = self.buffer[:n]
                    self.buffer = self.buffer[n:]
                    return b
                while True:
                    raw = self.file.read(n)
                    if not raw:
                        return b''
                    b = self.zlib.decompress(raw)
                    if len(b) > 0:
                        break
                if len(b) > n:
                    self.buffer = b[n:]
                    return b[:n]
            return b

        def read(self, n: Optional[int] = -1) -> bytes:
            if n is None or n < 0:
                n = -1
            return self.read1(n)

        def __getattr__(self, item):
            return getattr(self.file, item)

    def _open_file(self, path: str) -> IO[str]:
        if self._compression_type == 'zlib':
            f: IO[str] = io.TextIOWrapper(  # type: ignore
                self._ZlibWrapper(open(path, 'rb')), encoding=self._encoding)
        elif self._compression_type == 'gzip':
            import gzip
            f = gzip.open(path, 'rt', encoding=self._encoding)
        else:
            f = open(path, 'r', encoding=self._encoding)
        return f

    def __iter__(self) -> Iterator[List[str]]:
        for path in self._file_paths:
            with self._open_file(path) as f:
                for line in f:
                    tokens = line.split(self._delimiter)
                    if (self._max_length is not None and
                            len(tokens) > self._max_length):
                        continue
                    yield tokens


class TextDataBase(DatasetBase[RawExample, Example], ABC):
    r"""Base class inherited by all text data classes.
    """

    def __init__(self, source: DataSource[RawExample], hparams,
                 device: Optional[torch.device] = None):
        super().__init__(source, hparams, device=device)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.

        See the specific subclasses for the details.
        """
        hparams = DatasetBase.default_hparams()
        hparams.update({
            "bucket_boundaries": [],
            "bucket_batch_sizes": None,
            "bucket_length_fn": None})
        return hparams
