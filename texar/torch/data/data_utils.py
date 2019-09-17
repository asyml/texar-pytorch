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
Various utilities specific to data processing.
"""
import collections
import logging
import os
import sys
import tarfile
import urllib.request
import zipfile
from typing import List, Optional, overload, Union, Dict, Tuple

import numpy as np

from texar.torch.utils import utils_io
from texar.torch.utils.types import MaybeList, MaybeTuple, PathLike

__all__ = [
    "maybe_download",
    "read_words",
    "make_vocab",
    "count_file_lines",
]

Py3 = sys.version_info[0] == 3


# TODO: Remove these once pylint supports function stubs.
# pylint: disable=unused-argument,function-redefined,missing-docstring

@overload
def maybe_download(urls: List[str], path: PathLike,
                   filenames: Optional[List[str]] = None,
                   extract: bool = False) -> List[str]: ...


@overload
def maybe_download(urls: str, path: PathLike, filenames: Optional[str] = None,
                   extract: bool = False) -> str: ...


def maybe_download(urls, path, filenames=None, extract=False):
    r"""Downloads a set of files.

    Args:
        urls: A (list of) URLs to download files.
        path (str): The destination path to save the files.
        filenames: A (list of) strings of the file names. If given,
            must have the same length with :attr:`urls`. If `None`,
            filenames are extracted from :attr:`urls`.
        extract (bool): Whether to extract compressed files.

    Returns:
        A list of paths to the downloaded files.
    """
    utils_io.maybe_create_dir(path)

    if not isinstance(urls, (list, tuple)):
        is_list = False
        urls = [urls]
    else:
        is_list = True
    if filenames is not None:
        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]
        if len(urls) != len(filenames):
            raise ValueError(
                '`filenames` must have the same number of elements as `urls`.')

    result = []
    for i, url in enumerate(urls):
        if filenames is not None:
            filename = filenames[i]
        elif 'drive.google.com' in url:
            filename = _extract_google_drive_file_id(url)
        else:
            filename = url.split('/')[-1]
            # If downloading from GitHub, remove suffix ?raw=True
            # from local filename
            if filename.endswith("?raw=true"):
                filename = filename[:-9]

        filepath = os.path.join(path, filename)
        result.append(filepath)

        # if not tf.gfile.Exists(filepath):
        if not os.path.exists(filepath):
            if 'drive.google.com' in url:
                filepath = _download_from_google_drive(url, filename, path)
            else:
                filepath = _download(url, filename, path)

            if extract:
                logging.info('Extract %s', filepath)
                if tarfile.is_tarfile(filepath):
                    tarfile.open(filepath, 'r').extractall(path)
                elif zipfile.is_zipfile(filepath):
                    with zipfile.ZipFile(filepath) as zfile:
                        zfile.extractall(path)
                else:
                    logging.info("Unknown compression type. Only .tar.gz"
                                 ".tar.bz2, .tar, and .zip are supported")
    if not is_list:
        return result[0]
    return result


# pylint: enable=unused-argument,function-redefined,missing-docstring


def _download(url: str, filename: str, path: str) -> str:
    def _progress_hook(count, block_size, total_size):
        percent = float(count * block_size) / float(total_size) * 100.
        sys.stdout.write(f'\r>> Downloading {filename} {percent:.1f}%')
        sys.stdout.flush()

    filepath = os.path.join(path, filename)
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress_hook)
    print()
    statinfo = os.stat(filepath)
    print(f'Successfully downloaded {filename} {statinfo.st_size} bytes')

    return filepath


def _extract_google_drive_file_id(url: str) -> str:
    # id is between `/d/` and '/'
    url_suffix = url[url.find('/d/') + 3:]
    file_id = url_suffix[:url_suffix.find('/')]
    return file_id


def _download_from_google_drive(url: str, filename: str, path: str) -> str:
    r"""Adapted from `https://github.com/saurabhshri/gdrive-downloader`
    """

    try:
        import requests
    except ImportError:
        print("The requests library must be installed to download files from "
              "Google drive. Please see: https://github.com/psf/requests")
        raise

    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    file_id = _extract_google_drive_file_id(url)

    gurl = "https://docs.google.com/uc?export=download"
    sess = requests.Session()
    response = sess.get(gurl, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = sess.get(gurl, params=params, stream=True)

    filepath = os.path.join(path, filename)
    CHUNK_SIZE = 32768
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    print(f'Successfully downloaded {filename}')

    return filepath


def read_words(filename: str, newline_token: Optional[str] = None) -> List[str]:
    r"""Reads word from a file.

    Args:
        filename (str): Path to the file.
        newline_token (str, optional): The token to replace the original newline
            token "\\n". For example, :python:`tx.data.SpecialTokens.EOS`.
            If `None`, no replacement is performed.

    Returns:
        A list of words.
    """
    with open(filename, "r") as f:
        if Py3:
            if newline_token is None:
                return f.read().split()
            else:
                return f.read().replace("\n", newline_token).split()
        else:
            if newline_token is None:
                return f.read().split()
            else:
                return f.read().replace("\n", newline_token).split()


# TODO: Remove these once pylint supports function stubs.
# pylint: disable=unused-argument,function-redefined,missing-docstring

# A saner overloaded version with default arguments...
@overload
def make_vocab(filenames: MaybeList[str], max_vocab_size: int = -1,
               newline_token: Optional[str] = None) -> List[str]: ...


# ... and an insane version.
@overload
def make_vocab(filenames: MaybeList[str], max_vocab_size: int = -1,
               newline_token: Optional[str] = None,
               return_type: str = "list", return_count: bool = False) \
        -> Union[Union[List[str], Tuple[List[str], List[int]]],
                 MaybeTuple[Dict[str, int]]]: ...


def make_vocab(filenames, max_vocab_size=-1, newline_token=None,
               return_type="list", return_count=False):
    r"""Builds vocab of the files.

    Args:
        filenames (str): A (list of) files.
        max_vocab_size (int): Maximum size of the vocabulary. Low frequency
            words that exceeding the limit will be discarded.
            Set to `-1` (default) if no truncation is wanted.
        newline_token (str, optional): The token to replace the original newline
            token "\\n". For example, :python:`tx.data.SpecialTokens.EOS`.
            If `None`, no replacement is performed.
        return_type (str): Either ``list`` or ``dict``. If ``list`` (default),
            this function returns a list of words sorted by frequency. If
            ``dict``, this function returns a dict mapping words to their index
            sorted by frequency.
        return_count (bool): Whether to return word counts. If `True` and
            :attr:`return_type` is ``dict``, then a count dict is returned,
            which is a mapping from words to their frequency.

    Returns:
        - If :attr:`return_count` is False, returns a list or dict containing
          the vocabulary words.

        - If :attr:`return_count` if True, returns a pair of list or dict
          `(a, b)`, where `a` is a list or dict containing the vocabulary
          words, `b` is a list or dict containing the word counts.
    """

    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]

    words: List[str] = []
    for fn in filenames:
        words += read_words(fn, newline_token=newline_token)

    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, counts = list(zip(*count_pairs))
    words: List[str]
    counts: List[int]
    if max_vocab_size >= 0:
        words = words[:max_vocab_size]
    counts = counts[:max_vocab_size]

    if return_type == "list":
        if not return_count:
            return words
        else:
            return words, counts
    elif return_type == "dict":
        word_to_id = dict(zip(words, range(len(words))))
        if not return_count:
            return word_to_id
        else:
            word_to_count = dict(zip(words, counts))
            return word_to_id, word_to_count
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


# pylint: enable=unused-argument,function-redefined,missing-docstring

def count_file_lines(filenames: MaybeList[str]) -> int:
    r"""Counts the number of lines in the file(s).
    """

    def _count_lines(fn):
        with open(fn, "rb") as f:
            i = -1
            for i, _ in enumerate(f):
                pass
            return i + 1

    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]
    num_lines = np.sum([_count_lines(fn) for fn in filenames]).item()
    return num_lines
