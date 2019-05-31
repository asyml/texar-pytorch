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
Base data class that is inherited by all data classes.
A data defines data reading, parsing, batching, and other
preprocessing operations.
"""
import warnings
from abc import ABC
from typing import Dict, Generic, Iterable, Iterator, List, \
    Optional, Sequence, Tuple, TypeVar, Union

import torch
from torch.utils.data import Dataset

from texar.data.data.dataset_utils import Batch
from texar.data.data.dataset_utils import _CacheStrategy, _LazyStrategy
from texar.hyperparams import HParams

__all__ = [
    "DataSource",
    "SequenceDataSource",
    "IterDataSource",
    "ZipDataSource",
    "RecordDataSource",
    "DataBase",
]

RawExample = TypeVar('RawExample')  # type of a raw example loaded from source
Example = TypeVar('Example')  # type of a data example


class DataSource(Generic[RawExample], ABC):
    r"""Base class for all datasets. Different to PyTorch
    :class:`~torch.utils.data.Dataset`, subclasses of this class are not
    required to implement `__getitem__` (default implementation raises
    `TypeError`), which is beneficial for certain sources that only supports
    iteration (reading from text files, reading Python iterators, etc.)
    """

    def __getitem__(self, index: int) -> RawExample:
        raise TypeError("This DataSource does not support random access")

    def __iter__(self) -> Iterator[RawExample]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise TypeError("This DataSource does not support random access")


class SequenceDataSource(DataSource[RawExample]):
    r"""Data source for reading from Python sequences.
    """

    def __init__(self, sequence: Sequence[RawExample]):
        self._seq = sequence

    def __getitem__(self, index: int) -> RawExample:
        return self._seq[index]

    def __iter__(self) -> Iterator[RawExample]:
        return iter(self._seq)

    def __len__(self) -> int:
        return len(self._seq)


class IterDataSource(DataSource[RawExample]):
    r"""Data source for reading from Python iterables. Please note: if passed
    an *iterator* and caching strategy is set to 'none', then the data source
    can only be iterated over once.
    """

    def __init__(self, iterable: Iterable[RawExample]):
        self._iter = iterable

    def __iter__(self) -> Iterator[RawExample]:
        return iter(self._iter)


class ZipDataSource(DataSource[Tuple[RawExample, ...]]):
    r"""Data source by combining multiple sources.
    """

    def __init__(self, *sources: DataSource[RawExample]):
        self._sources = list(sources)

    def __getitem__(self, index: int) -> Tuple[RawExample, ...]:
        return tuple(source[index] for source in self._sources)

    def __iter__(self) -> Iterator[Tuple[RawExample, ...]]:
        return zip(*[iter(source) for source in self._sources])

    def __len__(self) -> int:
        return min(len(source) for source in self._sources)


class RecordDataSource(DataSource[Dict[str, RawExample]]):
    r"""Data source by structuring multiple source.
    """

    def __init__(self, sources: Dict[str, DataSource[RawExample]]):
        self._sources = sources

    def __getitem__(self, index: int) -> Dict[str, RawExample]:
        return {key: source[index] for key, source in self._sources.items()}

    def __iter__(self) -> Iterator[Dict[str, RawExample]]:
        keys = list(self._sources.keys())
        iterator = zip(*[iter(source) for source in self._sources.values()])
        for values in iterator:
            yield {key: value for key, value in zip(keys, values)}

    def __len__(self) -> int:
        return min(len(source) for source in self._sources.values())


class _CachedDataSource(DataSource[Example]):
    r"""Wrapper for random access support over a data source that does not
    implement `__getitem__`. This class is only used internally in
    :class:`~texar.data.data.DataBase`, while conforming to user
    `cache_strategy` and `shuffle_buffer_size` settings.
    """

    def __init__(self, data_source: DataSource[Example],
                 erase_after_access: bool = True):
        r"""

        Args:
            data_source: The data source to wrap around.
            erase_after_access: If `True`, cached examples are erased after
                being accessed through `__getitem__`. Useful when
                :class:`~texar.data.data.DataBase` hyperparameter
                `cache_strategy` is set to `none` or `processed`.
        """
        self._source = data_source
        self._iter = iter(data_source)
        self._max_index = -1
        self._erase_after_access = erase_after_access
        if erase_after_access:
            self._cache: Dict[int, Example] = {}
        else:
            self._cache: List[Example] = []

    def __getitem__(self, index: int) -> Example:
        # If specified `index` is not yet prefetched (or has already been
        # accessed), this method may throw `IndexError` or `KeyError`.
        example = self._cache[index]
        if self._erase_after_access:
            del self._cache[index]
        return example

    def __iter__(self) -> Iterator[Example]:
        return iter(self._source)

    def prefetch(self, index: int):
        while self._max_index < index:
            self._max_index += 1
            example = next(self._iter)
            if self._erase_after_access:
                self._cache[self._max_index] = example
            else:
                self._cache.append(example)

    @property
    def max_index(self) -> int:
        return self._max_index

    def reset(self) -> None:
        self._iter = iter(self._source)
        self._max_index = -1


class DataBase(Dataset, Generic[RawExample, Example], ABC):
    r"""Base class inherited by all data classes.
    """

    _source: DataSource[RawExample]

    def __init__(self, source: DataSource[RawExample], hparams,
                 device: Optional[torch.device] = None):
        self._source = source
        self._hparams = HParams(hparams, self.default_hparams())
        self.device = device

        # Check and convert strategy hyperparameters.
        self._lazy_strategy = _LazyStrategy(self._hparams.lazy_strategy)
        self._cache_strategy = _CacheStrategy(self._hparams.cache_strategy)
        if self._lazy_strategy is _LazyStrategy.NONE:
            self._cache_strategy = _CacheStrategy.PROCESSED
        elif self._lazy_strategy is _LazyStrategy.PROCESS:
            if self._cache_strategy is _CacheStrategy.NONE:
                self._cache_strategy = _CacheStrategy.LOADED
        self._uses_multi_processing = self._hparams.num_parallel_calls > 1

        # Check whether data source supports random access, and obtain dataset
        # size if it does.
        self._supports_random_access = True
        if self._lazy_strategy is not _LazyStrategy.NONE:
            try:
                self._dataset_size = len(self._source)
                _ = self._source[0]
            except TypeError:
                self._supports_random_access = False
                erase_after_access = (
                        self._cache_strategy is not _CacheStrategy.LOADED)
                self._source = _CachedDataSource(source, erase_after_access)
                self._dataset_size = None

        self._processed_cache = []
        self._fully_cached = False
        # Perform eager loading/processing if required.
        if self._lazy_strategy is _LazyStrategy.NONE:
            # Process entire dataset and cache.
            for raw_example in self._source:
                self._processed_cache.append(self._process(raw_example))
            self._dataset_size = len(self._processed_cache)
            self._fully_cached = True
        else:
            if self._lazy_strategy is _LazyStrategy.PROCESS:
                # Load entire dataset. Note that if data source supports random
                # access, we assume it is already loaded into memory.
                if not self._supports_random_access:
                    self._prefetch_source(None)

            if self._cache_strategy is _CacheStrategy.PROCESSED:
                self._reorder_cache: Dict[int, Example] = {}

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
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
                "name": "data",
            }

        Here:

            "num_epochs" : int
                Number of times the dataset should be repeated. An
                :tf_main:`OutOfRangeError <errors/OutOfRangeError>` signal will
                be raised after the whole repeated dataset has been iterated
                through.

                E.g., For training data, set it to 1 (default) so that you
                will get the signal after each epoch of training. Set to -1
                to repeat the dataset indefinitely.

            "batch_size" : int
                Batch size, i.e., the number of consecutive elements of the
                dataset to combine in a single batch.

            "allow_smaller_final_batch" : bool
               Whether to allow the final batch to be smaller if there are
               insufficient elements left. If `False`, the final batch is
               discarded if it is smaller than batch size. Note that,
               if `True`, `output_shapes` of the resulting dataset
               will have a a **static** batch_size dimension equal to
               "batch_size".

            "shuffle" : bool
                Whether to randomly shuffle the elements of the dataset.

            "shuffle_buffer_size" : int
                The buffer size for data shuffling. The larger, the better
                the resulting data is mixed.

                If `None` (default), buffer size is set to the size of the
                whole dataset (i.e., make the shuffling the maximally
                effective).

            "shard_and_shuffle" : bool
                Whether to first shard the dataset and then shuffle each
                block respectively. Useful when the whole data is too large to
                be loaded efficiently into the memory.

                If `True`, :attr:`shuffle_buffer_size` must be specified to
                determine the size of each shard.

            "num_parallel_calls" : int
                Number of elements from the datasets to process in parallel.

            "prefetch_buffer_size" : int
                The maximum number of elements that will be buffered when
                prefetching.

            max_dataset_size : int
                Maximum number of instances to include in
                the dataset. If set to `-1` or greater than the size of
                dataset, all instances will be included. This constraint is
                imposed after data shuffling and filtering.

            seed : int, optional
                The random seed for shuffle.

                Note that if a seed is set, the shuffle order will be exact
                the same every time when going through the (repeated) dataset.

                For example, consider a dataset with elements [1, 2, 3], with
                "num_epochs"`=2` and some fixed seed, the resulting sequence
                can be: `2 1 3, 1 3 2 | 2 1 3, 1 3 2, ...` That is, the orders
                are different **within** every `num_epochs`, but are the same
                **across** the `num_epochs`.

            name : str
                Name of the data.

            lazy_strategy : str
                Lazy strategy for data examples. Lazy loading/processing defers
                data loading/processing until when it's being accessed.
                Non-lazy (eager) loading/processing would load/process all data
                upon construction of dataset. Available options are:

                    - `none`: Perform eager loading and processing.
                    - `process`: Perform eager loading and lazy processing.
                    - `all`: Perform lazy loading and processing.

                Defaults to `all`.

            cache_strategy: str
                Caching strategy for data examples. Available options are:

                    - `none`: No data is cached. Data is always loaded from
                        source (e.g. file) and processed upon access.
                    - `loaded`: Only cache raw data loaded from source,
                        processing routines are performed upon access.
                    - `processed`: Processed data is cached. **Note:** raw data
                        will not be cached in this case, because raw data is
                        only used to construct the processed data.

                Default value is `loaded`. This option depends on the value of
                `lazy_strategy`, specifically:

                    - When `lazy_strategy` is `none`, all choices of
                        `cache_strategy` are equivalent to `processed`.
                    - When `lazy_strategy` is `process`, `none` is equivalent
                        to `loaded`.
        """
        return {
            "name": "data",
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
            "lazy_strategy": 'none',
            "cache_strategy": 'processed',
        }

    def to(self, device: torch.device):
        self.device = device

    def _prefetch_source(self, index: Optional[int]) -> Optional[int]:
        r"""Prefetches data so `__getitem__` will be available. This method
        should only be called in the main process, because data sources are not
        guaranteed to be thread-safe.

        Args:
            index: Prefetch data up to this index. If `None`, prefetch all data.

        Returns:
            If `index` is `None`, or `index` is greater than dataset size,
            returns the inferred dataset size. Otherwise, returns `None`.
        """
        if not self._supports_random_access:
            self._source: _CachedDataSource
            try:
                if index is not None:
                    self._source.prefetch(index)
                else:
                    max_index = 10 ** 8
                    self._source.prefetch(max_index)
                    warnings.warn(
                        f"The data source contains more than {max_index:.2e} "
                        f"examples. Please check whether it is infinite.")
                    while True:
                        max_index *= 2
                        self._source.prefetch(max_index)
            except StopIteration:
                self._dataset_size = self._source.max_index
                self._source.reset()
                return self._dataset_size
        return None

    def __len__(self) -> int:
        if self._dataset_size is None:
            warnings.warn(
                "The provided data source does not support random access. To "
                "obtain dataset size, a full traversal must be performed. "
                "This is often unnecessary and slow, consider redesigning your "
                "use case.")
            self._prefetch_source(None)
        return self._dataset_size

    def _process(self, raw_example: RawExample) -> Example:
        raise NotImplementedError

    def __getitem__(self, index: Union[int, Tuple[int, RawExample]]) -> Example:
        if isinstance(index, int):
            if self._fully_cached:
                return self._processed_cache[index]
            else:
                return self._process(self._source[index])
        else:
            # `index` is a tuple of (index, example).
            return self._process(index[1])

    def _add_cached_examples(self, indices: List[int], examples: List[Example]):
        # In this case, `_CachedDataSource.__getitem__` will be
        # called on worker processes, so the cache cannot be
        # deleted. Thus, we move deletion to
        # `_add_cached_examples`.
        if (not self._supports_random_access and
                self._uses_multi_processing and
                (self._lazy_strategy is _LazyStrategy.PROCESS and
                 self._cache_strategy is _CacheStrategy.PROCESSED)):
            for index in indices:
                del self._source._cache[index]
        for index, example in zip(indices, examples):
            self._reorder_cache[index] = example
        while len(self._processed_cache) in self._reorder_cache:
            index = len(self._processed_cache)
            self._processed_cache.append(self._reorder_cache[index])
            del self._reorder_cache[index]
        if self._dataset_size == len(self._processed_cache):
            self._fully_cached = True

    def _start_iteration(self) -> None:
        if not self._supports_random_access:
            self._source.reset()

    @property
    def num_epochs(self):
        r"""Number of epochs.
        """
        return self._hparams.num_epochs

    @property
    def batch_size(self):
        r"""The batch size.
        """
        return self._hparams.batch_size

    @property
    def hparams(self):
        r"""A :class:`~texar.HParams` instance of the
        data hyperparameters.
        """
        return self._hparams

    @property
    def name(self):
        r"""Name of the module.
        """
        return self._hparams.name

    @property
    def dataset(self):  # TODO: maybe change this to `data_source`
        r"""The data source.
        """
        return self._source

    def _collate(self, examples: List[Example]) -> Batch:
        r"""Create a `collate_fn` for :class:`~torch.utils.data.DataLoader` that
        is used to collate (combine) examples into batches. The function takes a
        list of processed examples, and returns an instance of
        :class:`~texar.data.data.Batch`.

        Implementation should make sure that the returned callable is safe and
        efficient under multi-processing scenarios. Basically, do not rely on
        attributes of `self` that could be modified during iteration.

        Args:
            examples: A list of processed examples in a batch.

        Returns:
            The collated batch.
        """
        raise NotImplementedError

    def _collate_and_maybe_return(self, examples: List[Example]) -> \
            Union[Batch, Tuple[List[Example], Batch]]:
        r"""Called by :class:`~texar.data.data.DataIterator` to obtain the
        collated batch (and processed examples under certain circumstances).

        Args:
            examples: A list of processed examples in a batch.

        Returns:
            The collated batch.
        """
        batch = self._collate(examples)
        if self._should_return_processed_examples:
            return examples, batch
        return batch

    @property
    def _should_return_processed_examples(self):
        r"""Returns `True` if the worker threads should perform processing and
        return the processed examples.
        """
        return (not self._fully_cached and
                self._lazy_strategy is not _LazyStrategy.NONE and
                self._cache_strategy is _CacheStrategy.PROCESSED)

    @property
    def _should_yield_raw_example(self):
        r"""Returns `True` if the sampler should yield raw examples.
        """
        return (self._lazy_strategy is _LazyStrategy.ALL and
                (self._cache_strategy is _CacheStrategy.NONE or
                 not self._fully_cached))

    @property
    def _should_call_prefetch_source(self):
        return (self._dataset_size is None or
                (self._lazy_strategy is _LazyStrategy.ALL and
                 self._cache_strategy is _CacheStrategy.NONE))
