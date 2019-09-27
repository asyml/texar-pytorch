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
from typing import (
    Callable, Dict, Generic, Iterable, Iterator, List, Optional, Sequence,
    Tuple, TypeVar, Union)

import torch
from torch.utils.data import Dataset

from texar.torch.data.data.dataset_utils import Batch
from texar.torch.data.data.dataset_utils import _CacheStrategy, _LazyStrategy
from texar.torch.hyperparams import HParams

__all__ = [
    "DataSource",
    "SequenceDataSource",
    "IterDataSource",
    "ZipDataSource",
    "FilterDataSource",
    "RecordDataSource",
    "DatasetBase",
]

RawExample = TypeVar('RawExample')  # type of a raw example loaded from source
Example = TypeVar('Example')  # type of a data example


class DataSource(Generic[RawExample], ABC):
    r"""Base class for all data sources. A data source represents the *source*
    of the data, from which raw data examples are read and returned.

    Different to PyTorch :class:`~torch.utils.data.Dataset`, subclasses of this
    class are not required to implement :meth:`__getitem__` (default
    implementation raises `TypeError`), which is beneficial for certain sources
    that only supports iteration (reading from text files, reading Python
    iterators, etc.)
    """

    def __getitem__(self, index: int) -> RawExample:
        raise TypeError("This DataSource does not support random access")

    def __iter__(self) -> Iterator[RawExample]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise TypeError("This DataSource does not support random access")


class SequenceDataSource(DataSource[RawExample]):
    r"""Data source for reading from Python sequences.

    This data source supports indexing.

    Args:
        sequence: The Python sequence to read from. Note that a sequence should
            be iterable and supports `len`.
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

    This data source does not support indexing.

    Args:
        iterable: The Python iterable to read from.
    """

    def __init__(self, iterable: Iterable[RawExample]):
        self._iter = iterable

    def __iter__(self) -> Iterator[RawExample]:
        return iter(self._iter)


class ZipDataSource(DataSource[Tuple[RawExample, ...]]):
    r"""Data source by combining multiple sources. The raw examples returned
    from this data source are tuples, with elements being raw examples from each
    of the constituting data sources.

    This data source supports indexing if all the constituting data sources
    support indexing.

    Args:
        sources: The list of data sources to combine.
    """

    def __init__(self, *sources: DataSource[RawExample]):
        self._sources = list(sources)

    def __getitem__(self, index: int) -> Tuple[RawExample, ...]:
        return tuple(source[index] for source in self._sources)

    def __iter__(self) -> Iterator[Tuple[RawExample, ...]]:
        return zip(*[iter(source) for source in self._sources])

    def __len__(self) -> int:
        return min(len(source) for source in self._sources)


class FilterDataSource(DataSource[RawExample]):
    r"""Data source for filtering raw examples with a user-specified filter
    function. Only examples for which the filter functions returns `True` are
    returned.

    This data source supports indexing if the wrapped data source supports
    indexing.

    Args:
        source: The data source to filter.
        filter_fn: A callable taking a raw example as argument and returning a
            boolean value, indicating whether the raw example should be
            **kept**.
    """

    def __init__(self, source: DataSource[RawExample],
                 filter_fn: Callable[[RawExample], bool]):
        self._source = source
        self._filter_fn = filter_fn

    def __iter__(self) -> Iterator[RawExample]:
        for sentence in self._source:
            if self._filter_fn(sentence):
                yield sentence


class RecordDataSource(DataSource[Dict[str, RawExample]]):
    r"""Data source by structuring multiple sources. The raw examples returned
    from this data source are dictionaries, with values being raw examples from
    each of the constituting data sources.

    This data source supports indexing if all the constituting data sources
    support indexing.

    Args:
        sources: A dictionary mapping names to data sources, containing the
            data sources to combine.
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


class _TruncatedDataSource(DataSource[RawExample]):
    def __init__(self, data_source: DataSource[RawExample], max_size: int):
        self._source = data_source
        self._max_size = max_size

    def __getitem__(self, item) -> RawExample:
        if item >= self._max_size:
            raise IndexError(
                f"Data index ({item}) out of range [0, {self._max_size})")
        return self._source[item]

    def __iter__(self) -> Iterator[RawExample]:
        count = 0
        iterator = iter(self._source)
        while count < self._max_size:
            yield next(iterator)
            count += 1

    def __len__(self) -> int:
        try:
            length = min(len(self._source), self._max_size)
        except TypeError:
            length = self._max_size
        return length


class _TransformedDataSource(DataSource[Example], Generic[RawExample, Example]):
    r"""Data source by performing transformations on another data source.
    """

    def __init__(self, data_source: DataSource[RawExample],
                 process_fn: Callable[[RawExample], Example]):
        self._source = data_source
        self._process = process_fn

    def __getitem__(self, item):
        return self._process(self._source[item])

    def __iter__(self):
        return map(self._process, iter(self._source))

    def __len__(self):
        return len(self._source)

    def __getattr__(self, item):
        return getattr(self._source, item)


class _CachedDataSource(DataSource[RawExample]):
    r"""Wrapper for random access support over a data source that does not
    implement `__getitem__`. This class is only used internally in
    :class:`~texar.torch.data.data.DatasetBase`, while conforming to user
    `cache_strategy` and `shuffle_buffer_size` settings.
    """

    _cache: Union[Dict[int, RawExample], List[RawExample]]

    def __init__(self, data_source: DataSource[RawExample],
                 erase_after_access: bool = True):
        r"""

        Args:
            data_source: The data source to wrap around.
            erase_after_access: If `True`, cached examples are erased after
                being accessed through `__getitem__`. Useful when
                :class:`~texar.torch.data.data.DatasetBase` hyperparameter
                `cache_strategy` is set to `none` or `processed`.
        """
        self._source = data_source
        self._iter = iter(data_source)
        self._max_index = -1
        self._erase_after_access = erase_after_access
        if erase_after_access:
            self._cache: Dict[int, RawExample] = {}
        else:
            self._cache: List[RawExample] = []

    def __getitem__(self, index: int) -> RawExample:
        # If specified `index` is not yet prefetched (or has already been
        # accessed), this method may throw `IndexError` or `KeyError`.
        example = self._cache[index]
        if self._erase_after_access:
            del self._cache[index]
        return example

    def __iter__(self) -> Iterator[RawExample]:
        return iter(self._source)

    def prefetch(self, index: int):
        while self._max_index < index:
            example = next(self._iter)
            self._max_index += 1
            if self._erase_after_access:
                self._cache[self._max_index] = example
            else:
                self._cache.append(example)  # type: ignore

    @property
    def max_index(self) -> int:
        return self._max_index

    def reset(self) -> None:
        if self._erase_after_access:
            self._iter = iter(self._source)
            self._max_index = -1


class DatasetBase(Dataset, Generic[RawExample, Example], ABC):
    r"""Base class inherited by all data classes.

    Args:
        source: An instance of type :class:`~texar.torch.data.DataSource`,
        hparams: A `dict` or instance of :class:`~texar.torch.HParams`
            containing hyperparameters. See :meth:`default_hparams` for the
            defaults.
        device: The device of the produced batches. For GPU training, set to
            current CUDA device.

            .. note::
                When :attr:`device` is set to a CUDA device, tensors in the
                batch will be automatically moved to the specified device. This
                may result in performance issues if your data examples contain
                complex structures (e.g., nested lists with many elements). In
                this case, it is recommended to set :attr:`device` to `None` and
                manually move your data.

                For more details, see :meth:`collate`.

    Users can also directly inherit from this class to implement customized data
    processing routines. Two methods should be implemented in the subclass:

    - :meth:`process`: Process a single data example read from the data source
      (*raw example*). Default implementation returns the raw example as is.
    - :meth:`collate`: Combine a list of processed examples into a single batch,
      and return an object of type :class:`~texar.torch.data.Batch`.

    Example:

        Here, we define a custom data class named ``MyDataset``, which is
        equivalent to the most basic usage of
        :class:`~texar.torch.data.MonoTextData`.

        .. code-block:: python

            class MyDataset(tx.data.DatasetBase):
                def __init__(self, data_path, vocab, hparams=None, device=None):
                    source = tx.data.TextLineDataSource(data_path)
                    self.vocab = vocab
                    super().__init__(source, hparams, device)

                def process(self, raw_example):
                    # `raw_example` is a data example read from `self.source`,
                    # in this case, a line of tokenized text, represented as a
                    # list of `str`.
                    return {
                        "text": raw_example,
                        "ids": self.vocab.map_tokens_to_ids_py(raw_example),
                    }

                def collate(self, examples):
                    # `examples` is a list of objects returned from the
                    # `process` method. These data examples should be collated
                    # into a batch.

                    # `text` is a list of list of `str`, storing the tokenized
                    # sentences for each example in the batch.
                    text = [ex["text"] for ex in examples]
                    # `ids` is the NumPy tensor built from the token IDs of each
                    # sentence, and `lengths` the lengths of each sentence.
                    # The `tx.data.padded_batch` function pads IDs to the same
                    # length and then stack them together. This function is
                    # commonly used in `collate` methods.
                    ids, lengths = tx.data.padded_batch(
                        [ex["ids"] for ex in examples])
                    return tx.data.Batch(
                        len(examples),
                        text=text,
                        text_ids=torch.from_numpy(ids),
                        lengths=torch.tensor(lengths))

            vocab = tx.data.Vocab("vocab.txt")
            hparams = {'batch_size': 1}
            data = MyDataset("data.txt", vocab, hparams)
            iterator = DataIterator(data)
            for batch in iterator:
                # batch contains the following
                # batch_ == {
                #    'text': [['<BOS>', 'example', 'sequence', '<EOS>']],
                #    'text_ids': [[1, 5, 10, 2]],
                #    'length': [4]
                # }
    """

    # pylint: disable=line-too-long

    # The `DatasetBase` is used in combination with Texar `DataIterator`, which internally uses the PyTorch `DataLoader`
    # for multi-processing support.
    #
    # We divide the entire data pipeline into three stages, namely *load*, *process*, and *batch*:
    # - **Load** refers to loading data from the data source (e.g., a file, a Python list or iterator). In Texar,
    #   loading is handled by `DataSource` classes.
    # - **Process** refers to preprocessing routines for each data example (e.g., vocabulary mapping, tokenization). In
    #   Texar, this is the `process` function of each `DatasetBase` class.
    # - **Batch** refers to combining multiple examples to form a batch, which typically includes padding and moving
    #   data across devices. In Texar, this is the `collate` function of each `DatasetBase` class.
    #
    # PyTorch DataLoader only performs batching, and since multi-processing is used, the entire dataset is expected to
    # be in memory before iteration, i.e. loading and processing cannot be lazy. The `DatasetBase` class is carefully
    # crafted to provide laziness and caching options at all possible stages.
    #
    # To support laziness, we pass data examples (either raw or processed, depending on whether processing is lazy) to
    # the worker processes. To prevent modifying the underlying `DataLoader` implementation, we hack the PyTorch
    # `Sampler` classes (responsible for sampling the next data example from the dataset, and returning its index) to
    # also return data examples. To support caching, the worker may also need to return the processed examples through
    # pipes.
    #
    # The following table describes the intended behavior of each combination of lazy/caching modes, and the exact
    # behaviors of the sampler and workers. `<X>` means the mode combination does not make sense (e.g. with `Lazy.None`,
    # processed data examples are effectively cached, so `Cache.None` makes no sense). Parts in `*[blah]*` hold true
    # only for the first epoch.
    #
    # +---------------+-------------------------------+-------------------------------+-------------------------------+
    # |               | Cache.None                    | Cache.Loaded                  | Cache.Processed               |
    # |               | no caching                    | only cache loaded examples    | only cache processed examples |
    # +===============+===============================+===============================+===============================+
    # | Lazy.None     | <X>                           | <X>                           | Sampler returns indices.      |
    # | eager load,   |                               |                               | Worker only does batching.    |
    # | eager process |                               |                               | Worker returns batch.         |
    # +---------------+-------------------------------+-------------------------------+-------------------------------+
    # | Lazy.Process  | <X>                           | Sampler returns indices.      | Sampler returns indices.      |
    # | eager load,   |                               | Worker does batching and      | Worker does batching          |
    # | lazy process  |                               |   processing.                 |   *[and processing]*.         |
    # |               |                               | Worker returns batch.         | Worker returns batch          |
    # |               |                               |                               |   *[and processed examples]*. |
    # +---------------+-------------------------------+-------------------------------+-------------------------------+
    # | Lazy.All      | Sampler returns indices and   | Sampler returns indices       | Sampler returns indices       |
    # | lazy load,    |   data examples.              |   *[and data examples]*.      |   *[and data examples]*.      |
    # | lazy process  | Worker does batching and      | Worker does batching and      | Worker does batching          |
    # |               |   processing.                 |   processing.                 |   *[and processing]*.         |
    # |               | Worker returns batch.         | Worker returns batch.         | Worker returns batch          |
    # |               |                               |                               |   *[and processed examples]*. |
    # +---------------+-------------------------------+-------------------------------+-------------------------------+
    #
    # Note that in the above table we assume `parallelize_processing` to be True. In rare cases this may not be desired,
    # for instance, when `process` depends on some shared variable that must be modified during iteration, e.g. a
    # vocabulary constructed on-the-fly. When `parallelize_processing` is False, behaviors are as the following (much
    # simpler) table. Although, note that compared to the above cases, this often results in worse performance.
    #
    # +---------------+-------------------------------+-------------------------------+-------------------------------+
    # |               | Cache.None                    | Cache.Loaded                  | Cache.Processed               |
    # |               | no caching                    | only cache loaded examples    | only cache processed examples |
    # +===============+===============================+===============================+===============================+
    # | Lazy.None     | <X>                           | <X>                           | Sampler returns indices.      |
    # | eager load,   |                               |                               | Worker only does batching.    |
    # | eager process |                               |                               | Worker returns batch.         |
    # +---------------+-------------------------------+-------------------------------+-------------------------------+
    # | Lazy.Process  | <X>                           | Sampler returns indices and processed examples.               |
    # | eager load,   |                               | Worker only does batching.                                    |
    # | lazy process  |                               | Worker returns batch.                                         |
    # +---------------+-------------------------------+---------------------------------------------------------------+
    # | Lazy.All      | Sampler returns indices and processed examples.                                               |
    # | lazy load,    | Worker only does batching.                                                                    |
    # | lazy process  | Worker returns batch.                                                                         |
    # +---------------+-----------------------------------------------------------------------------------------------+

    # pylint: enable=line-too-long

    _source: DataSource[RawExample]
    _dataset_size: Optional[int]

    def __init__(self, source: DataSource[RawExample], hparams=None,
                 device: Optional[torch.device] = None):
        self._source = source
        self._hparams = HParams(hparams, self.default_hparams())
        self.device = device

        if self._hparams.num_epochs != 1:
            warnings.warn(f"'num_epochs' is set to {self._hparams.num_epochs}, "
                          f"but will be treated as 1.")

        # Check and convert strategy hyperparameters.
        self._lazy_strategy = _LazyStrategy(self._hparams.lazy_strategy)
        self._cache_strategy = _CacheStrategy(self._hparams.cache_strategy)
        if self._lazy_strategy is _LazyStrategy.NONE:
            if self._cache_strategy is not _CacheStrategy.PROCESSED:
                warnings.warn(
                    f"Using '{self._cache_strategy}' cache strategy with "
                    f"'none' lazy strategy. This will be equivalent to "
                    f"'processed' cache strategy.")
            self._cache_strategy = _CacheStrategy.PROCESSED
        elif self._lazy_strategy is _LazyStrategy.PROCESS:
            if self._cache_strategy is _CacheStrategy.NONE:
                warnings.warn(
                    f"Using 'none' cache strategy with 'process' lazy "
                    f"strategy. This will be equivalent to 'loaded' cache "
                    f"strategy.")
                self._cache_strategy = _CacheStrategy.LOADED
        self._uses_multi_processing = self._hparams.num_parallel_calls > 0
        self._parallelize_processing = self._hparams.parallelize_processing

        self._processed_cache: List[Example] = []
        self._fully_cached = False

        # If specified maximum dataset size, wrap the data source. This is done
        # before caching to avoid caching excess elements.
        if self._hparams.max_dataset_size != -1:
            self._source = _TruncatedDataSource[RawExample](
                self._source, self._hparams.max_dataset_size)

        # If processing should not be parallelized, combine processing with
        # loading by wrapping the data source. In this case, **processed** data
        # will be cached.
        if (not self._parallelize_processing and
                self._lazy_strategy is _LazyStrategy.ALL and
                self._cache_strategy is not _CacheStrategy.LOADED):
            self._transformed_source = _TransformedDataSource[
                RawExample, Example](self._source, self.process)
            self._source = self._transformed_source  # type: ignore

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
                self._cached_source = _CachedDataSource[RawExample](
                    self._source, erase_after_access)
                self._source = self._cached_source
                self._dataset_size = None

        # If processing should not be parallelized, combine processing with
        # loading by wrapping the data source. In this case, **loaded** data
        # will be cached.
        if (not self._parallelize_processing and
                self._cache_strategy is _CacheStrategy.LOADED):
            self._transformed_source = _TransformedDataSource[
                RawExample, Example](self._source, self.process)
            self._source = self._transformed_source  # type: ignore

        # Simplify some logic-heavy checks.
        self.__should_return_processed_examples = (
                self._lazy_strategy is not _LazyStrategy.NONE and
                self._cache_strategy is _CacheStrategy.PROCESSED and
                self._parallelize_processing)
        self.__should_call_prefetch_source = (
                self._lazy_strategy is _LazyStrategy.ALL and
                self._cache_strategy is _CacheStrategy.NONE)
        self.__should_call_prefetch_processed = (
                not self._parallelize_processing and
                self._lazy_strategy is _LazyStrategy.PROCESS and
                self._cache_strategy is _CacheStrategy.PROCESSED)
        self.__should_delete_source_in_add_cache = (
                not self._supports_random_access and
                self._parallelize_processing and
                self._uses_multi_processing and
                self._lazy_strategy is _LazyStrategy.PROCESS and
                self._cache_strategy is _CacheStrategy.PROCESSED)

        # Perform eager loading/processing if required.
        if self._lazy_strategy is _LazyStrategy.NONE:
            # Process entire dataset and cache.
            self._processed_cache = [self.process(raw_example)
                                     for raw_example in self._source]
            self._dataset_size = len(self._processed_cache)
            self._fully_cached = True
        else:
            if self._lazy_strategy is _LazyStrategy.PROCESS:
                # Load entire dataset. Note that if data source supports random
                # access, we assume it is already loaded into memory.
                if not self._supports_random_access:
                    self._prefetch_all_source()

            if self._cache_strategy is _CacheStrategy.PROCESSED:
                # Data can be processed in arbitrary order, so they need to be
                # reordered before storing in the cache list.
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
                "lazy_strategy": 'none',
                "cache_strategy": 'processed',
                "parallelize_processing": True,
                "name": "data"
            }

        Here:

        `"num_epochs"`: int
            Number of times the dataset should be repeated.

            .. note::
                This option only exists for compatibility, and will be
                ignored. A warning will be generated is any value other than
                1 is used.

        `"batch_size"`: int
            Batch size, i.e., the number of consecutive elements of the
            dataset to combine in a single batch.

        `"allow_smaller_final_batch"`: bool
           Whether to allow the final batch to be smaller if there are
           insufficient elements left. If `False`, the final batch is
           discarded if it is smaller than batch size. Note that,
           if `True`, `output_shapes` of the resulting dataset
           will have a a **static** batch_size dimension equal to
           "batch_size".

        `"shuffle"`: bool
            Whether to randomly shuffle the elements of the dataset.

        `"shuffle_buffer_size"`: int
            The buffer size for data shuffling. The larger, the better
            the resulting data is mixed.

            If `None` (default), buffer size is set to the size of the
            whole dataset (i.e., make the shuffling the maximally
            effective).

        `"shard_and_shuffle"`: bool
            Whether to first shard the dataset and then shuffle each
            block respectively. Useful when the whole data is too large to
            be loaded efficiently into the memory.

            If `True`, :attr:`shuffle_buffer_size` must be specified to
            determine the size of each shard.

            .. warning::
                Sharding is not yet supported. This option will be ignored.

        `"num_parallel_calls"`: int
            Number of elements from the datasets to process in parallel.
            When ``"num_parallel_calls"`` equals 0, no worker processes will
            be created; when the value is greater than 0, the number of worker
            processes will be equal to ``"num_parallel_calls"``.

        `"prefetch_buffer_size"`: int
            The maximum number of elements that will be buffered when
            prefetching.

            .. note::
                This option exists only for compatibility. Currently data
                is only prefetched when ``"num_parallel_calls"`` is greater
                than 1, and the number of examples to prefetch is controlled
                internally by PyTorch :torch_docs:`DataLoader
                <data.html#torch.utils.data.DataLoader>`.

        `"max_dataset_size"`: int
            Maximum number of instances to include in
            the dataset. If set to `-1` or greater than the size of
            dataset, all instances will be included. This constraint is
            imposed after data shuffling and filtering.

        `"seed"`: int, optional
            The random seed for shuffle.

            Note that if a seed is set, the shuffle order will be exact
            the same every time when going through the (repeated) dataset.

            .. warning::
                Manual seeding is not yet supported. This option will be
                ignored.

        `"lazy_strategy"`: str
            Lazy strategy for data examples. Lazy loading/processing defers
            data loading/processing until when it's being accessed.
            Non-lazy (eager) loading/processing would load/process all data
            upon construction of dataset. Available options are:

            - `none`: Perform eager loading and processing.
            - `process`: Perform eager loading and lazy processing.
            - `all`: Perform lazy loading and processing.

            Defaults to `all`. Note that currently, all eager operations
            are performed on a single process only.

        `"cache_strategy"`: str
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

        `"parallelize_processing"`: bool
            Whether to perform parallelized processing of data. Since
            multi-processing parallelism is utilized, this flag should be
            `False` if your process routine involves modifying a shared
            object across examples.

            Note that this only affects cases where `lazy_strategy` is not
            `none`. If `lazy_strategy` is `none`, processing will be
            performed on a single process regardless of this value.

        `"name"`: str
            Name of the data.
        """
        # TODO: Sharding not yet supported.
        # TODO: `seed` is not yet applied.
        # TODO: `prefetch_buffer_size` will not be supported, but could remain
        #   for compatibility.
        return {
            "name": "data",
            "num_epochs": 1,
            "batch_size": 64,
            "allow_smaller_final_batch": True,
            "shuffle": True,
            "shuffle_buffer_size": None,
            "shard_and_shuffle": False,
            "num_parallel_calls": 0,
            "prefetch_buffer_size": 0,
            "max_dataset_size": -1,
            "seed": None,
            "lazy_strategy": 'none',
            "cache_strategy": 'processed',
            "parallelize_processing": True,
        }

    def to(self, device: Optional[torch.device]):
        r"""Move the dataset to the specific device. Note that we don't actually
        move data or do anything here -- data will be moved to the appropriate
        device after :class:`~texar.torch.data.DataIterator` fetches the batch.
        """
        if device is not None:
            self.device = device
        return self

    def _prefetch_processed(self, index: int):
        r"""Performs processing on the main process. This is called in
        :meth:`texar.torch.data.data.DatasetBase._prefetch_source` if
        `parallelize_processing` is `False`."""
        if len(self._processed_cache) <= index:
            self._processed_cache.extend(
                self.process(self._source[x])
                for x in range(len(self._processed_cache), index + 1))
            if len(self._processed_cache) == self._dataset_size:
                self._fully_cached = True

    def _prefetch_all_source(self) -> int:
        r"""Prefetches all examples from data source. This is only called if
        `__len__` is called before dataset size can be determined, or when using
        eager loading.
        """

        try:
            max_index = 10 ** 8
            self._cached_source.prefetch(max_index)
            warnings.warn(
                f"The data source contains more than {max_index:.2e} "
                f"examples. Please check whether it is infinite.")
            while True:
                max_index *= 2
                self._cached_source.prefetch(max_index)
        except StopIteration:
            self._dataset_size = self._cached_source.max_index + 1
            return self._dataset_size

    def _prefetch_source(self, index: int) -> Optional[int]:
        r"""Prefetches data so `__getitem__` will be available. This method
        should only be called in the main process, because data sources are not
        guaranteed to be thread-safe.

        Args:
            index: Prefetch data up to this index.

        Returns:
            If `index` is greater than dataset size, returns the inferred
            dataset size. Otherwise, returns `None`.
        """
        if not self._supports_random_access:
            try:
                self._cached_source.prefetch(index)
            except StopIteration:
                self._dataset_size = self._cached_source.max_index + 1
                # self._cached_source.reset()
                if self._should_call_prefetch_processed:
                    self._prefetch_processed(self._dataset_size - 1)
                return self._dataset_size
            if self._should_call_prefetch_processed:
                self._prefetch_processed(index)
        else:
            # Dataset size must be known.
            if index >= self._dataset_size:  # type: ignore
                return self._dataset_size
        return None

    def __len__(self) -> int:
        if self._dataset_size is None:
            raise TypeError(
                "__len__ not supported for datasets with undetermined size")
        return self._dataset_size

    def process(self, raw_example: RawExample) -> Example:
        r"""The process routine. A default implementation of no-op is provided,
        but subclasses are free to override this behavior.

        The process routine would take raw examples loaded from the data source
        as input, and return processed examples. If `parallelize_processing`
        is `True`, this method **must not** access shared variables that are
        modified during iterator (e.g., constructing vocabularies on-the-fly).

        Args:
            raw_example: The raw example loaded from data.

        Returns:
            The processed example.
        """
        return raw_example  # type: ignore

    def __getitem__(self, index: Union[int, Tuple[int, RawExample]]) -> Example:
        if isinstance(index, int):
            if self._fully_cached:
                return self._processed_cache[index]
            elif not self._parallelize_processing:
                return self._transformed_source[index]
            else:
                return self.process(self._source[index])
        else:
            # `index` is a tuple of (index, example).
            if not self._parallelize_processing:
                return index[1]  # type: ignore
            else:
                return self.process(index[1])

    def _add_cached_examples(self, indices: List[int], examples: List[Example]):
        r"""Called by :class:`texar.torch.data.data._CacheDataLoaderIter` to
        cache examples processed in worker processes.

        Args:
            indices: Indices for each example.
            examples: The examples processed in worker processes.
        """
        if self._should_delete_source_in_add_cache:
            # In this case, `_CachedDataSource.__getitem__` will be
            # called on worker processes, so the cache cannot be
            # deleted. Thus, we move deletion to
            # `_add_cached_examples`.
            for index in indices:
                del self._cached_source._cache[index]  # pylint: disable=protected-access
        for index, example in zip(indices, examples):
            if index == len(self._processed_cache):
                self._processed_cache.append(example)
            else:
                self._reorder_cache[index] = example

        while len(self._processed_cache) in self._reorder_cache:
            index = len(self._processed_cache)
            self._processed_cache.append(self._reorder_cache[index])
            del self._reorder_cache[index]
        if len(self._processed_cache) == self._dataset_size:
            self._fully_cached = True

    def _start_iteration(self) -> None:
        r"""Called by :class:`texar.torch.data.data.SamplerBase` before a new
        round of iteration starts. Note that this method will only be called if
        an unknown-sized iterator is used.
        """
        if not self._supports_random_access:
            self._cached_source.reset()

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
        r"""A :class:`~texar.torch.HParams` instance of the
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

    def collate(self, examples: List[Example]) -> Batch:
        r"""The collate routine. Subclasses must implement this method.

        The collate routine is called to collate (combine) examples into
        batches. This function takes a list of processed examples, and returns
        an instance of :class:`~texar.torch.data.Batch`.

        .. note::
            Implementation should make sure that the returned callable is safe
            and efficient under multi-processing scenarios. Basically, do not
            rely on variables that could be modified during iteration, and avoid
            accessing unnecessary variables, as each access would result in a
            cross-process memory copy.

        .. warning::
            The recommended pattern is not to move tensor storage within this
            method, but you are free to do so.

            However, if multiple workers are used
            (:attr:`num_parallel_calls` > 0), moving tensors to CUDA devices
            within this method would result in CUDA errors being thrown.

        Args:
            examples: A list of processed examples in a batch.

        Returns:
            The collated batch.
        """
        raise NotImplementedError

    def _collate_and_maybe_return(self, examples: List[Example]) -> \
            Union[Batch, Tuple[List[Example], Batch]]:
        r"""Called by :class:`~texar.torch.data.DataIterator` to obtain the
        collated batch (and processed examples under certain circumstances).

        Args:
            examples: A list of processed examples in a batch.

        Returns:
            The collated batch.
        """
        batch = self.collate(examples)
        if self._should_return_processed_examples:
            return examples, batch
        return batch

    @property
    def _should_return_processed_examples(self):
        r"""Returns `True` if the worker threads should perform processing and
        return the processed examples.
        """
        return (not self._fully_cached and
                self.__should_return_processed_examples)

    @property
    def _should_yield_raw_example(self):
        r"""Returns `True` if the sampler should yield raw examples.
        """
        return (self._lazy_strategy is _LazyStrategy.ALL and
                (self._cache_strategy is _CacheStrategy.NONE or
                 not self._fully_cached))

    @property
    def _should_call_prefetch_source(self):
        r"""Returns `True` if the sampler should call `_prefetch_source`.
        """
        return (self._dataset_size is None or
                self.__should_call_prefetch_source)

    @property
    def _should_call_prefetch_processed(self):
        r"""Returns `True` if `_prefetch_source` should call
        `_prefetch_processed`.
        """
        return self.__should_call_prefetch_processed

    @property
    def _should_delete_source_in_add_cache(self):
        r"""Returns `True` if `_add_cached_examples` should delete cached raw
        examples.
        """
        return self.__should_delete_source_in_add_cache
