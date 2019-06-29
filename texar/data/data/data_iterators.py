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
Various data iterator classes.
"""

# pylint: disable=protected-access

from typing import (
    Any, Callable, Dict, Generic, Iterable, Iterator, List, Mapping, Optional,
    Sequence, Tuple, TypeVar, Union)

from torch.utils.data import DataLoader
from torch.utils.data import sampler as torch_sampler
from torch.utils.data.dataloader import _DataLoaderIter as torch_DataLoaderIter

import torch
from texar.data.data.data_base import DataBase
from texar.data.data.dataset_utils import Batch
from texar.utils.types import MaybeSeq
from texar.utils.utils import ceildiv

__all__ = [
    "DataIterator",
    "TrainTestDataIterator",
    "BatchingStrategy",
    "TokenCountBatchingStrategy",
]

DatasetsType = Union[Dict[str, DataBase], MaybeSeq[DataBase]]
Example = TypeVar('Example')


# pylint: disable=attribute-defined-outside-init
# TODO: Remove this when Pylint fixes the bug. If the `disable` directive is not
#  added, Pylint incorrectly reports this error for `self.size` in subclasses of
#  `SamplerBase` in Python 3.6 due to use of the Generic class.
#  See Pylint issue: https://github.com/PyCQA/pylint/issues/2981

class SamplerBase(torch_sampler.Sampler, Generic[Example]):
    r"""A subclass of :torch_docs:`~torch.utils.data.Sampler
    <data.html#torch.utils.data.Sampler>` that supports:

    - Returning raw examples when required.
    - Creating iterators with unknown dataset size.

    This class is used internally in :class:`~texar.data.data.DataIterator`.
    It calls the :meth:`~texar.data.data.DataBase._prefetch_source` method to
    ensure the required number of raw examples are prefetched from source.

    Args:
        data: The :class:`~texar.data.data.DataBase` instance.
    """
    size: Optional[int]

    def __init__(self, data: DataBase[Any, Example]):
        super().__init__(data)

        self._data = data
        self.size = None

    def _iterator_given_size(self, size: int) -> Iterator[int]:
        r"""Return an iterator that generates samples when the dataset size
        is given.

        Args:
            size: The dataset size.
        """
        raise NotImplementedError

    def _iterator_unknown_size(self) -> Iterator[int]:
        r"""Return an iterator that generates samples when the dataset size
        is unknown. This iterator must also call
        :meth:`texar.data.data.DataBase._prefetch_source` and check whether
        the dataset size can be determined, before yielding the index. See
        example implementations for details.
        """
        raise NotImplementedError

    def __iter__(self) -> Union[Iterator[int], Iterator[Tuple[int, Example]]]:
        r"""Return an iterator based on the dataset settings.
        """
        self.size = self._data._dataset_size
        if (not self._data._fully_cached or
                self._data._should_call_prefetch_source):
            self._data._start_iteration()
            # First epoch of lazy loading, calling prefetch, and returning
            # indices and examples.
            iterator = self._iterator_unknown_size()
        else:
            # Non-lazy loading, or when dataset has been fully iterated.
            assert self.size is not None
            iterator = self._iterator_given_size(self.size)

        if self._data._should_call_prefetch_processed:
            # Processing routine is performed in main process. Yield
            # processed examples instead.
            map_fn = lambda idx: (idx, self._data._processed_cache[idx])
        elif self._data._should_yield_raw_example:
            # Return indices and examples for any epoch in this case.
            map_fn = lambda idx: (idx, self._data._source[idx])
        else:
            map_fn = None  # type: ignore
        if map_fn is not None:
            return map(map_fn, iterator)

        return iterator

    def __len__(self):
        if self.size is not None:
            return self.size
        raise AttributeError("Dataset size cannot be determined at this point")


class SequentialSampler(SamplerBase[Example]):
    r"""Samples elements sequentially, always in the same order. Same as
    :torch_docs:`~torch.utils.data.SequentialSampler
    <data.html#torch.utils.data.SequentialSampler>`
    """

    def _iterator_given_size(self, size: int) -> Iterator[int]:  # pylint: disable=no-self-use
        return iter(range(size))

    def _iterator_unknown_size(self) -> Iterator[int]:
        index = 0
        while True:
            cur_size = self._data._prefetch_source(index)
            if cur_size is not None:
                self.size = cur_size
                break
            yield index
            index += 1


class RandomSampler(SamplerBase[Example]):
    r"""Samples elements randomly. If without replacement, then sample from a
    shuffled dataset. If with replacement, then user can specify ``num_samples``
    to draw.

    This class uses :torch_docs:`torch.utils.data.RandomSampler
    <data.html#torch.utils.data.RandomSampler>` directly. Given the
    nature of such shuffling, it cannot be used for iterators with unknown size.

    Args:
        data: The :class:`~texar.data.data.DataBase` instance.
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if `True`,
            default=False
    """

    def __init__(self, data: DataBase[Any, Example], replacement: bool = False,
                 num_samples: Optional[int] = None):
        super().__init__(data)
        self._sampler = torch_sampler.RandomSampler(
            data, replacement, num_samples)

    def _iterator_given_size(self, size: int) -> Iterator[int]:
        del size  # not used
        return iter(self._sampler)

    def _iterator_unknown_size(self) -> Iterator[int]:  # pylint: disable=no-self-use
        raise TypeError(
            "RandomSampler does not support lazy data loading. To perform "
            "shuffling with lazy loading, use BufferShuffleSampler.")


class BufferShuffleSampler(SamplerBase[Example]):
    r"""A :torch_docs:`~torch.utils.data.Sampler
    <data.html#torch.utils.data.Sampler>` that uses a shuffle buffer, as
    in TensorFlow. The buffer is first filled with data examples. Each time a
    sample is drawn from the buffer, and the drawn sample is replaced with the
    next data example.

    This class is used internally in :class:`~texar.data.data.DataIterator`.
    It calls the :meth:`~texar.data.data.DataBase._prefetch_source` method to
    ensure the required number of

    Args:
        data: The :class:`~texar.data.data.DataBase` instance.
        buffer_size: The size of the shuffle buffer. Use larger buffer sizes for
            more uniformly-random shuffling.
    """

    def __init__(self, data: DataBase[Any, Example], buffer_size: int):
        super().__init__(data)
        self.buffer_size = buffer_size

    def _iterator_given_size(self, size) -> Iterator[int]:
        if self.buffer_size >= size:
            return iter(torch.randperm(size).tolist())

        buffer = list(range(self.buffer_size))
        for x in range(self.buffer_size, size):
            sample = torch.randint(self.buffer_size, (1,)).item()
            index = buffer[sample]
            yield index
            buffer[sample] = x
        yield from (buffer[x] for x in torch.randperm(self.buffer_size))

    def _iterator_unknown_size(self) -> Iterator[int]:
        buffer = list(range(self.buffer_size))
        x = self.buffer_size
        while True:
            sample = torch.randint(self.buffer_size, (1,)).item()
            index = buffer[sample]
            cur_size = self._data._prefetch_source(index)
            if cur_size is not None and index >= cur_size:
                self.size = cur_size
            if self.size is not None and index >= self.size:
                break
            yield index
            buffer[sample] = x
            x += 1
        yield from (buffer[x] for x in torch.randperm(self.buffer_size)
                    if buffer[x] < self.size)


# pylint: enable=attribute-defined-outside-init


class BatchingStrategy(Generic[Example]):
    r"""Decides batch boundaries in dynamic batching. Please refer to
    :class:`TokenCountBatchingStrategy` for a concrete example.
    """

    def reset_batch(self) -> None:
        r"""Reset the internal state of the batching strategy. This method is
        called at the start of iteration, and after each batch is yielded.
        """
        raise NotImplementedError

    def add_example(self, example: Example) -> bool:
        r"""Add an example into the current batch, and modify internal states
        accordingly. If the example should not be added to the batch, this
        method does not modify the internal state, and returns `False`.

        Args:
            example: The example to add to the batch.

        Returns:
            A boolean value indicating whether :attr:`example` should be added
            to the batch.
        """
        raise NotImplementedError


class TokenCountBatchingStrategy(BatchingStrategy[Example]):
    r"""Create dynamically-sized batches so that the total number of tokens
    inside each batch is constrained.

    Args:
        max_tokens (int): The maximum number of tokens inside each batch.
        max_batch_size (int, optional): The maximum number of examples for each
            batch. If `None`, batches can contain arbitrary number of examples
            as long as the total number of tokens does not exceed
            :attr:`max_tokens`.
        length_fn (callable, optional): A function taking a data example as
            argument, and returning the number of tokens in the example. By
            default, :python:`len` is used, which is the desired behavior if the
            dataset in question is a :class:`~texar.data.MonoTextData`.
    """

    def __init__(self, max_tokens: int, max_batch_size: Optional[int] = None,
                 length_fn: Optional[Callable[[Example], int]] = None):
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.length_fn: Callable[[Example], int]
        self.length_fn = length_fn or len  # type: ignore
        self.sum_tokens = 0
        self.cur_batch_size = 0

    def reset_batch(self) -> None:
        self.sum_tokens = 0
        self.cur_batch_size = 0

    def add_example(self, example: Example) -> bool:
        if self.cur_batch_size == self.max_batch_size:
            return False
        cur_tokens = self.length_fn(example)
        if cur_tokens + self.sum_tokens > self.max_tokens:
            return False

        self.cur_batch_size += 1
        self.sum_tokens += cur_tokens
        return True


class DynamicBatchSampler(torch_sampler.BatchSampler, Generic[Example]):
    r"""A subclass of :torch_docs:`~torch.utils.data.BatchSampler
    <data.html#torch.utils.data.BatchSampler>` that supports dynamic batching
    through a user-provided :class:`BatchingStrategy`. This class is used
    internally.

    Args:
        dataset: The dataset to create batches from.
        sampler: An instance of :class:`SamplerBase` that returns indices of
            each sampled example.
        strategy: An instance of :class:`BatchingStrategy` that decides whether
            a batch should be yielded.
    """

    def __init__(self, dataset: DataBase[Any, Example],  # pylint: disable=super-init-not-called
                 sampler: SamplerBase, strategy: BatchingStrategy[Example]):
        self.dataset = dataset
        self.sampler = sampler
        self.strategy = strategy

    def __iter__(self) -> Union[Iterator[List[int]],  # type: ignore
                                Iterator[List[Tuple[int, Example]]]]:
        batch = []  # type: ignore
        self.strategy.reset_batch()
        for idx in self.sampler:
            if isinstance(idx, tuple):
                example = self.dataset[idx[0]]
            else:
                example = self.dataset[idx]
            while not self.strategy.add_example(example):
                if len(batch) == 0:
                    raise ValueError(f"Batching strategy refused to add "
                                     f"example {idx} to empty batch.")
                yield batch
                batch = []
                self.strategy.reset_batch()
            batch.append(idx)
        if len(batch) > 0:
            yield batch
            self.strategy.reset_batch()

    def __len__(self):
        raise TypeError("DynamicBatchSampler does not support __len__")


class _DataLoaderIter(torch_DataLoaderIter):  # pylint: disable=abstract-method
    r"""Iterates once over the DataLoader's dataset. This is almost identical
    to PyTorch :class:`torch.utils.data.dataloader._DataLoaderIter`, except
    that we check `allow_smaller_final_batch` here. This is because using
    `drop_last` in :class:`~torch.utils.data.sampler.BatchSampler` would cause
    the dataset to not load/process/cache certain elements from the final batch,
    which complicates the already complex logic.
    """

    def __init__(self, loader: DataLoader):
        self._batch_size = loader.batch_size
        super().__init__(loader)

    def __next__(self):
        batch = super().__next__()
        # Drop smaller final batch according to settings. Note that
        # `_batch_size` could be None if dynamic batching is used.
        if (self._batch_size is not None and
                batch.batch_size < self._batch_size and
                not self.dataset.hparams.allow_smaller_final_batch):
            raise StopIteration
        return batch


class _CacheDataLoaderIter(torch_DataLoaderIter):  # pylint: disable=abstract-method
    r"""Iterates once over the DataLoader's dataset. This class is used when
    examples are processed and returned by worker processes. We need to record
    the corresponding indices of each batch, call
    :meth:`texar.data.data.DataBase._add_cached_examples` to cache the processed
    examples, and return only the :class:`texar.data.data.Batch` instance to
    the user.
    """
    dataset: DataBase

    def __init__(self, loader: DataLoader):
        self._indices_dict: Dict[int, List[int]] = {}
        self._batch_size = loader.batch_size
        super().__init__(loader)

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        if self.dataset._should_yield_raw_example:
            indices = [index[0] for index in indices]
        self._indices_dict[self.send_idx] = indices
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        batch = super()._process_next_batch(batch)
        indices = self._indices_dict[self.rcvd_idx - 1]
        del self._indices_dict[self.rcvd_idx - 1]
        examples, batch = batch
        self.dataset._add_cached_examples(indices, examples)
        return batch

    @staticmethod
    def pin_memory_batch(batch):
        if isinstance(batch, torch.Tensor):
            return batch.pin_memory()
        elif isinstance(batch, (str, bytes)):
            return batch
        elif isinstance(batch, Mapping):
            return {k: _CacheDataLoaderIter.pin_memory_batch(sample)
                    for k, sample in batch.items()}
        elif isinstance(batch, Sequence):
            return [_CacheDataLoaderIter.pin_memory_batch(sample)
                    for sample in batch]
        else:
            return batch

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:
                batch = self.pin_memory_batch(batch)
            if self.dataset._should_yield_raw_example:
                indices = [index[0] for index in indices]
            examples, batch = batch
            self.dataset._add_cached_examples(indices, examples)
        else:
            batch = super().__next__()
        if (self._batch_size is not None and
                batch.batch_size < self.dataset.batch_size and
                not self.dataset.hparams.allow_smaller_final_batch):
            raise StopIteration
        return batch


class SingleDatasetIterator(DataLoader):
    r"""Iterator for a single dataset. This iterator is based on the PyTorch
    :class:`~torch.utils.data.DataLoader` interface, with a custom shuffling
    routine. This class is used internally.

    Args:
        dataset: The dataset to iterator through. The dataset must be an
            instance of :class:`texar.data.DataBase`, because configurations are
            read from the dataset `HParams`.
        batching_strategy: The batching strategy to use when performing dynamic
            batching. If `None`, fixed-sized batching is used.
    """
    dataset: DataBase

    def __init__(self, dataset: DataBase,
                 batching_strategy: Optional[BatchingStrategy] = None):
        shuffle = dataset.hparams.shuffle
        shuffle_buffer_size = dataset.hparams.shuffle_buffer_size
        sampler: SamplerBase
        if shuffle and shuffle_buffer_size is not None:
            sampler = BufferShuffleSampler(dataset, shuffle_buffer_size)
        elif shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        num_parallel_calls = dataset.hparams.num_parallel_calls
        collate_fn = dataset._collate_and_maybe_return
        num_workers = (0 if num_parallel_calls == 1 else num_parallel_calls)

        if batching_strategy is not None:
            batch_sampler = DynamicBatchSampler(
                dataset, sampler, batching_strategy)
            super().__init__(
                dataset, batch_sampler=batch_sampler,
                collate_fn=collate_fn, num_workers=num_workers)
        else:
            super().__init__(
                dataset, batch_size=dataset.batch_size, drop_last=False,
                sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)

    def __iter__(self):
        if self.dataset._should_return_processed_examples:
            # Accepts processed examples from workers and add to dataset cache.
            return _CacheDataLoaderIter(self)
        else:
            return _DataLoaderIter(self)

    def __len__(self):
        if self.batch_size is None:
            raise TypeError("__len__ not supported for dynamic batching")
        data_length = len(self.dataset)  # may throw TypeError
        if self.dataset.hparams.allow_smaller_final_batch:
            return ceildiv(data_length, self.batch_size)
        return data_length // self.batch_size


class DataIterator:
    r"""Data iterator that switches and iterates through multiple datasets.

    This is a wrapper of :class:`~texar.data.SingleDatasetIterator`.

    Args:
        datasets: Datasets to iterate through. This can be:

            - A single instance of :class:`~texar.data.DataBase`.
            - A `dict` that maps dataset name to instances of
              :class:`~texar.data.DataBase`.
            - A `list` of instances of :class:`texar.data.DataBase`. The name
              of instances (:attr:`texar.data.DataBase.name`) must be unique.

        batching_strategy: The batching strategy to use when performing dynamic
            batching. If `None`, fixed-sized batching is used.

    Example:

        Create an iterator over two datasets and generating fixed-sized batches:

        .. code-block:: python

            train_data = MonoTextData(hparams_train)
            test_data = MonoTextData(hparams_test)
            iterator = DataIterator({'train': train_data, 'test': test_data})

            for epoch in range(200): # Run 200 epochs of train/test
                # Starts iterating through training data from the beginning.
                iterator.switch_to_dataset('train')
                for batch in iterator:
                    ... # Do training with the batch.

                # Starts iterating through test data from the beginning
                for batch in iterator.get_iterator('test'):
                    ... # Do testing with the batch.

        Dynamic batching based on total number of tokens:

        .. code-block:: python

            iterator = DataIterator(
                {'train': train_data, 'test': test_data},
                batching_strategy=TokenCountBatchingStrategy(max_tokens=1000))

        Dynamic batching with custom strategy (e.g. total number of tokens in
        examples from :class:`~texar.data.PairedTextData`, including padding):

        .. code-block:: python

            class CustomBatchingStrategy(BatchingStrategy):
                def __init__(self, max_tokens: int):
                    self.max_tokens = max_tokens
                    self.reset_batch()

                def reset_batch(self) -> None:
                    self.max_src_len = 0
                    self.max_tgt_len = 0
                    self.cur_batch_size = 0

                def add_example(self, ex: Tuple[List[str], List[str]]) -> bool:
                    max_src_len = max(self.max_src_len, len(ex[0]))
                    max_tgt_len = max(self.max_tgt_len, len(ex[0]))
                    if (max(max_src_len + max_tgt_len) *
                            (self.cur_batch_size + 1) > self.max_tokens):
                        return False
                    self.max_src_len = max_src_len
                    self.max_tgt_len = max_tgt_len
                    self.cur_batch_size += 1
                    return True

            iterator = DataIterator(
                {'train': train_data, 'test': test_data},
                batching_strategy=CustomBatchingStrategy(max_tokens=1000))
    """

    # TODO: Think about whether we should support save/load.

    def __init__(self, datasets: DatasetsType,
                 batching_strategy: Optional[BatchingStrategy] = None):
        self._default_dataset_name = 'data'
        if isinstance(datasets, DataBase):
            datasets = {self._default_dataset_name: datasets}
        elif isinstance(datasets, Sequence):
            if any(not isinstance(d, DataBase) for d in datasets):
                raise ValueError("`datasets` must be an non-empty list of "
                                 "`texar.data.DataBase` instances.")
            num_datasets = len(datasets)
            datasets = {d.name: d for d in datasets}
            if len(datasets) < num_datasets:
                raise ValueError("Names of datasets must be unique.")

        _datasets = {name: SingleDatasetIterator(dataset, batching_strategy)
                     for name, dataset in datasets.items()}
        self._datasets = _datasets

        if len(self._datasets) <= 0:
            raise ValueError("`datasets` must not be empty.")

        self._current_dataset_name: Optional[str] = None

    @property
    def num_datasets(self) -> int:
        r"""Number of datasets.
        """
        return len(self._datasets)

    @property
    def dataset_names(self) -> List[str]:
        r"""A list of dataset names.
        """
        return list(self._datasets.keys())

    def _validate_dataset_name(self, dataset_name: Optional[str]) -> str:
        r"""Validate the provided dataset name, and return the validated name.
        """
        if dataset_name is None:
            if self.num_datasets > 1:
                raise ValueError("`dataset_name` is required if there are "
                                 "more than one datasets.")
            dataset_name = next(iter(self._datasets))
        if dataset_name not in self._datasets:
            raise ValueError("Dataset not found: ", dataset_name)
        return dataset_name

    def switch_to_dataset(self, dataset_name: Optional[str] = None):
        r"""Re-initializes the iterator of a given dataset and starts iterating
        over the dataset (from the beginning).

        Args:
            dataset_name (optional): Name of the dataset. If not provided,
                there must be only one Dataset.
        """
        self._current_dataset_name = self._validate_dataset_name(dataset_name)

    def get_iterator(self,
                     dataset_name: Optional[str] = None) -> Iterator[Batch]:
        r"""Re-initializes the iterator of a given dataset and starts iterating
        over the dataset (from the beginning).

        Args:
            dataset_name (optional): Name of the dataset. If not provided,
                there must be only one Dataset.
        """
        if dataset_name is not None or self._current_dataset_name is None:
            dataset_name = self._validate_dataset_name(dataset_name)
        elif self._current_dataset_name is not None:
            dataset_name = self._current_dataset_name
        else:
            raise ValueError("No dataset is selected.")

        return iter(self._datasets[dataset_name])

    def __iter__(self) -> Iterator[Batch]:
        r"""Returns the iterator for the currently selected or default dataset.
        """
        return self.get_iterator()

    def __len__(self):
        return len(self._datasets[self._validate_dataset_name(None)])


class TrainTestDataIterator(DataIterator):
    r"""Data iterator that alternatives between train, val, and test datasets.

    :attr:`train`, :attr:`val`, and :attr:`test` are instances of
    :class:`~texar.data.DataBase`. At least one of them must be provided.

    This is a wrapper of :class:`~texar.data.DataIterator`.

    Args:
        train (optional): Training data.
        val (optional): Validation data.
        test (optional): Test data.

    Example:

        .. code-block:: python

            train_data = MonoTextData(hparams_train)
            val_data = MonoTextData(hparams_val)
            iterator = TrainTestDataIterator(train=train_data, val=val_data)

            for epoch in range(200): # Run 200 epochs of train/val
                # Starts iterating through training data from the beginning.
                iterator.switch_to_train_data(sess)
                for batch in iterator:
                    ... # Do training with the batch.

                # Starts iterating through val data from the beginning.
                for batch in iterator.get_val_iterator():
                    ... # Do validation on the batch.
    """

    def __init__(self, train: Optional[DataBase] = None,
                 val: Optional[DataBase] = None,
                 test: Optional[DataBase] = None):
        dataset_dict = {}
        self._train_name = 'train'
        self._val_name = 'val'
        self._test_name = 'test'
        if train is not None:
            dataset_dict[self._train_name] = train
        if val is not None:
            dataset_dict[self._val_name] = val
        if test is not None:
            dataset_dict[self._test_name] = test
        if len(dataset_dict) == 0:
            raise ValueError("At least one of `train`, `val`, and `test` "
                             "must be provided.")

        super().__init__(dataset_dict)

    def switch_to_train_data(self) -> None:
        r"""Starts to iterate through training data (from the beginning).
        """
        if self._train_name not in self._datasets:
            raise ValueError("Training data not provided.")
        self.switch_to_dataset(self._train_name)

    def switch_to_val_data(self) -> None:
        r"""Starts to iterate through val data (from the beginning).
        """
        if self._val_name not in self._datasets:
            raise ValueError("Val data not provided.")
        self.switch_to_dataset(self._val_name)

    def switch_to_test_data(self) -> None:
        r"""Starts to iterate through test data (from the beginning).
        """
        if self._test_name not in self._datasets:
            raise ValueError("Test data not provided.")
        self.switch_to_dataset(self._test_name)

    def get_train_iterator(self) -> Iterable[Batch]:
        r"""Starts to iterate through train data (from the beginning).
        """
        if self._train_name not in self._datasets:
            raise ValueError("Training data not provided.")
        return self.get_iterator(self._train_name)

    def get_val_iterator(self) -> Iterable[Batch]:
        r"""Starts to iterate through val data (from the beginning).
        """
        if self._val_name not in self._datasets:
            raise ValueError("Val data not provided.")
        return self.get_iterator(self._val_name)

    def get_test_iterator(self) -> Iterable[Batch]:
        r"""Starts to iterate through test data (from the beginning).
        """
        if self._test_name not in self._datasets:
            raise ValueError("Test data not provided.")
        return self.get_iterator(self._test_name)
