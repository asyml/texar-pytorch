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
from typing import (
    Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence,
    Tuple, Union)

import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler as torch_sampler
from torch.utils.data.dataloader import _DataLoaderIter as torch_DataLoaderIter

from texar.data.data.data_base import DataBase
from texar.data.data.dataset_utils import Batch
from texar.utils.types import MaybeSeq

__all__ = [
    "DataIterator",
    "TrainTestDataIterator",
]

DatasetsType = Union[Dict[str, DataBase], MaybeSeq[DataBase]]


class SamplerBase(torch_sampler.Sampler):
    r"""A subclass of :class:`~torch.utils.data.Sampler` that supports:

    - Returning raw examples when required.
    - Creating iterators with unknown dataset size.

    This class is used internally in :class:`~texar.data.data.DataIterator`.
    It calls the :meth:`~texar.data.data.DataBase._prefetch_source` method to
    ensure the required number of raw examples are prefetched from source.

    Args:
        data: The :class:`~texar.data.data.DataBase` instance.
    """

    def __init__(self, data: DataBase):
        super().__init__(data)

        self._data = data
        self.size: Optional[int] = None

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

    def __iter__(self) -> Union[Iterator[int], Iterator[Tuple[int, Any]]]:
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


class SequentialSampler(SamplerBase):
    r"""Samples elements sequentially, always in the same order. Same as
    :class:`torch.utils.data.SequentialSampler`.

    Args:
        data: The :class:`~texar.data.data.DataBase` instance.
    """

    def __init__(self, data: DataBase):
        super().__init__(data)

    def _iterator_given_size(self, size: int) -> Iterator[int]:
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


class RandomSampler(SamplerBase):
    r"""Samples elements randomly. If without replacement, then sample from a
    shuffled dataset. If with replacement, then user can specify ``num_samples``
    to draw.

    This class uses :class:`torch.utils.data.RandomSampler` directly. Given the
    nature of such shuffling, it cannot be used for iterators with unknown size.

    Args:
        data: The :class:`~texar.data.data.DataBase` instance.
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``,
            default=False
    """

    def __init__(self, data: DataBase, replacement: bool = False,
                 num_samples: Optional[int] = None):
        super().__init__(data)
        self._sampler = torch_sampler.RandomSampler(
            data, replacement, num_samples)

    def _iterator_given_size(self, size: int) -> Iterator[int]:
        return iter(self._sampler)

    def _iterator_unknown_size(self) -> Iterator[int]:
        raise TypeError(
            "RandomSampler does not support lazy data loading. To perform "
            "shuffling with lazy loading, use BufferShuffleSampler.")


class BufferShuffleSampler(SamplerBase):
    r"""A :class:`~torch.utils.data.Sampler` that uses a shuffle buffer, as
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

    def __init__(self, data: DataBase, buffer_size: int):
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


class _DataLoaderIter(torch_DataLoaderIter):
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
        if (batch.batch_size < self._batch_size and
                not self.dataset.hparams.allow_smaller_final_batch):
            raise StopIteration
        return batch


class _CacheDataLoaderIter(torch_DataLoaderIter):
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
        if (batch.batch_size < self.dataset.batch_size and
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
    """
    dataset: DataBase

    def __init__(self, dataset: DataBase):
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
        super().__init__(
            dataset, dataset.batch_size, sampler=sampler, collate_fn=collate_fn,
            num_workers=(0 if num_parallel_calls == 1 else num_parallel_calls),
            drop_last=False)

    def __iter__(self):
        if self.dataset._should_return_processed_examples:
            # Accepts processed examples from workers and add to dataset cache.
            return _CacheDataLoaderIter(self)
        else:
            return _DataLoaderIter(self)


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

    Example:

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
    """

    # TODO: Think about whether we should support save/load.

    def __init__(self, datasets: DatasetsType):
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

        _datasets = {name: SingleDatasetIterator(dataset)
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
