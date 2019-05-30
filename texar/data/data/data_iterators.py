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
from typing import Dict, Iterable, List, Optional, Sequence, Union, Iterator

import torch
from torch.utils.data import DataLoader, Sampler

from texar.data.data.data_base import DataBase
from texar.data.data.dataset_utils import Batch, _CacheStrategy, _LazyStrategy
from texar.utils.types import MaybeSeq

__all__ = [
    "DataIterator",
    "TrainTestDataIterator",
]

DatasetsType = Union[Dict[str, DataBase], MaybeSeq[DataBase]]


class BufferBasedShuffler(Sampler):
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

        self._source = data
        self.size: Optional[int] = data._dataset_size

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
            cur_size = self._source._prefetch_source(index)
            if cur_size is not None:
                self.size = cur_size
            if self.size is not None and index >= self.size:
                break
            yield index
            buffer[sample] = x
            x += 1
        yield from (buffer[x] for x in torch.randperm(self.buffer_size)
                    if buffer[x] < self.size)

    def __iter__(self) -> Iterator[int]:
        if self.size is not None:
            # Non-lazy loading, or when dataset has been fully iterated.
            iterator = self._iterator_given_size(self.size)
        else:
            # First epoch of lazy loading, calling prefetch, and returning
            # indices and examples.
            iterator = self._iterator_unknown_size()

        if self.size is None or (
                self._source.cache_strategy is _CacheStrategy.NONE and
                self._source.lazy_strategy is _LazyStrategy.ALL):
            # Return indices and examples for any epoch in this case.
            iterator = map(lambda idx: (idx, self._source[idx]), iterator)
        return iterator

    def __len__(self):
        return self.size


class SingleDatasetIterator(DataLoader):
    r"""Iterator for a single dataset. This iterator is based on the PyTorch
    :class:`~torch.utils.data.DataLoader` interface, with a custom shuffling
    routine. This class is used internally.

    Args:
        dataset: The dataset to iterator through. The dataset must be an
            instance of :class:`texar.data.DataBase`, because configurations are
            read from the dataset `HParams`.
    """

    def __init__(self, dataset: DataBase):
        shuffle = dataset.hparams.shuffle
        shuffle_buffer_size = dataset.hparams.shuffle_buffer_size
        sampler = None
        if shuffle and shuffle_buffer_size is not None:
            sampler = BufferBasedShuffler(dataset, shuffle_buffer_size)
            shuffle = None

        num_parallel_calls = dataset.hparams.num_parallel_calls
        allow_smaller_final_batch = dataset.hparams.allow_smaller_final_batch
        collate_fn = dataset.collate_fn
        super().__init__(  # type: ignore
            dataset, dataset.batch_size, shuffle=shuffle, sampler=sampler,
            collate_fn=collate_fn,
            num_workers=(0 if num_parallel_calls == 1 else num_parallel_calls),
            drop_last=(not allow_smaller_final_batch))


class DataIterator:
    r"""Data iterator that switches and iterates through multiple datasets.

    This is a wrapper of TF reinitializable :tf_main:`iterator <data/Iterator>`.

    Args:
        datasets: Datasets to iterate through. This can be:

            - A single instance of :tf_main:`tf.data.Dataset <data/Dataset>` \
            or instance of subclass of :class:`~texar.data.DataBase`.
            - A `dict` that maps dataset name to \
            instance of :tf_main:`tf.data.Dataset <data/Dataset>` or \
            subclass of :class:`~texar.data.DataBase`.
            - A `list` of instances of subclasses of \
            :class:`texar.data.DataBase`. The name of instances \
            (:attr:`texar.data.DataBase.name`) must be unique.

    Example:

        .. code-block:: python

            train_data = MonoTextData(hparams_train)
            test_data = MonoTextData(hparams_test)
            iterator = DataIterator({'train': train_data, 'test': test_data})
            batch = iterator.get_next()

            sess = tf.Session()

            for _ in range(200): # Run 200 epochs of train/test
                # Starts iterating through training data from the beginning
                iterator.switch_to_dataset(sess, 'train')
                while True:
                    try:
                        train_batch_ = sess.run(batch)
                    except tf.errors.OutOfRangeError:
                        print("End of training epoch.")
                # Starts iterating through test data from the beginning
                iterator.switch_to_dataset(sess, 'test')
                while True:
                    try:
                        test_batch_ = sess.run(batch)
                    except tf.errors.OutOfRangeError:
                        print("End of test epoch.")
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
                     dataset_name: Optional[str] = None) -> Iterable[Batch]:
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

    def __iter__(self) -> Iterable[Batch]:
        r"""Returns the iterator for the currently selected or default dataset.
        """
        return self.get_iterator()


class TrainTestDataIterator(DataIterator):
    r"""Data iterator that alternatives between train, val, and test datasets.

    :attr:`train`, :attr:`val`, and :attr:`test` can be instance of
    either :tf_main:`tf.data.Dataset <data/Dataset>` or subclass of
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
            batch = iterator.get_next()

            sess = tf.Session()

            for _ in range(200): # Run 200 epochs of train/val
                # Starts iterating through training data from the beginning
                iterator.switch_to_train_data(sess)
                while True:
                    try:
                        train_batch_ = sess.run(batch)
                    except tf.errors.OutOfRangeError:
                        print("End of training epoch.")
                # Starts iterating through val data from the beginning
                iterator.switch_to_val_dataset(sess)
                while True:
                    try:
                        val_batch_ = sess.run(batch)
                    except tf.errors.OutOfRangeError:
                        print("End of val epoch.")
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
