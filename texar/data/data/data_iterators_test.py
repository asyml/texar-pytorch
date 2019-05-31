"""
Unit tests for data iterator related operations.
"""
import copy
import tempfile
import unittest
from typing import List, Optional, Tuple

import numpy as np

import texar
import torch
from texar.data.data.data_base import DataBase, IterDataSource, \
    SequenceDataSource, ZipDataSource
from texar.data.data.data_iterators import BufferShuffleSampler, DataIterator, TrainTestDataIterator
from texar.data.data.dataset_utils import Batch
from texar.data.data.dataset_utils import _CacheStrategy, _LazyStrategy
from texar.data.data.mono_text_data import MonoTextData


class SamplerTest(unittest.TestCase):
    r"""Tests samplers.
    """

    class MockDataBase(DataBase):
        class MockSource:
            def __getitem__(self, item):
                return item

            def reset(self):
                pass

        def __init__(self, size: int, lazy_strategy: str,
                     cache_strategy: str, unknown_size: bool = False):
            self._source = self.MockSource()
            self.size = size
            self._lazy_strategy = _LazyStrategy(lazy_strategy)
            self._cache_strategy = _CacheStrategy(cache_strategy)
            self._dataset_size = size if not unknown_size else None
            self._unknown_size = unknown_size
            self._supports_random_access = False
            self._fully_cached = False

        def _prefetch_source(self, index: int) -> Optional[int]:
            if self._unknown_size:
                if index >= self.size:
                    self._dataset_size = self.size
                    return self._dataset_size
            return None

    def setUp(self) -> None:
        self.size = 10
        self.buffer_size = 5

    def _test_data(self, data: MockDataBase,
                   returns_data: bool = False,
                   always_returns_data: bool = False):
        sampler = BufferShuffleSampler(data, self.buffer_size)
        for epoch in range(2):
            indices = list(iter(sampler))
            if always_returns_data or (returns_data and epoch == 0):
                examples = [ex[1] for ex in indices]
                indices = [ex[0] for ex in indices]
                np.testing.assert_array_equal(indices, examples)
            self.assertEqual(len(set(indices)), self.size)
            self.assertEqual(min(indices), 0)
            self.assertEqual(max(indices), self.size - 1)
            data._fully_cached = True

    def test_known_size(self):
        data = self.MockDataBase(self.size, 'none', 'processed')
        self._test_data(data)
        data = self.MockDataBase(self.size, 'all', 'none', unknown_size=True)
        self._test_data(data, always_returns_data=True)

    def test_non_lazy_loading(self):
        strategies = [
            ('none', 'processed'),
            ('process', 'loaded'),
            ('process', 'processed'),
        ]
        for lazy, cache in strategies:
            data = self.MockDataBase(self.size, lazy, cache)
            self._test_data(data)

    def test_lazy_loading(self):
        data = self.MockDataBase(self.size, 'all', 'loaded', unknown_size=True)
        self._test_data(data, returns_data=True)
        data = self.MockDataBase(self.size, 'all', 'processed',
                                 unknown_size=True)
        self._test_data(data, returns_data=True)


class DataIteratorTest(unittest.TestCase):
    r"""Tests data iterators.
    """

    def setUp(self):
        # Create data
        self.train_text = list(np.linspace(1, 1000, num=1000, dtype=np.int64))
        self.train_text = [str(x) for x in self.train_text]
        train_text_file = tempfile.NamedTemporaryFile()
        train_text_file.write('\n'.join(self.train_text).encode("utf-8"))
        train_text_file.flush()
        self._train_text_file = train_text_file

        test_text = list(np.linspace(1001, 2000, num=1000, dtype=np.int64))
        test_text = [str(x) for x in test_text]
        test_text_file = tempfile.NamedTemporaryFile()
        test_text_file.write('\n'.join(test_text).encode("utf-8"))
        test_text_file.flush()
        self._test_text_file = test_text_file

        vocab_list = self.train_text + test_text
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()
        self._vocab_file = vocab_file
        self._vocab_size = len(vocab_list)

        self._train_hparams = {
            "num_epochs": 2,
            "batch_size": 1,
            "shuffle": False,
            "dataset": {
                "files": self._train_text_file.name,
                "vocab_file": self._vocab_file.name,
                "bos_token": '',
                "eos_token": ''
            },
            "name": "train"
        }

        self._test_hparams = {
            "num_epochs": 1,
            "batch_size": 2,
            "shuffle": False,
            "dataset": {
                "files": self._test_text_file.name,
                "vocab_file": self._vocab_file.name,
                "bos_token": '',
                "eos_token": ''
            },
            "name": "test"
        }

    def test_iterator_single_dataset(self):
        r"""Tests iterating over a single dataset.
        """
        data = MonoTextData(self._test_hparams)
        data_iterator = DataIterator(data)
        data_iterator.switch_to_dataset(dataset_name="data")
        iterator = data_iterator.get_iterator()
        i = 1001
        for idx, batch in enumerate(iterator):
            self.assertEqual(batch.batch_size, self._test_hparams['batch_size'])
            np.testing.assert_array_equal(batch['length'], [1, 1])
            for example in batch['text']:
                self.assertEqual(example[0], str(i))
                i += 1
        self.assertEqual(i, 2001)

    def test_iterator_single_dataset_parallel(self):
        r"""Tests iterating over a single dataset with multiple workers.
        """
        hparams = copy.deepcopy(self._test_hparams)
        hparams['num_parallel_calls'] = 2
        data = MonoTextData(hparams)
        data_iterator = DataIterator(data)
        data_iterator.switch_to_dataset(dataset_name="data")
        iterator = data_iterator.get_iterator()
        i = 1001
        for idx, batch in enumerate(iterator):
            self.assertEqual(batch.batch_size, self._test_hparams['batch_size'])
            np.testing.assert_array_equal(batch['length'], [1, 1])
            for example in batch['text']:
                self.assertEqual(example[0], str(i))
                i += 1
        self.assertEqual(i, 2001)

    def test_iterator_multi_datasets(self):
        r"""Tests iterating over multiple datasets.
        """
        train = MonoTextData(self._train_hparams)
        test = MonoTextData(self._test_hparams)
        train_batch_size = self._train_hparams["batch_size"]
        test_batch_size = self._test_hparams["batch_size"]
        data_iterator = DataIterator({"train": train, "test": test})
        data_iterator.switch_to_dataset(dataset_name="train")
        iterator = data_iterator.get_iterator()
        for idx, val in enumerate(iterator):
            self.assertEqual(len(val), train_batch_size)
            number = idx * train_batch_size + 1
            self.assertEqual(val.text[0], [str(number)])
            # numbers: 1 - 2000, first 4 vocab entries are special tokens
            self.assertEqual(val.text_ids[0], torch.tensor(number + 3))

        data_iterator.switch_to_dataset(dataset_name="test")
        iterator = data_iterator.get_iterator()
        for idx, val in enumerate(iterator):
            self.assertEqual(len(val), test_batch_size)
            number = idx * test_batch_size + 1001
            self.assertEqual(val.text[0], [str(number)])
            self.assertEqual(val.text_ids[0], torch.tensor(number + 3))

        # test `get_iterator` interface
        for idx, val in enumerate(data_iterator.get_iterator('train')):
            self.assertEqual(len(val), train_batch_size)
            number = idx * train_batch_size + 1
            self.assertEqual(val.text[0], [str(number)])
            self.assertEqual(val.text_ids[0], torch.tensor(number + 3))

        # test exception for invalid dataset name
        with self.assertRaises(ValueError) as context:
            data_iterator.switch_to_dataset('val')
        self.assertTrue('not found' in str(context.exception))

    def test_train_test_data_iterator(self):
        r"""Tests :class:`texar.data.TrainTestDataIterator`
        """
        train = MonoTextData(self._train_hparams)
        test = MonoTextData(self._test_hparams)
        train_batch_size = self._train_hparams["batch_size"]
        test_batch_size = self._test_hparams["batch_size"]

        data_iterator = TrainTestDataIterator(train=train, test=test)
        data_iterator.switch_to_train_data()
        iterator = data_iterator.get_iterator()

        for idx, val in enumerate(iterator):
            self.assertEqual(len(val), train_batch_size)
            number = idx * train_batch_size + 1
            self.assertEqual(val.text[0], [str(number)])
            # numbers: 1 - 2000, first 4 vocab entries are special tokens
            self.assertEqual(val.text_ids[0], torch.tensor(number + 3))

        data_iterator.switch_to_test_data()
        iterator = data_iterator.get_iterator()
        for idx, val in enumerate(iterator):
            self.assertEqual(len(val), test_batch_size)
            number = idx * test_batch_size + 1001
            self.assertEqual(val.text[0], [str(number)])
            self.assertEqual(val.text_ids[0], torch.tensor(number + 3))

        # test `get_*_iterator` interface
        for idx, val in enumerate(data_iterator.get_test_iterator()):
            self.assertEqual(len(val), test_batch_size)
            number = idx * test_batch_size + 1001
            self.assertEqual(val.text[0], [str(number)])
            self.assertEqual(val.text_ids[0], torch.tensor(number + 3))

        # test exception for invalid dataset name
        with self.assertRaises(ValueError) as context:
            data_iterator.switch_to_val_data()
        self.assertTrue('Val data not provided' in str(context.exception))


RawExample = Tuple[List[int], str]
Example = Tuple[List[int], List[str]]


class MockDataBase(DataBase[RawExample, Example]):
    def _process(self, raw_example: RawExample) -> Example:
        numbers, string = raw_example
        numbers = [x + 1 for x in numbers]
        strings = string.split()
        return numbers, strings

    def _collate(self, examples: List[Example]) -> Batch:
        numbers = np.asarray([ex[0] for ex in examples])
        strings = np.asarray([ex[1] for ex in examples])
        return Batch(len(numbers), numbers=numbers, strings=strings)


class LazinessCachingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.size = 102
        self.seq_len = 10
        self.batch_size = 5
        self.num_workers = 3

    def _test_modes_with_workers(self, lazy_mode: str, cache_mode: str,
                                 num_workers: int):
        hparams = {
            'batch_size': self.batch_size,
            'lazy_strategy': lazy_mode,
            'cache_strategy': cache_mode,
            'num_parallel_calls': num_workers,
            'shuffle': False,
        }
        source = ZipDataSource(
            IterDataSource([[x] * self.seq_len for x in range(self.size)]),
            SequenceDataSource([' '.join(map(str, range(self.seq_len)))
                                for _ in range(self.size)]))
        data = MockDataBase(source, hparams)
        iterator = DataIterator(data)

        total_batches = (self.size + self.batch_size - 1) // self.batch_size

        def check_batch(idx, batch):
            if idx == total_batches - 1:
                batch_size = (self.size - 1) % self.batch_size + 1
            else:
                batch_size = self.batch_size
            self.assertEqual(batch.numbers.shape,
                             (batch_size, self.seq_len))
            numbers = [idx * self.batch_size + x + 1 for x in range(batch_size)]
            self.assertTrue(np.all(batch.numbers ==
                                   np.asarray(numbers)[:, np.newaxis]))

        # check laziness
        if lazy_mode == 'none':
            self.assertEqual(len(data._processed_cache), self.size)
        else:
            self.assertIsInstance(data._source,
                                  texar.data.data.data_base._CachedDataSource)
            self.assertEqual(len(data._processed_cache), 0)
            if lazy_mode == 'process':
                self.assertEqual(len(data._source._cache), self.size)
            else:
                self.assertEqual(len(data._source._cache), 0)

        # first epoch
        cnt = 0
        for idx, batch in enumerate(iterator):
            check_batch(idx, batch)
            cnt += 1
        self.assertEqual(cnt, total_batches)

        # check cache
        if cache_mode == 'none':
            self.assertEqual(len(data._processed_cache), 0)
        elif cache_mode == 'loaded':
            self.assertEqual(len(data._processed_cache), 0)
        else:
            self.assertEqual(len(data._processed_cache), self.size)
        if lazy_mode != 'none':
            if cache_mode == 'none':
                self.assertEqual(len(data._source._cache), 0)
            elif cache_mode == 'loaded':
                self.assertEqual(len(data._source._cache), self.size)
            else:
                self.assertEqual(len(data._source._cache), 0)

        # second epoch
        cnt = 0
        for idx, batch in enumerate(iterator):
            check_batch(idx, batch)
            cnt += 1
        self.assertEqual(cnt, total_batches)

    def _test_modes(self, lazy_mode: str, cache_mode: str):
        self._test_modes_with_workers(lazy_mode, cache_mode, self.num_workers)
        self._test_modes_with_workers(lazy_mode, cache_mode, 1)

    def test_none_processed(self):
        self._test_modes('none', 'processed')

    def test_process_loaded(self):
        self._test_modes('process', 'loaded')

    def test_process_processed(self):
        self._test_modes('process', 'processed')

    def test_all_none(self):
        self._test_modes('all', 'none')

    def test_all_loaded(self):
        self._test_modes('all', 'loaded')

    def test_all_processed(self):
        self._test_modes('all', 'processed')


if __name__ == "__main__":
    unittest.main()