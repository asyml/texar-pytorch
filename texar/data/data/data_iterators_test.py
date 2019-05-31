"""
Unit tests for data iterator related operations.
"""
import copy
import tempfile
import unittest
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import TensorDataset

import texar.data
from texar.data.data.data_iterators import BufferShuffleSampler
from texar.data.data.dataset_utils import _CacheStrategy, _LazyStrategy


class SamplerTest(unittest.TestCase):
    r"""Tests samplers.
    """

    class MockDataBase:
        def __init__(self, size: int, lazy_strategy: str,
                     cache_strategy: str, unknown_size: bool = False):
            self.size = size
            self._lazy_strategy = _LazyStrategy(lazy_strategy)
            self._cache_strategy = _CacheStrategy(cache_strategy)
            self._dataset_size = size if not unknown_size else None
            self._unknown_size = unknown_size

        def _prefetch_source(self, index: int) -> Optional[int]:
            if self._unknown_size:
                if index >= self.size:
                    self._dataset_size = self.size
                    return self._dataset_size
            return None

        def __getitem__(self, item):
            return item

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
        data = texar.data.MonoTextData(self._test_hparams)
        data_iterator = texar.data.DataIterator(data)
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
        data = texar.data.MonoTextData(hparams)
        data_iterator = texar.data.DataIterator(data)
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
        train = texar.data.MonoTextData(self._train_hparams)
        test = texar.data.MonoTextData(self._test_hparams)
        train_batch_size = self._train_hparams["batch_size"]
        test_batch_size = self._test_hparams["batch_size"]
        data_iterator = texar.data.DataIterator({"train": train, "test": test})
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
        train = texar.data.MonoTextData(self._train_hparams)
        test = texar.data.MonoTextData(self._test_hparams)
        train_batch_size = self._train_hparams["batch_size"]
        test_batch_size = self._test_hparams["batch_size"]

        data_iterator = texar.data.TrainTestDataIterator(train=train, test=test)
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


if __name__ == "__main__":
    unittest.main()
