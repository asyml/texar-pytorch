"""
Unit tests for data iterator related operations.
"""
import copy
import tempfile
import unittest
from unittest.mock import patch
from typing import List, Tuple

import numpy as np
import torch

from texar.torch.data.data.data_base import (
    DatasetBase, IterDataSource, SequenceDataSource, ZipDataSource)
from texar.torch.data.data.data_iterators import (
    DataIterator, TrainTestDataIterator)
from texar.torch.data.data.dataset_utils import Batch
from texar.torch.data.data.mono_text_data import MonoTextData
from texar.torch.data.data.sampler import TokenCountBatchingStrategy


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
            "num_epochs": 1,
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
        r"""Tests :class:`texar.torch.data.TrainTestDataIterator`
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

    def test_dynamic_batching(self):
        r"""Tests dynamic batching using :class:`texar.torch.data.BatchingStrategy`.
        """
        sent_lengths = np.random.randint(10, 20, size=(100,))
        sentences = [['a'] * length for length in sent_lengths]
        data_source = SequenceDataSource(sentences)

        class CustomData(DatasetBase):
            def __init__(self, source):
                super().__init__(source)

            def collate(self, examples):
                return Batch(len(examples), text=examples)

        train_data = CustomData(data_source)

        batch_size = 5
        max_tokens = 75
        strategy = TokenCountBatchingStrategy(
            max_tokens, batch_size, len)
        iterator = DataIterator(train_data, strategy)

        for batch in iterator:
            self.assertLessEqual(len(batch), batch_size)
            self.assertLessEqual(sum(len(s) for s in batch.text), max_tokens)

    @patch("torch.cuda.is_available", lambda: True)
    def test_auto_storage_moving(self):
        cuda_tensors = set()

        def move_tensor(tensor, device, non_blocking=False):
            if isinstance(device, torch.device) and device.type == "cuda":
                self.assertTrue(non_blocking)
                cuda_tensors.add(id(tensor))
            return tensor

        device = torch.device("cuda:0")

        with patch.object(torch.Tensor, "to", move_tensor):
            train = MonoTextData(self._train_hparams, device=device)
            iterator = DataIterator(train)
            for batch in iterator:
                self.assertTrue(id(batch.text_ids) in cuda_tensors)
                self.assertTrue(id(batch.length) in cuda_tensors)


RawExample = Tuple[List[int], str]
Example = Tuple[List[int], List[str]]


class MockDataBase(DatasetBase[RawExample, Example]):
    def process(self, raw_example: RawExample) -> Example:
        numbers, string = raw_example
        numbers = [x + 1 for x in numbers]
        strings = string.split()
        return numbers, strings

    def collate(self, examples: List[Example]) -> Batch:
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
                                 num_workers: int,
                                 parallelize_processing: bool = True,
                                 support_random_access: bool = False,
                                 shuffle: bool = False,
                                 **kwargs):
        hparams = {
            'batch_size': self.batch_size,
            'lazy_strategy': lazy_mode,
            'cache_strategy': cache_mode,
            'num_parallel_calls': num_workers,
            'shuffle': shuffle,
            'shuffle_buffer_size': self.size // 5,
            'parallelize_processing': parallelize_processing,
            'allow_smaller_final_batch': False,
            **kwargs,
        }
        numbers_data = [[x] * self.seq_len for x in range(self.size)]
        string_data = [' '.join(map(str, range(self.seq_len)))
                       for _ in range(self.size)]
        if not support_random_access:
            source = ZipDataSource(  # type: ignore
                IterDataSource(numbers_data),
                SequenceDataSource(string_data))
        else:
            source = ZipDataSource(
                SequenceDataSource(numbers_data),
                SequenceDataSource(string_data))
        data = MockDataBase(source, hparams)  # type: ignore
        iterator = DataIterator(data)

        if data._hparams.allow_smaller_final_batch:
            total_examples = self.size
            total_batches = (self.size + self.batch_size - 1) // self.batch_size
        else:
            total_examples = self.size // self.batch_size * self.batch_size
            total_batches = self.size // self.batch_size

        def check_batch(idx, batch):
            if idx == total_batches - 1:
                batch_size = (total_examples - 1) % self.batch_size + 1
            else:
                batch_size = self.batch_size
            self.assertEqual(batch.numbers.shape,
                             (batch_size, self.seq_len))
            if not shuffle:
                numbers = np.asarray([idx * self.batch_size + x + 1
                                      for x in range(batch_size)])
                self.assertTrue(np.all(batch.numbers == numbers[:, np.newaxis]))

        # check laziness
        if parallelize_processing:
            if lazy_mode == 'none':
                self.assertEqual(len(data._processed_cache), self.size)
            else:
                self.assertEqual(len(data._processed_cache), 0)
                if not support_random_access:
                    if lazy_mode == 'process':
                        self.assertEqual(len(data._cached_source._cache),
                                         self.size)
                    else:
                        self.assertEqual(len(data._cached_source._cache), 0)

        # first epoch
        cnt = 0
        for idx, batch in enumerate(iterator):
            check_batch(idx, batch)
            cnt += 1
        self.assertEqual(cnt, total_batches)

        # check cache
        if parallelize_processing:
            if cache_mode == 'none':
                self.assertEqual(len(data._processed_cache), 0)
            elif cache_mode == 'loaded':
                self.assertEqual(len(data._processed_cache), 0)
            else:
                self.assertEqual(len(data._processed_cache), self.size)
            if lazy_mode != 'none' and not support_random_access:
                if cache_mode == 'none':
                    self.assertEqual(len(data._cached_source._cache), 0)
                elif cache_mode == 'loaded':
                    self.assertEqual(len(data._cached_source._cache), self.size)
                else:
                    self.assertEqual(len(data._cached_source._cache), 0)

        # second epoch
        cnt = 0
        for idx, batch in enumerate(iterator):
            check_batch(idx, batch)
            cnt += 1
        self.assertEqual(cnt, total_batches)

        # check again
        if parallelize_processing:
            if cache_mode == 'none':
                self.assertEqual(len(data._processed_cache), 0)
            elif cache_mode == 'loaded':
                self.assertEqual(len(data._processed_cache), 0)
            else:
                self.assertEqual(len(data._processed_cache), self.size)
            if lazy_mode != 'none' and not support_random_access:
                if cache_mode == 'none':
                    self.assertEqual(len(data._cached_source._cache), 0)
                elif cache_mode == 'loaded':
                    self.assertEqual(len(data._cached_source._cache), self.size)
                else:
                    self.assertEqual(len(data._cached_source._cache), 0)

    def _test_modes(self, lazy_mode: str, cache_mode: str):
        self._test_modes_with_workers(lazy_mode, cache_mode, self.num_workers)
        self._test_modes_with_workers(lazy_mode, cache_mode, self.num_workers,
                                      parallelize_processing=False)
        self._test_modes_with_workers(lazy_mode, cache_mode, 1)
        self._test_modes_with_workers(lazy_mode, cache_mode, self.num_workers,
                                      support_random_access=True)
        self._test_modes_with_workers(lazy_mode, cache_mode, self.num_workers,
                                      shuffle=True)

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
