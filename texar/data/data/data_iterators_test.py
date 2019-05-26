# -*- coding: utf-8 -*-
#
"""
Unit tests for data iterator related operations.
"""
import torch
from torch.utils.data import TensorDataset

import unittest
import tempfile
import numpy as np

import texar as tx


class DataIteratorTest(unittest.TestCase):
    """Tests data iterators.
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
            "batch_size": 1,
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
        """Tests iterating over a single dataset.
        """
        # todo(avinash): use this dataset once MonoTextData is ready
        # data = tx.data.MonoTextData(self._test_hparams)
        data = TensorDataset(torch.from_numpy(np.arange(0, 100, 1)))
        data_iterator = tx.data.DataIterator(data)
        data_iterator.switch_to_dataset(dataset_name="data")
        iterator = data_iterator.get_iterator()
        for idx, val in enumerate(iterator):
            self.assertEqual(len(val), self._test_hparams["batch_size"])
            self.assertEqual(val[0], torch.tensor(idx))

    def test_iterator_multi_datasets(self):
        """Tests iterating over multiple datasets.
        """
        train = TensorDataset(torch.from_numpy(np.arange(0, 100, 1)))
        test = TensorDataset(torch.from_numpy(np.arange(100, 200, 1)))
        data_iterator = tx.data.DataIterator({"train": train, "test": test})
        data_iterator.switch_to_dataset(dataset_name="train")
        iterator = data_iterator.get_iterator()
        for idx, val in enumerate(iterator):
            self.assertEqual(len(val), self._train_hparams["batch_size"])
            self.assertEqual(val[0], torch.tensor(idx))

        data_iterator.switch_to_dataset(dataset_name="test")
        iterator = data_iterator.get_iterator()
        for idx, val in enumerate(iterator):
            self.assertEqual(len(val), self._test_hparams["batch_size"])
            self.assertEqual(val[0], torch.tensor(idx + 100))

    def test_train_test_data_iterator(self):
        """Tests :class:`texar.data.TrainTestDataIterator`
        """
        train_data = TensorDataset(torch.from_numpy(np.arange(0, 100, 1)))
        test_data = TensorDataset(torch.from_numpy(np.arange(100, 200, 1)))

        data_iterator = tx.data.TrainTestDataIterator(train=train_data,
                                                      test=test_data)
        data_iterator.switch_to_train_data()
        iterator = data_iterator.get_iterator()

        for idx, val in enumerate(iterator):
            self.assertEqual(len(val), self._train_hparams["batch_size"])
            self.assertEqual(val[0], torch.tensor(idx))

        data_iterator.switch_to_test_data()
        iterator = data_iterator.get_iterator()
        for idx, val in enumerate(iterator):
            self.assertEqual(len(val), self._test_hparams["batch_size"])
            self.assertEqual(val[0], torch.tensor(idx + 100))


if __name__ == "__main__":
    unittest.main()
