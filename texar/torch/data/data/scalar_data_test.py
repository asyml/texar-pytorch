# -*- coding: utf-8 -*-
#
"""
Unit tests for data related operations.
"""
import torch

import copy
import tempfile
import numpy as np

import unittest

from texar.torch.data import DataIterator, ScalarData
from texar.torch.utils.dtypes import torch_bool


class ScalarDataTest(unittest.TestCase):
    """Tests scalar data class.
    """

    def setUp(self):
        # Create test data
        int_data = np.linspace(0, 100, num=101, dtype=np.int32).tolist()
        int_data = [str(i) for i in int_data]
        int_file = tempfile.NamedTemporaryFile()
        int_file.write('\n'.join(int_data).encode("utf-8"))
        int_file.flush()
        self._int_file = int_file

        self._int_hparams = {
            "num_epochs": 1,
            "batch_size": 1,
            "shuffle": False,
            "dataset": {
                "files": self._int_file.name,
                "data_type": "int",
                "data_name": "label"
            }
        }

        self._float_hparams = {
            "num_epochs": 1,
            "batch_size": 1,
            "shuffle": False,
            "dataset": {
                "files": self._int_file.name,
                "data_type": "float",
                "data_name": "feat"
            }
        }

        bool_data = [0, 1]
        bool_data = [str(i) for i in bool_data]
        bool_file = tempfile.NamedTemporaryFile()
        bool_file.write('\n'.join(bool_data).encode("utf-8"))
        bool_file.flush()
        self._bool_file = bool_file

        self._bool_hparams = {
            "num_epochs": 1,
            "batch_size": 1,
            "shuffle": False,
            "dataset": {
                "files": self._bool_file.name,
                "data_type": "bool",
                "data_name": "feat"
            }
        }

    def _run_and_test(self, hparams, test_transform=False):
        # Construct database
        scalar_data = ScalarData(hparams)

        self.assertEqual(scalar_data.list_items()[0],
                         hparams["dataset"]["data_name"])

        iterator = DataIterator(scalar_data)

        i = 0
        for batch in iterator:
            self.assertEqual(set(batch.keys()),
                             set(scalar_data.list_items()))
            value = batch[scalar_data.data_name][0]
            if test_transform:
                self.assertEqual(2 * i, value)
            else:
                self.assertEqual(i, value)
            i += 1
            data_type = hparams["dataset"]["data_type"]
            if data_type == "int":
                self.assertEqual(value.dtype, torch.int32)
            elif data_type == "float":
                self.assertEqual(value.dtype, torch.float32)
            elif data_type == "bool":
                self.assertTrue(value.dtype, torch_bool)
            self.assertIsInstance(value, torch.Tensor)

    def test_default_setting(self):
        """Tests the logic of ScalarData.
        """
        self._run_and_test(self._int_hparams)
        self._run_and_test(self._float_hparams)
        self._run_and_test(self._bool_hparams)

    def test_shuffle(self):
        """Tests results of toggling shuffle.
        """
        hparams = copy.copy(self._int_hparams)
        hparams["batch_size"] = 10
        scalar_data = ScalarData(hparams)
        iterator = DataIterator(scalar_data)

        hparams_sfl = copy.copy(hparams)
        hparams_sfl["shuffle"] = True
        scalar_data_sfl = ScalarData(hparams_sfl)
        iterator_sfl = DataIterator(scalar_data_sfl)

        vals = []
        vals_sfl = []

        for batch, batch_sfl in zip(iterator, iterator_sfl):
            vals += batch["label"].tolist()
            vals_sfl += batch_sfl["label"].tolist()

        self.assertEqual(len(vals), len(vals_sfl))
        self.assertSetEqual(set(vals), set(vals_sfl))

    def test_transform(self):
        """Tests transform logic.
        """
        hparams = copy.deepcopy(self._int_hparams)
        hparams["dataset"].update(
            {"other_transformations": [lambda x: x * 2]})
        self._run_and_test(hparams, test_transform=True)

        hparams = copy.deepcopy(self._float_hparams)
        hparams["dataset"].update(
            {"other_transformations": [lambda x: x * 2]})
        self._run_and_test(hparams, test_transform=True)

    def test_unsupported_scalar_types(self):
        """Tests exception for unsupported scalar types.
        """
        hparams = copy.copy(self._int_hparams)
        hparams["dataset"].update({
            "data_type": "XYZ"
        })

        with self.assertRaises(ValueError):
            self._run_and_test(hparams)

        hparams = copy.copy(self._int_hparams)
        hparams["dataset"].update({
            "data_type": "str"
        })

        with self.assertRaises(ValueError):
            self._run_and_test(hparams)


if __name__ == "__main__":
    unittest.main()
