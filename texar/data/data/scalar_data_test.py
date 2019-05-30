# -*- coding: utf-8 -*-
#
"""
Unit tests for data related operations.
"""

import copy
import tempfile
import numpy as np
import unittest

from texar.data.data.data_iterators import DataIterator

import torch

import texar as tx


class ScalarDataTest(unittest.TestCase):
    """Tests scalar data class.
    """

    def setUp(self):
        # Create test data
        # pylint: disable=no-member
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
                "data_name": "label"
            }
        }

    def _run_and_test(self, hparams):
        # Construct database
        scalar_data = tx.data.ScalarData(hparams)

        self.assertEqual(scalar_data.list_items()[0],
                         hparams["dataset"]["data_name"])

        iterator = DataIterator(scalar_data)

        for idx, data_batch in enumerate(iterator):
            self.assertEqual(set(data_batch.keys()),
                             set(scalar_data.list_items()))
            value = data_batch[scalar_data.data_name][0]
            self.assertEqual(idx, value)
            # pylint: disable=no-member
            if hparams["dataset"]["data_type"] == "int":
                self.assertTrue(value.dtype, torch.int32)
            else:
                self.assertTrue(value.dtype, torch.float32)

    def test_default_setting(self):
        """Tests the logics of ScalarData.
        """
        self._run_and_test(self._int_hparams)
        self._run_and_test(self._float_hparams)

    def test_shuffle(self):
        """Tests results of toggling shuffle.
        """
        hparams = copy.copy(self._int_hparams)
        hparams["batch_size"] = 10
        scalar_data = tx.data.ScalarData(hparams)
        iterator = DataIterator(scalar_data)

        hparams_sfl = copy.copy(hparams)
        hparams_sfl["shuffle"] = True
        scalar_data_sfl = tx.data.ScalarData(hparams_sfl)
        iterator_sfl = DataIterator(scalar_data_sfl)

        vals = []
        vals_sfl = []

        for data_batch in iterator:
            vals += data_batch[scalar_data.data_name].tolist()

        for data_batch_sfl in iterator_sfl:
            vals_sfl += data_batch_sfl[scalar_data.data_name].tolist()

        self.assertEqual(len(vals), len(vals_sfl))
        self.assertSetEqual(set(vals), set(vals_sfl))


if __name__ == "__main__":
    unittest.main()
