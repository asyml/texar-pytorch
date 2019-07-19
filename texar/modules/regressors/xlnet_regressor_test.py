"""
Unit tests for XLNet regressor.
"""

import unittest

import torch

from texar.modules.regressors.xlnet_regressor import *


@unittest.skip("Manual test only")
class XLNetRegressorTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.XLNetRegressor` class.
    """

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1
        regressor = XLNetRegressor()
        _ = regressor(inputs)
        self.assertEqual(len(regressor.trainable_variables), 182 + 4)

        # case 2
        hparams = {
            "use_projection": False
        }
        regressor = XLNetRegressor(hparams=hparams)
        _ = regressor(inputs)
        self.assertEqual(len(regressor.trainable_variables), 182 + 2)

        # case 3
        hparams = {
            "summary_type": "first"
        }
        regressor = XLNetRegressor(hparams=hparams)
        _ = regressor(inputs)
        self.assertEqual(len(regressor.trainable_variables), 182 + 4)

        # case 4
        hparams = {
            "summary_type": "mean"
        }
        regressor = XLNetRegressor(hparams=hparams)
        _ = regressor(inputs)
        self.assertEqual(len(regressor.trainable_variables), 182 + 4)

    def test_regression(self):
        r"""Tests regression.
        """
        max_time = 8
        batch_size = 16
        inputs = torch.randint(32000, (batch_size, max_time), dtype=torch.int64)

        # case 1
        regressor = XLNetRegressor()
        preds = regressor(inputs)

        self.assertEqual(preds.shape, torch.Size([batch_size]))

        # case 2
        hparams = {
            "summary_type": "first"
        }
        regressor = XLNetRegressor(hparams=hparams)
        preds = regressor(inputs)

        self.assertEqual(preds.shape, torch.Size([batch_size]))

        # case 3
        hparams = {
            "summary_type": "mean"
        }
        regressor = XLNetRegressor(hparams=hparams)
        preds = regressor(inputs)

        self.assertEqual(preds.shape, torch.Size([batch_size]))


if __name__ == "__main__":
    unittest.main()
