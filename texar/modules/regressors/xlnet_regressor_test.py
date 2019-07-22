"""
Unit tests for XLNet regressor.
"""

import unittest

import torch

from texar.modules.regressors.xlnet_regressor import *


class XLNetRegressorTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.XLNetRegressor` class.
    """

    @unittest.skip("Manual test only")
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1
        regressor = XLNetRegressor(pretrained_model_name="xlnet-base-cased")
        _ = regressor(inputs)

        # case 2
        regressor = XLNetRegressor(pretrained_model_name="xlnet-large-cased")
        _ = regressor(inputs)

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1
        hparams = {
            "pretrained_model_name": None
        }
        regressor = XLNetRegressor(hparams=hparams)
        _ = regressor(inputs)
        self.assertEqual(len(regressor.trainable_variables), 182 + 4)

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "use_projection": False
        }
        regressor = XLNetRegressor(hparams=hparams)
        _ = regressor(inputs)
        self.assertEqual(len(regressor.trainable_variables), 182 + 2)

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "regr_strategy": "all_time",
            "max_seq_length": 8
        }
        regressor = XLNetRegressor(hparams=hparams)
        _ = regressor(inputs)
        self.assertEqual(len(regressor.trainable_variables), 182 + 4)

        # case 4
        hparams = {
            "pretrained_model_name": None,
            "regr_strategy": "time_wise"
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
        hparams = {
            "pretrained_model_name": None
        }
        regressor = XLNetRegressor(hparams=hparams)
        preds = regressor(inputs)
        self.assertEqual(preds.shape, torch.Size([batch_size]))

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "regr_strategy": "all_time",
            "max_seq_length": max_time
        }
        regressor = XLNetRegressor(hparams=hparams)
        preds = regressor(inputs)
        self.assertEqual(preds.shape, torch.Size([batch_size]))

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "regr_strategy": "time_wise"
        }
        regressor = XLNetRegressor(hparams=hparams)
        preds = regressor(inputs)
        self.assertEqual(preds.shape, torch.Size([batch_size, max_time]))


if __name__ == "__main__":
    unittest.main()
