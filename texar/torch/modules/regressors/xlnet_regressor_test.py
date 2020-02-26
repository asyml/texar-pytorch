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
Unit tests for XLNet regressor.
"""

import unittest

import torch

from texar.torch.modules.regressors.xlnet_regressor import *
from texar.torch.utils.test import pretrained_test


class XLNetRegressorTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.XLNetRegressor` class.
    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        # case 1
        regressor = XLNetRegressor(pretrained_model_name="xlnet-base-cased")
        _ = regressor(self.inputs)

        # case 2
        regressor = XLNetRegressor(pretrained_model_name="xlnet-large-cased")
        _ = regressor(self.inputs)

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        regressor = XLNetRegressor(hparams=hparams)
        self.assertEqual(len(regressor.trainable_variables), 182 + 4)
        _ = regressor(self.inputs)

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "use_projection": False,
        }
        regressor = XLNetRegressor(hparams=hparams)
        self.assertEqual(len(regressor.trainable_variables), 182 + 2)
        _ = regressor(self.inputs)

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "regr_strategy": "all_time",
            "max_seq_length": 8,
        }
        regressor = XLNetRegressor(hparams=hparams)
        self.assertEqual(len(regressor.trainable_variables), 182 + 4)
        _ = regressor(self.inputs)

        # case 4
        hparams = {
            "pretrained_model_name": None,
            "regr_strategy": "time_wise",
        }
        regressor = XLNetRegressor(hparams=hparams)
        self.assertEqual(len(regressor.trainable_variables), 182 + 4)
        _ = regressor(self.inputs)

    def test_regression(self):
        r"""Tests regression.
        """
        inputs = torch.randint(32000, (self.batch_size, self.max_length))

        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        regressor = XLNetRegressor(hparams=hparams)
        preds = regressor(inputs)
        self.assertEqual(preds.shape, torch.Size([self.batch_size]))

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "regr_strategy": "all_time",
            "max_seq_length": self.max_length,
        }
        regressor = XLNetRegressor(hparams=hparams)
        preds = regressor(inputs)
        self.assertEqual(preds.shape, torch.Size([self.batch_size]))

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "regr_strategy": "time_wise",
        }
        regressor = XLNetRegressor(hparams=hparams)
        preds = regressor(inputs)
        self.assertEqual(preds.shape, torch.Size(
            [self.batch_size, self.max_length]))

    def test_soft_ids(self):
        r"""Tests soft ids.
        """
        inputs = torch.rand(self.batch_size, self.max_length, 32000)

        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        regressor = XLNetRegressor(hparams=hparams)
        preds = regressor(inputs)
        self.assertEqual(preds.shape, torch.Size([self.batch_size]))


if __name__ == "__main__":
    unittest.main()
