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
Unit tests for regression related operations.
"""
import unittest

import numpy as np

from texar.torch.run.metric.regression import *
from texar.torch.utils.test import external_library_test


class RegressionMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n_examples = 100
        self.batch_size = 2
        self.labels = np.random.randn(self.n_examples)
        self.guesses = np.random.randn(self.n_examples)

    def _test_metric(self, metric, reference_fn):
        for idx in range(0, self.n_examples, self.batch_size):
            end_idx = idx + self.batch_size
            metric.add(self.guesses[idx:end_idx], self.labels[idx:end_idx])
            value = metric.value()
            answer = reference_fn(self.labels[:end_idx], self.guesses[:end_idx])
            self.assertAlmostEqual(value, answer)

    @external_library_test("scipy")
    def test_pearsonr(self):
        from scipy.stats import pearsonr
        metric = PearsonR(pred_name="")
        self._test_metric(metric, lambda *args: pearsonr(*args)[0])

    @external_library_test("sklearn")
    def test_rmse(self):
        from sklearn.metrics import mean_squared_error
        metric = RMSE(pred_name="")
        self._test_metric(
            metric, lambda *args: np.sqrt(mean_squared_error(*args)))
