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
Unit tests for summary related operations.
"""
import unittest

import numpy as np

from texar.torch.run.metric.summary import *


class RegressionMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n_examples = 100
        self.batch_size = 2
        self.values = np.random.randn(self.n_examples)

    def test_running_average(self):
        queue_size = 10
        metric = RunningAverage(queue_size)
        for idx in range(0, self.n_examples, self.batch_size):
            end_idx = idx + self.batch_size
            metric.add(self.values[idx:end_idx], None)
            value = metric.value()
            answer = self.values[max(0, end_idx - queue_size):end_idx].mean()
            self.assertAlmostEqual(value, answer)
