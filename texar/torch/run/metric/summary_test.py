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
