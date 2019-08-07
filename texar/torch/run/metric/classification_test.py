import functools
import unittest

import numpy as np

from texar.torch.run.metric.classification import *
from texar.torch.utils.test import external_library_test


@external_library_test("sklearn")
class ClassificationMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n_classes = 10
        self.n_examples = 100
        self.batch_size = 2
        self.labels = np.random.randint(self.n_classes, size=self.n_examples)
        self.guesses = np.random.randint(self.n_classes, size=self.n_examples)

    def _test_metric(self, metric, reference_fn):
        for idx in range(0, self.n_examples, self.batch_size):
            end_idx = idx + self.batch_size
            metric.add(self.guesses[idx:end_idx], self.labels[idx:end_idx])
            value = metric.value()
            answer = reference_fn(self.labels[:end_idx], self.guesses[:end_idx])
            self.assertAlmostEqual(value, answer)

    def test_accuracy(self):
        from sklearn.metrics import accuracy_score
        metric = Accuracy(pred_name=None)
        self._test_metric(metric, accuracy_score)

    def test_precision(self):
        from sklearn.metrics import precision_score
        for mode in ["micro", "macro", "weighted"]:
            metric = Precision(mode=mode, pred_name=None)
            self._test_metric(
                metric, functools.partial(precision_score, average=mode))

    def test_recall(self):
        from sklearn.metrics import recall_score
        for mode in ["micro", "macro", "weighted"]:
            metric = Recall(mode=mode, pred_name=None)
            self._test_metric(
                metric, functools.partial(recall_score, average=mode))

    def test_f1(self):
        from sklearn.metrics import f1_score
        for mode in ["micro", "macro", "weighted"]:
            metric = F1(mode=mode, pred_name=None)
            self._test_metric(
                metric, functools.partial(f1_score, average=mode))
