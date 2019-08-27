import functools
import unittest

import numpy as np

from texar.torch.run.executor import make_deterministic
from texar.torch.run.metric.classification import *
from texar.torch.utils.test import external_library_test


@external_library_test("sklearn")
class ClassificationMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        make_deterministic(0)
        self.n_classes = 10
        self.n_examples = 300
        self.batch_size = 2
        self.labels = np.random.randint(self.n_classes, size=self.n_examples)
        self.guesses = np.random.randint(self.n_classes, size=self.n_examples)
        self.binary_labels = np.random.randint(2, size=self.n_examples)
        self.binary_guesses = np.random.randint(2, size=self.n_examples)

    def _test_metric(self, metric, reference_fn, binary=False):
        guesses = self.binary_guesses if binary else self.guesses
        labels = self.binary_labels if binary else self.labels
        for idx in range(0, self.n_examples, self.batch_size):
            end_idx = idx + self.batch_size
            metric.add(guesses[idx:end_idx], labels[idx:end_idx])
            value = metric.value()
            answer = reference_fn(labels[:end_idx], guesses[:end_idx])
            self.assertAlmostEqual(value, answer)

    def test_accuracy(self):
        from sklearn.metrics import accuracy_score
        metric = Accuracy(pred_name="")
        self._test_metric(metric, accuracy_score)

    def test_precision(self):
        from sklearn.metrics import precision_score
        for mode in Precision._valid_modes:
            metric = Precision(mode=mode, pos_label=1, pred_name="")
            self._test_metric(
                metric, functools.partial(precision_score, average=mode),
                binary=(mode == 'binary'))

    def test_recall(self):
        from sklearn.metrics import recall_score
        for mode in Recall._valid_modes:
            metric = Recall(mode=mode, pos_label=1, pred_name="")
            self._test_metric(
                metric, functools.partial(recall_score, average=mode),
                binary=(mode == 'binary'))

    def test_f1(self):
        from sklearn.metrics import f1_score
        for mode in F1._valid_modes:
            metric = F1(mode=mode, pos_label=1, pred_name="")
            self._test_metric(
                metric, functools.partial(f1_score, average=mode),
                binary=(mode == 'binary'))
