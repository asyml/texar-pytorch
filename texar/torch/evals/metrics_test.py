"""
Unit tests for metrics.
"""

import unittest

import torch

from texar.torch.evals import metrics


class MetricsTest(unittest.TestCase):
    r"""Tests metrics.
    """

    def test_accuracy(self):
        r"""Tests :meth:`~texar.torch.evals.accuracy`.
        """
        labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        preds = torch.tensor([1.0, 2.1, 3.0, 4, 5.2, 6, 8, 8])
        accuracy = metrics.accuracy(labels, preds)
        self.assertEqual(accuracy, 0.625)

    def test_binary_clas_accuracy(self):
        r"""Tests :meth:`~texar.torch.evals.binary_clas_accuracy
        """
        pos_preds = torch.tensor([1, 1, 0, 0, 0])
        neg_preds = torch.tensor([1, 1, 0, 0, 0])

        accuracy = metrics.binary_clas_accuracy(pos_preds, neg_preds)
        self.assertEqual(accuracy, 0.5)

        accuracy = metrics.binary_clas_accuracy(pos_preds, None)
        self.assertEqual(accuracy, 0.4)

        accuracy = metrics.binary_clas_accuracy(None, neg_preds)
        self.assertEqual(accuracy, 0.6)

        accuracy = metrics.binary_clas_accuracy(None, None)
        self.assertEqual(accuracy, None)


if __name__ == "__main__":
    unittest.main()
