# -*- coding: utf-8 -*-
#
"""
Unit tests for metrics.
"""

# pylint: disable=invalid-name

import unittest

import torch

import texar as tx


class MetricsTest(unittest.TestCase):
    r"""Tests metrics.
    """

    def test_accuracy(self):
        r"""Tests :meth:`~texar.evals.accuracy`.
        """
        labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        preds = torch.tensor([1.0, 2.1, 3.0, 4, 5.2, 6, 8, 8])
        accuracy = tx.evals.accuracy(labels, preds)
        self.assertEqual(accuracy, 0.625)

    def test_binary_clas_accuracy(self):
        r"""Tests :meth:`~texar.evals.binary_clas_accuracy
        """
        pos_preds = torch.tensor([1, 1, 0, 0, 0])
        neg_preds = torch.tensor([1, 1, 0, 0, 0])

        accuracy = tx.evals.binary_clas_accuracy(pos_preds=pos_preds,
                                                 neg_preds=neg_preds)
        self.assertEqual(accuracy, 0.5)

        accuracy = tx.evals.binary_clas_accuracy(pos_preds=pos_preds,
                                                 neg_preds=None)
        self.assertEqual(accuracy, 0.4)

        accuracy = tx.evals.binary_clas_accuracy(pos_preds=None,
                                                 neg_preds=neg_preds)
        self.assertEqual(accuracy, 0.6)

        accuracy = tx.evals.binary_clas_accuracy(pos_preds=None,
                                                 neg_preds=None)
        self.assertEqual(accuracy, None)


if __name__ == "__main__":
    unittest.main()
