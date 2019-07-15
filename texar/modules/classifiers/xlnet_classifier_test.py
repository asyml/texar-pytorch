"""
Unit tests for XLNet classifiers.
"""

import unittest

import torch

from texar.modules.classifiers.xlnet_classifier import *


@unittest.skip("Manual test only")
class XLNetClassifierTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.XLNetClassifier` class.
    """

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1
        classifier = XLNetClassifier()
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 362 + 4)

        # case 2
        hparams = {
            "use_projection": False
        }
        classifier = XLNetClassifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 362 + 2)

        # case 3
        hparams = {
            "summary_type": "first"
        }
        classifier = XLNetClassifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 362 + 4)

        # case 4
        hparams = {
            "summary_type": "mean"
        }
        classifier = XLNetClassifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 362 + 4)

    def test_classification(self):
        r"""Tests classification.
        """
        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (max_time, batch_size), dtype=torch.int64)

        # case 1
        classifier = XLNetClassifier()
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, classifier.hparams.num_classes]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))

        # case 2
        hparams = {
            "num_classes": 10,
        }
        classifier = XLNetClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, classifier.hparams.num_classes]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))

        # case 3
        hparams = {
            "num_classes": 0,
        }
        classifier = XLNetClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, classifier.hparams.hidden_dim]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))

    def test_binary(self):
        r"""Tests binary classification.
        """
        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (max_time, batch_size), dtype=torch.int64)

        # case 1
        hparams = {
            "num_classes": 1,
        }
        classifier = XLNetClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([batch_size]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))


if __name__ == "__main__":
    unittest.main()
