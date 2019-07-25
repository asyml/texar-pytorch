"""
Unit tests for XLNet classifiers.
"""

import unittest

import torch

from texar.modules.classifiers.xlnet_classifier import *


class XLNetClassifierTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.XLNetClassifier` class.
    """

    @unittest.skip("Manual test only")
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1
        classifier = XLNetClassifier(pretrained_model_name="xlnet-base-cased")
        _, _ = classifier(inputs)

        # case 2
        classifier = XLNetClassifier(pretrained_model_name="xlnet-large-cased")
        _, _ = classifier(inputs)

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        classifier = XLNetClassifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 182 + 4)

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "use_projection": False
        }
        classifier = XLNetClassifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 182 + 2)

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "all_time",
            "max_seq_length": 8
        }
        classifier = XLNetClassifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 182 + 4)

        # case 4
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "time_wise"
        }
        classifier = XLNetClassifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 182 + 4)

    def test_classification(self):
        r"""Tests classification.
        """
        max_time = 8
        batch_size = 16
        inputs = torch.randint(32000, (batch_size, max_time), dtype=torch.int64)

        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        classifier = XLNetClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, classifier.hparams.num_classes]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 10,
            "clas_strategy": "time_wise"
        }
        classifier = XLNetClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, max_time, classifier.hparams.num_classes]))
        self.assertEqual(preds.shape, torch.Size([batch_size, max_time]))

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 0,
            "clas_strategy": "time_wise"
        }
        classifier = XLNetClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, max_time, classifier.hparams.hidden_dim]))
        self.assertEqual(preds.shape, torch.Size([batch_size, max_time]))

        # case 4
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 10,
            "clas_strategy": "all_time",
            "max_seq_length": max_time
        }
        inputs = torch.randint(30521, (batch_size, 6), dtype=torch.int64)
        classifier = XLNetClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, classifier.hparams.num_classes]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))

    def test_binary(self):
        r"""Tests binary classification.
        """
        max_time = 8
        batch_size = 16
        inputs = torch.randint(32000, (batch_size, max_time), dtype=torch.int64)

        # case 1
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "time_wise"
        }
        classifier = XLNetClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([batch_size, max_time]))
        self.assertEqual(preds.shape, torch.Size([batch_size, max_time]))

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "cls_time",
            "max_seq_length": max_time
        }
        inputs = torch.randint(32000, (batch_size, 6), dtype=torch.int64)
        classifier = XLNetClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([batch_size]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "all_time",
            "max_seq_length": max_time
        }
        inputs = torch.randint(32000, (batch_size, 6), dtype=torch.int64)
        classifier = XLNetClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([batch_size]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))


if __name__ == "__main__":
    unittest.main()
