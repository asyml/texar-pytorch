"""
Unit tests for BERT classifiers.
"""

import unittest

import torch

from texar.modules.classifiers.bert_classifier import *


class BERTClassifierTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.BERTClassifier` class.
    """

    @unittest.skip("Manual test only")
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1
        classifier = BERTClassifier(pretrained_model_name="bert-base-uncased")
        _, _ = classifier(inputs)

        # case 2
        classifier = BERTClassifier(pretrained_model_name="bert-large-uncased")
        _, _ = classifier(inputs)

        # case 3
        classifier = BERTClassifier(pretrained_model_name="bert-base-cased")
        _, _ = classifier(inputs)

        # case 4
        classifier = BERTClassifier(pretrained_model_name="bert-large-cased")
        _, _ = classifier(inputs)

        # case 5
        classifier = BERTClassifier(
            pretrained_model_name="bert-base-multilingual-uncased")
        _, _ = classifier(inputs)

        # case 6
        classifier = BERTClassifier(
            pretrained_model_name="bert-base-multilingual-cased")
        _, _ = classifier(inputs)

        # case 7
        classifier = BERTClassifier(pretrained_model_name="bert-base-chinese")
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
        classifier = BERTClassifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 199 + 2)

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "all_time",
            "max_seq_length": 8,
        }
        classifier = BERTClassifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 199 + 2)

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "time_wise",
        }
        classifier = BERTClassifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 199 + 2)

    def test_classification(self):
        r"""Tests classification.
        """
        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (batch_size, max_time), dtype=torch.int64)

        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        classifier = BERTClassifier(hparams=hparams)
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
        classifier = BERTClassifier(hparams=hparams)
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
        classifier = BERTClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, max_time, classifier.hparams.encoder.dim]))
        self.assertEqual(preds.shape, torch.Size([batch_size, max_time]))

        # case 4
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 10,
            "clas_strategy": "all_time",
            "max_seq_length": max_time
        }
        inputs = torch.randint(30521, (batch_size, 6), dtype=torch.int64)
        classifier = BERTClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, classifier.hparams.num_classes]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))

    def _test_binary(self):
        r"""Tests binary classification.
        """
        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (batch_size, max_time), dtype=torch.int64)

        # case 1
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "time_wise"
        }
        classifier = BERTClassifier(hparams=hparams)
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
        inputs = torch.randint(30521, (batch_size, 6), dtype=torch.int64)
        classifier = BERTClassifier(hparams=hparams)
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
        inputs = torch.randint(30521, (batch_size, 6), dtype=torch.int64)
        classifier = BERTClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([batch_size]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))


if __name__ == "__main__":
    unittest.main()
