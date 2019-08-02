"""
Unit tests for BERT classifiers.
"""

import unittest

import torch

from texar.torch.modules.classifiers.bert_classifier import *
from texar.torch.utils.test import pretrained_test


class BERTClassifierTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.BERTClassifier` class.
    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        # case 1
        classifier = BERTClassifier(pretrained_model_name="bert-base-uncased")
        _, _ = classifier(self.inputs)

        # case 2
        classifier = BERTClassifier(pretrained_model_name="bert-large-uncased")
        _, _ = classifier(self.inputs)

        # case 3
        classifier = BERTClassifier(pretrained_model_name="bert-base-cased")
        _, _ = classifier(self.inputs)

        # case 4
        classifier = BERTClassifier(pretrained_model_name="bert-large-cased")
        _, _ = classifier(self.inputs)

        # case 5
        classifier = BERTClassifier(
            pretrained_model_name="bert-base-multilingual-uncased")
        _, _ = classifier(self.inputs)

        # case 6
        classifier = BERTClassifier(
            pretrained_model_name="bert-base-multilingual-cased")
        _, _ = classifier(self.inputs)

        # case 7
        classifier = BERTClassifier(pretrained_model_name="bert-base-chinese")
        _, _ = classifier(self.inputs)

        # case 8
        classifier = BERTClassifier(pretrained_model_name="roberta-base")
        _, _ = classifier(self.inputs)

        # case 9
        classifier = BERTClassifier(pretrained_model_name="roberta-large")
        _, _ = classifier(self.inputs)

        # case 10
        classifier = BERTClassifier(pretrained_model_name="roberta-large-mnli")
        _, _ = classifier(self.inputs)

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        classifier = BERTClassifier(hparams=hparams)
        self.assertEqual(len(classifier.trainable_variables), 199 + 2)
        _, _ = classifier(self.inputs)

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "all_time",
            "max_seq_length": 8,
        }
        classifier = BERTClassifier(hparams=hparams)
        self.assertEqual(len(classifier.trainable_variables), 199 + 2)
        _, _ = classifier(self.inputs)

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "time_wise",
        }
        classifier = BERTClassifier(hparams=hparams)
        self.assertEqual(len(classifier.trainable_variables), 199 + 2)
        _, _ = classifier(self.inputs)

    def test_classification(self):
        r"""Tests classification.
        """
        inputs = torch.randint(30521, (self.batch_size, self.max_length))

        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        classifier = BERTClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [self.batch_size, classifier.output_size]))
        self.assertEqual(preds.shape, torch.Size([self.batch_size]))

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 10,
            "clas_strategy": "time_wise",
        }
        classifier = BERTClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [self.batch_size, self.max_length, classifier.output_size]))
        self.assertEqual(preds.shape, torch.Size(
            [self.batch_size, self.max_length]))

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 0,
            "clas_strategy": "time_wise",
        }
        classifier = BERTClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [self.batch_size, self.max_length, classifier.output_size]))
        self.assertEqual(preds.shape, torch.Size(
            [self.batch_size, self.max_length]))

        # case 4
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 10,
            "clas_strategy": "all_time",
            "max_seq_length": self.max_length,
        }
        classifier = BERTClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [self.batch_size, classifier.output_size]))
        self.assertEqual(preds.shape, torch.Size([self.batch_size]))

    def test_binary(self):
        r"""Tests binary classification.
        """
        inputs = torch.randint(30521, (self.batch_size, self.max_length))

        # case 1
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "time_wise",
        }
        classifier = BERTClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [self.batch_size, self.max_length]))
        self.assertEqual(preds.shape, torch.Size(
            [self.batch_size, self.max_length]))

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "cls_time",
            "max_seq_length": self.max_length,
        }
        classifier = BERTClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([self.batch_size]))
        self.assertEqual(preds.shape, torch.Size([self.batch_size]))

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "all_time",
            "max_seq_length": self.max_length,
        }
        classifier = BERTClassifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([self.batch_size]))
        self.assertEqual(preds.shape, torch.Size([self.batch_size]))


if __name__ == "__main__":
    unittest.main()
