"""
Unit tests for GPT2 classifier.
"""

import unittest

import torch

from texar.torch.modules.classifiers.gpt2_classifier import *
from texar.torch.utils.test import pretrained_test


class GPT2ClassifierTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.GPT2Classifier` class.
    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""
        for pretrained_model_name in GPT2Classifier.available_checkpoints():
            classifier = GPT2Classifier(
                pretrained_model_name=pretrained_model_name)
            _, _ = classifier(self.inputs)

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        classifier = GPT2Classifier(hparams=hparams)
        self.assertEqual(len(classifier.trainable_variables), 198)
        _, _ = classifier(self.inputs)

        # case 2
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "all_time",
            "max_seq_length": 8,
        }
        classifier = GPT2Classifier(hparams=hparams)
        self.assertEqual(len(classifier.trainable_variables), 198)
        _, _ = classifier(self.inputs)

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "clas_strategy": "time_wise",
        }
        classifier = GPT2Classifier(hparams=hparams)
        self.assertEqual(len(classifier.trainable_variables), 198)
        _, _ = classifier(self.inputs)

    def test_classification(self):
        r"""Tests classificaiton.
        """
        inputs = torch.randint(30521, (self.batch_size, self.max_length))

        # case 1
        hparams = {
            "pretrained_model_name": None,
        }
        classifier = GPT2Classifier(hparams=hparams)
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
        classifier = GPT2Classifier(hparams=hparams)
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
        classifier = GPT2Classifier(hparams=hparams)
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
        classifier = GPT2Classifier(hparams=hparams)
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
        classifier = GPT2Classifier(hparams=hparams)
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
        classifier = GPT2Classifier(hparams=hparams)
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
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([self.batch_size]))
        self.assertEqual(preds.shape, torch.Size([self.batch_size]))

    def test_soft_ids(self):
        r"""Tests soft ids.
        """
        inputs = torch.rand(self.batch_size, self.max_length, 50257)
        hparams = {
            "pretrained_model_name": None,
            "num_classes": 1,
            "clas_strategy": "time_wise",
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [self.batch_size, self.max_length]))
        self.assertEqual(preds.shape, torch.Size(
            [self.batch_size, self.max_length]))


if __name__ == "__main__":
    unittest.main()
