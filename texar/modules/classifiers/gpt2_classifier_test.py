"""
Unit tests for GPT2 classifier.
"""

import unittest

import torch

from texar.modules.classifiers.gpt2_classifier import *


@unittest.skip("Manual test only")
class GPT2ClassifierTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.GPT2Classifier` class.
    """

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1
        classifier = GPT2Classifier()
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 318)

        # case 2
        hparams = {
            "clas_strategy": "all_time",
            "max_seq_length": 8,
        }
        classifier = GPT2Classifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 318)

        # case 3
        hparams = {
            "clas_strategy": "time_wise",
        }
        classifier = GPT2Classifier(hparams=hparams)
        _, _ = classifier(inputs)
        self.assertEqual(len(classifier.trainable_variables), 318)

    def test_encode(self):
        r"""Tests encoding.
        """
        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (batch_size, max_time), dtype=torch.int64)

        # case 1
        classifier = GPT2Classifier()
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, classifier.hparams.num_classes]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))

        # case 2
        hparams = {
            "num_classes": 10,
            "clas_strategy": "time_wise"
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, max_time, classifier.hparams.num_classes]))
        self.assertEqual(preds.shape, torch.Size([batch_size, max_time]))

        # case 3
        hparams = {
            "num_classes": 0,
            "clas_strategy": "time_wise"
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, max_time, classifier.hparams.decoder.dim]))
        self.assertEqual(preds.shape, torch.Size([batch_size, max_time]))

        # case 4
        hparams = {
            "num_classes": 10,
            "clas_strategy": "all_time",
            "max_seq_length": max_time
        }
        inputs = torch.randint(30521, (batch_size, 6), dtype=torch.int64)
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size(
            [batch_size, classifier.hparams.num_classes]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))

    def test_binary(self):
        r"""Tests binary classification.
        """
        max_time = 8
        batch_size = 16
        inputs = torch.randint(30521, (batch_size, max_time), dtype=torch.int64)

        # case 1
        hparams = {
            "num_classes": 1,
            "clas_strategy": "time_wise"
        }
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([batch_size, max_time]))
        self.assertEqual(preds.shape, torch.Size([batch_size, max_time]))

        # case 2
        hparams = {
            "num_classes": 1,
            "clas_strategy": "cls_time",
            "max_seq_length": max_time
        }
        inputs = torch.randint(30521, (batch_size, 6), dtype=torch.int64)
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([batch_size]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))

        # case 3
        hparams = {
            "num_classes": 1,
            "clas_strategy": "all_time",
            "max_seq_length": max_time
        }
        inputs = torch.randint(30521, (batch_size, 6), dtype=torch.int64)
        classifier = GPT2Classifier(hparams=hparams)
        logits, preds = classifier(inputs)

        self.assertEqual(logits.shape, torch.Size([batch_size]))
        self.assertEqual(preds.shape, torch.Size([batch_size]))


if __name__ == "__main__":
    unittest.main()
