#
"""
Unit tests for conv encoders.
"""
import unittest

import torch
from torch import nn

from texar.torch.modules.classifiers.conv_classifiers import Conv1DClassifier


class Conv1DClassifierTest(unittest.TestCase):
    """Tests :class:`~texar.torch.modules.Conv1DClassifier` class.
    """

    def test_classifier(self):
        """Tests classification.
        """
        # case 1: default hparams
        inputs = torch.randn(128, 32, 300)
        classifier = Conv1DClassifier(in_channels=inputs.shape[1],
                                      in_features=inputs.shape[2])

        self.assertEqual(len(classifier.layers), 5)
        self.assertIsInstance(classifier.layers[-1], nn.Linear)
        logits, pred = classifier(inputs)
        self.assertEqual(logits.shape, torch.Size([128, 2]))
        self.assertEqual(pred.shape, torch.Size([128]))

        # case 2
        inputs = torch.randn(128, 32, 300)
        hparams = {
            "num_classes": 10,
            "logit_layer_kwargs": {"bias": False}
        }
        classifier = Conv1DClassifier(in_channels=inputs.shape[1],
                                      in_features=inputs.shape[2],
                                      hparams=hparams)
        logits, pred = classifier(inputs)
        self.assertEqual(logits.shape, torch.Size([128, 10]))
        self.assertEqual(pred.shape, torch.Size([128]))

    def test_classifier_with_no_dense_layer(self):
        """Tests classifier with no dense layer.
        """
        inputs = torch.randn(128, 32, 300)
        hparams = {
            "num_dense_layers": 0,
            "num_classes": 10,
            "logit_layer_kwargs": {"bias": False}
        }
        classifier = Conv1DClassifier(in_channels=inputs.shape[1],
                                      in_features=inputs.shape[2],
                                      hparams=hparams)
        logits, pred = classifier(inputs)
        self.assertEqual(logits.shape, torch.Size([128, 10]))
        self.assertEqual(pred.shape, torch.Size([128]))


if __name__ == "__main__":
    unittest.main()
