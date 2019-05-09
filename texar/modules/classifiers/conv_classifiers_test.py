#
"""
Unit tests for conv encoders.
"""
import torch
from torch import nn

import unittest

from texar.modules.classifiers.conv_classifiers import Conv1DClassifier


class Conv1DClassifierTest(unittest.TestCase):
    """Tests :class:`~texar.modules.Conv1DClassifier` class.
    """

    def test_classifier(self):
        """Tests classification.
        """
        # case 1: default hparams
        classifier = Conv1DClassifier()

        self.assertEqual(len(classifier.layers), 5)
        self.assertTrue(isinstance(classifier.layers[-1], nn.Linear))
        inputs = torch.randn(128, 32, 300)
        logits, pred = classifier(inputs)
        self.assertEqual(logits.shape, torch.Size([128, 2]))
        self.assertEqual(pred.shape, torch.Size([128]))

        # case 1
        hparams = {
            "num_classes": 10,
            "logit_layer_kwargs": {"bias": False}
        }
        classifier = Conv1DClassifier(hparams=hparams)
        inputs = torch.randn(128, 32, 300)
        logits, pred = classifier(inputs)
        self.assertEqual(logits.shape, torch.Size([128, 10]))
        self.assertEqual(pred.shape, torch.Size([128]))


if __name__ == "__main__":
    unittest.main()
