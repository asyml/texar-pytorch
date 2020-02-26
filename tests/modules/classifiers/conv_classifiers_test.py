# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
