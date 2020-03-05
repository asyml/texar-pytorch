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
Unit tests for RNN classifiers.
"""

import unittest

import torch

from texar.torch.modules.classifiers.rnn_classifiers import *


class UnidirectionalRNNClassifierTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.UnidirectionalRNNClassifier` class.
    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.emb_dim = 4
        self.inputs = torch.rand(
            self.batch_size, self.max_length, self.emb_dim)

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        # case 1
        classifier = UnidirectionalRNNClassifier(input_size=self.emb_dim)
        output, _ = classifier(self.inputs)
        self.assertEqual(len(classifier.trainable_variables), 4 + 2)
        self.assertEqual(output.size()[-1], classifier.output_size)

        # case 2
        hparams = {
            "output_layer": {"num_layers": 2},
            "logit_layer_kwargs": {"bias": False}
        }
        classifier = UnidirectionalRNNClassifier(input_size=self.emb_dim,
                                                 hparams=hparams)
        output, _ = classifier(self.inputs)
        self.assertEqual(len(classifier.trainable_variables), 4 + 2 + 2 + 1)

    def test_encode(self):
        r"""Tests encoding.
        """
        # case 1
        classifier = UnidirectionalRNNClassifier(input_size=self.emb_dim)
        logits, pred = classifier(self.inputs)
        self.assertEqual(logits.shape,
                         torch.Size([self.batch_size,
                                     classifier.hparams.num_classes]))
        self.assertEqual(pred.shape, torch.Size([self.batch_size]))

        # case 2
        hparams = {
            "num_classes": 10,
            "clas_strategy": "time_wise"
        }
        classifier = UnidirectionalRNNClassifier(input_size=self.emb_dim,
                                                 hparams=hparams)
        logits, pred = classifier(self.inputs)
        self.assertEqual(logits.shape,
                         torch.Size([self.batch_size, self.max_length,
                                     classifier.hparams.num_classes]))
        self.assertEqual(pred.shape,
                         torch.Size([self.batch_size, self.max_length]))

        # case 3
        hparams = {
            "output_layer": {
                "num_layers": 1,
                "layer_size": 10
            },
            "num_classes": 0,
            "clas_strategy": "time_wise"
        }
        classifier = UnidirectionalRNNClassifier(input_size=self.emb_dim,
                                                 hparams=hparams)
        logits, pred = classifier(self.inputs)
        self.assertEqual(logits.shape,
                         torch.Size([self.batch_size, self.max_length, 10]))
        self.assertEqual(pred.shape,
                         torch.Size([self.batch_size, self.max_length]))

        # case 4
        hparams = {
            "num_classes": 10,
            "clas_strategy": "all_time",
            "max_seq_length": 5
        }
        classifier = UnidirectionalRNNClassifier(input_size=self.emb_dim,
                                                 hparams=hparams)
        logits, pred = classifier(self.inputs)
        self.assertEqual(logits.shape,
                         torch.Size([self.batch_size,
                                     classifier.hparams.num_classes]))
        self.assertEqual(pred.shape, torch.Size([self.batch_size]))

    def test_binary(self):
        r"""Tests binary classification.
        """
        # case 1
        hparams = {
            "num_classes": 1,
            "clas_strategy": "time_wise"
        }
        classifier = UnidirectionalRNNClassifier(input_size=self.emb_dim,
                                                 hparams=hparams)
        logits, pred = classifier(self.inputs)
        self.assertEqual(logits.shape,
                         torch.Size([self.batch_size, self.max_length]))
        self.assertEqual(pred.shape,
                         torch.Size([self.batch_size, self.max_length]))

        # case 2
        hparams = {
            "output_layer": {
                "num_layers": 1,
                "layer_size": 10
            },
            "num_classes": 1,
            "clas_strategy": "time_wise"
        }
        classifier = UnidirectionalRNNClassifier(input_size=self.emb_dim,
                                                 hparams=hparams)
        logits, pred = classifier(self.inputs)
        self.assertEqual(logits.shape,
                         torch.Size([self.batch_size, self.max_length]))
        self.assertEqual(pred.shape,
                         torch.Size([self.batch_size, self.max_length]))

        # case 3
        hparams = {
            "num_classes": 1,
            "clas_strategy": "all_time",
            "max_seq_length": 5
        }
        classifier = UnidirectionalRNNClassifier(input_size=self.emb_dim,
                                                 hparams=hparams)
        logits, pred = classifier(self.inputs)
        self.assertEqual(logits.shape,
                         torch.Size([self.batch_size]))
        self.assertEqual(pred.shape,
                         torch.Size([self.batch_size]))


if __name__ == "__main__":
    unittest.main()
