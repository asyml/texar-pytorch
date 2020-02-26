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
Unit tests for Transformer encoder.
"""
import unittest

import torch

from texar.torch.modules.encoders import TransformerEncoder


class TransformerEncoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.TransformerEncoder`
    """

    def setUp(self):
        self._batch_size = 2
        self._emb_dim = 512
        self._max_time = 7

    def test_trainable_variables(self):
        r"""Tests train_greedy
        """
        inputs = torch.rand(
            self._batch_size, self._max_time, self._emb_dim, dtype=torch.float)

        sequence_length = torch.randint(
            self._max_time, (self._batch_size,), dtype=torch.long)

        encoder = TransformerEncoder()

        outputs = encoder(inputs=inputs, sequence_length=sequence_length)

        # 6 blocks
        # -self multihead_attention: 4 dense without bias + 2 layer norm vars
        # -poswise_network: Dense with bias, Dense with bias + 2 layer norm vars
        # 2 output layer norm vars
        self.assertEqual(len(encoder.trainable_variables), 74)
        self.assertEqual(outputs.size(-1), encoder.output_size)

        hparams = {"use_bert_config": True}
        encoder = TransformerEncoder(hparams=hparams)

        # 6 blocks
        # -self multihead_attention: 4 dense without bias + 2 layer norm vars
        # -poswise_network: Dense with bias, Dense with bias + 2 layer norm vars
        # -output: 2 layer norm vars
        # 2 input layer norm vars
        outputs = encoder(inputs=inputs, sequence_length=sequence_length)
        self.assertEqual(len(encoder.trainable_variables), 74)
        self.assertEqual(outputs.size(-1), encoder.output_size)

    def test_encode(self):
        r"""Tests encoding.
        """
        inputs = torch.rand(
            self._batch_size, self._max_time, self._emb_dim, dtype=torch.float)

        sequence_length = torch.randint(
            self._max_time, (self._batch_size,), dtype=torch.long)

        encoder = TransformerEncoder()
        outputs = encoder(inputs=inputs, sequence_length=sequence_length)
        self.assertEqual(outputs.size(),
                         torch.Size((self._batch_size,
                                     self._max_time,
                                     self._emb_dim)))

        hparams = {"use_bert_config": True}
        encoder = TransformerEncoder(hparams=hparams)
        outputs = encoder(inputs=inputs, sequence_length=sequence_length)
        self.assertEqual(outputs.size(),
                         torch.Size((self._batch_size,
                                     self._max_time,
                                     self._emb_dim)))


if __name__ == "__main__":
    unittest.main()
