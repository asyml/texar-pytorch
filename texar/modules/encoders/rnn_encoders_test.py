# -*- coding: utf-8 -*-
#
"""
Unit tests for RNN encoders.
"""

import unittest

import torch

from texar.modules.encoders.rnn_encoders import UnidirectionalRNNEncoder, \
    BidirectionalRNNEncoder


class UnidirectionalRNNEncoderTest(unittest.TestCase):
    """Tests unidirectional rnn encoder.
    """

    def setUp(self):
        self._batch_size = 16
        self._max_time = 8
        self._input_size = 10

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)

        # case 1
        encoder = UnidirectionalRNNEncoder(input_size=self._input_size)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 4)


class BidirectionalRNNEncoderTest(unittest.TestCase):
    """Tests bidirectional rnn encoder.
    """

    def setUp(self):
        self._batch_size = 16
        self._max_time = 8
        self._input_size = 10

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)

        # case 1
        encoder = BidirectionalRNNEncoder(input_size=self._input_size)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 8)







if __name__ == "__main__":
    unittest.main()
