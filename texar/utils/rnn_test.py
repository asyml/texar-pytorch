# -*- coding: utf-8 -*-
#
"""
Unit tests for rnn helpers.
"""

# pylint: disable=invalid-name

import unittest

import torch

from texar.core.cell_wrappers import RNNCell, GRUCell, LSTMCell
from texar.utils.rnn import dynamic_rnn


class DynamicRNNTest(unittest.TestCase):
    """Tests dynamic_rnn.
    """

    def setUp(self):
        self._batch_size = 8
        self._max_time = 64
        self._input_size = 16
        self._hidden_size = 32
        self._rnn = RNNCell(self._input_size, self._hidden_size)
        self._lstm = LSTMCell(self._input_size, self._hidden_size)
        self._gru = GRUCell(self._input_size, self._hidden_size)

    def test_dynamic_rnn_basic(self):
        """Tests :meth:`~texar.utils.rnn.dynamic_rnn`.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)

        # RNN
        outputs, final_state = dynamic_rnn(self._rnn,
                                           inputs,
                                           sequence_length=None,
                                           initial_state=None,
                                           time_major=False)

        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                   self._max_time,
                                                   self._hidden_size]))
        self.assertEqual(final_state.shape, torch.Size([self._batch_size,
                                                        self._hidden_size]))
        # LSTM
        outputs, final_state = dynamic_rnn(self._lstm,
                                           inputs,
                                           sequence_length=None,
                                           initial_state=None,
                                           time_major=False)

        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                    self._max_time,
                                                    self._hidden_size]))
        self.assertIsInstance(final_state, tuple)
        self.assertEqual(final_state[0].shape, torch.Size([self._batch_size,
                                                           self._hidden_size]))
        self.assertEqual(final_state[1].shape, torch.Size([self._batch_size,
                                                           self._hidden_size]))

        # GRU
        outputs, final_state = dynamic_rnn(self._gru,
                                           inputs,
                                           sequence_length=None,
                                           initial_state=None,
                                           time_major=False)

        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                   self._max_time,
                                                   self._hidden_size]))
        self.assertEqual(final_state.shape, torch.Size([self._batch_size,
                                                        self._hidden_size]))

    def test_dynamic_rnn_time_major(self):
        """Tests :meth:`~texar.utils.rnn.dynamic_rnn`.
        """
        inputs = torch.rand(self._max_time, self._batch_size, self._input_size)

        # RNN
        outputs, final_state = dynamic_rnn(self._rnn,
                                           inputs,
                                           sequence_length=None,
                                           initial_state=None,
                                           time_major=True)

        self.assertEqual(outputs.shape, torch.Size([self._max_time,
                                                    self._batch_size,
                                                    self._hidden_size]))
        self.assertEqual(final_state.shape, torch.Size([self._batch_size,
                                                        self._hidden_size]))
        # LSTM
        outputs, final_state = dynamic_rnn(self._lstm,
                                           inputs,
                                           sequence_length=None,
                                           initial_state=None,
                                           time_major=True)

        self.assertEqual(outputs.shape, torch.Size([self._max_time,
                                                    self._batch_size,
                                                    self._hidden_size]))
        self.assertIsInstance(final_state, tuple)
        self.assertEqual(final_state[0].shape, torch.Size([self._batch_size,
                                                           self._hidden_size]))
        self.assertEqual(final_state[1].shape, torch.Size([self._batch_size,
                                                           self._hidden_size]))

        # GRU
        outputs, final_state = dynamic_rnn(self._gru,
                                           inputs,
                                           sequence_length=None,
                                           initial_state=None,
                                           time_major=True)

        self.assertEqual(outputs.shape, torch.Size([self._max_time,
                                                    self._batch_size,
                                                    self._hidden_size]))
        self.assertEqual(final_state.shape, torch.Size([self._batch_size,
                                                        self._hidden_size]))

    def test_dynamic_rnn_sequence_length(self):
        """Tests :meth:`~texar.utils.rnn.dynamic_rnn`.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)
        sequence_length = [2, 43, 23, 63, 12, 54, 33, 8]

        # RNN
        outputs, final_state = dynamic_rnn(self._rnn,
                                           inputs,
                                           sequence_length=sequence_length,
                                           initial_state=None,
                                           time_major=False)

        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                    self._max_time,
                                                    self._hidden_size]))
        self.assertEqual(final_state.shape, torch.Size([self._batch_size,
                                                        self._hidden_size]))

        # LSTM
        outputs, final_state = dynamic_rnn(self._lstm,
                                           inputs,
                                           sequence_length=sequence_length,
                                           initial_state=None,
                                           time_major=False)

        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                    self._max_time,
                                                    self._hidden_size]))
        self.assertIsInstance(final_state, tuple)
        self.assertEqual(final_state[0].shape, torch.Size([self._batch_size,
                                                           self._hidden_size]))
        self.assertEqual(final_state[1].shape, torch.Size([self._batch_size,
                                                           self._hidden_size]))

        # GRU
        outputs, final_state = dynamic_rnn(self._gru,
                                           inputs,
                                           sequence_length=sequence_length,
                                           initial_state=None,
                                           time_major=False)

        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                    self._max_time,
                                                    self._hidden_size]))
        self.assertEqual(final_state.shape, torch.Size([self._batch_size,
                                                        self._hidden_size]))

    def test_dynamic_rnn_initial_state(self):
        """Tests :meth:`~texar.utils.rnn.dynamic_rnn`. 
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)

        rnn_initial_state = torch.rand(self._batch_size, self._hidden_size)
        lstm_initial_state = (torch.rand(self._batch_size, self._hidden_size),
                              torch.rand(self._batch_size, self._hidden_size))

        # RNN
        outputs, final_state = dynamic_rnn(self._rnn,
                                           inputs,
                                           sequence_length=None,
                                           initial_state=rnn_initial_state,
                                           time_major=False)

        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                    self._max_time,
                                                    self._hidden_size]))
        self.assertEqual(final_state.shape, torch.Size([self._batch_size,
                                                        self._hidden_size]))

        # LSTM
        outputs, final_state = dynamic_rnn(self._lstm,
                                           inputs,
                                           sequence_length=None,
                                           initial_state=lstm_initial_state,
                                           time_major=False)

        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                    self._max_time,
                                                    self._hidden_size]))
        self.assertIsInstance(final_state, tuple)
        self.assertEqual(final_state[0].shape, torch.Size([self._batch_size,
                                                           self._hidden_size]))
        self.assertEqual(final_state[1].shape, torch.Size([self._batch_size,
                                                           self._hidden_size]))

        # GRU
        outputs, final_state = dynamic_rnn(self._gru,
                                           inputs,
                                           sequence_length=None,
                                           initial_state=rnn_initial_state,
                                           time_major=False)

        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                    self._max_time,
                                                    self._hidden_size]))
        self.assertEqual(final_state.shape, torch.Size([self._batch_size,
                                                        self._hidden_size]))


if __name__ == "__main__":
    unittest.main()