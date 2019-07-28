# -*- coding: utf-8 -*-
#
"""
Unit tests for rnn helpers.
"""

import unittest

import torch

from texar.torch.core.cell_wrappers import RNNCell, GRUCell, LSTMCell
from texar.torch.utils.rnn import (
    dynamic_rnn, reverse_sequence, bidirectional_dynamic_rnn)


class ReverseSequenceTest(unittest.TestCase):
    r"""Tests reverse_sequence.
    """

    def setUp(self):
        self.inputs = [[[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                       [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]],
                       [[30, 31], [32, 33], [34, 35], [36, 37], [38, 39]],
                       [[40, 41], [42, 43], [44, 45], [46, 47], [48, 49]]]
        self.inputs = torch.tensor(self.inputs)

    def test_reverse_sequence(self):
        r"""Tests :meth:`~texar.torch.utils.rnn.reverse_sequence`.
        """
        seq_lengths_batch_first = torch.tensor([1, 2, 3, 4])
        output = reverse_sequence(inputs=self.inputs,
                                  seq_lengths=seq_lengths_batch_first,
                                  time_major=False)

        expect_out = [[[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                      [[22, 23], [20, 21], [24, 25], [26, 27], [28, 29]],
                      [[34, 35], [32, 33], [30, 31], [36, 37], [38, 39]],
                      [[46, 47], [44, 45], [42, 43], [40, 41], [48, 49]]]

        self.assertEqual(output.tolist(), expect_out)

        seq_lengths_time_first = torch.tensor([0, 1, 2, 3, 4])
        output = reverse_sequence(inputs=self.inputs,
                                  seq_lengths=seq_lengths_time_first,
                                  time_major=True)

        expect_out = [[[10, 11], [12, 13], [24, 25], [36, 37], [48, 49]],
                      [[20, 21], [22, 23], [14, 15], [26, 27], [38, 39]],
                      [[30, 31], [32, 33], [34, 35], [16, 17], [28, 29]],
                      [[40, 41], [42, 43], [44, 45], [46, 47], [18, 19]]]

        self.assertEqual(output.tolist(), expect_out)


class DynamicRNNTest(unittest.TestCase):
    r"""Tests dynamic_rnn.
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
        r"""Tests :meth:`~texar.torch.utils.rnn.dynamic_rnn`.
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
        r"""Tests :meth:`~texar.torch.utils.rnn.dynamic_rnn`.
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
        r"""Tests :meth:`~texar.torch.utils.rnn.dynamic_rnn`.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)
        sequence_length = [0, 43, 23, 63, 12, 54, 33, 8]

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
        r"""Tests :meth:`~texar.torch.utils.rnn.dynamic_rnn`.
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


class BidirectionalDynamicRNNTest(unittest.TestCase):
    r"""Tests bidirectional_dynamic_rnn.
    """

    def setUp(self):
        self._batch_size = 8
        self._max_time = 64
        self._input_size = 16
        self._hidden_size = 32
        self._rnn_fw = RNNCell(self._input_size, self._hidden_size)
        self._rnn_bw = RNNCell(self._input_size, self._hidden_size)
        self._lstm_fw = LSTMCell(self._input_size, self._hidden_size)
        self._lstm_bw = LSTMCell(self._input_size, self._hidden_size)
        self._gru_fw = GRUCell(self._input_size, self._hidden_size)
        self._gru_bw = GRUCell(self._input_size, self._hidden_size)

    def test_bidirectional_dynamic_rnn_basic(self):
        r"""Tests :meth:`~texar.torch.utils.rnn.bidirectional_dynamic_rnn`.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)

        # RNN
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._rnn_fw,
            cell_bw=self._rnn_bw,
            inputs=inputs,
            sequence_length=None,
            initial_state_fw=None,
            initial_state_bw=None,
            time_major=False)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertEqual(output_state[0].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))
        self.assertEqual(output_state[1].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))

        # LSTM
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._lstm_fw,
            cell_bw=self._lstm_bw,
            inputs=inputs,
            sequence_length=None,
            initial_state_fw=None,
            initial_state_bw=None,
            time_major=False)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertIsInstance(output_state[0], tuple)
        self.assertEqual(output_state[0][0].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertEqual(output_state[0][1].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertIsInstance(output_state[1], tuple)
        self.assertEqual(output_state[1][0].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertEqual(output_state[1][1].shape,
                         torch.Size([self._batch_size, self._hidden_size]))

        # GRU
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._gru_fw,
            cell_bw=self._gru_bw,
            inputs=inputs,
            sequence_length=None,
            initial_state_fw=None,
            initial_state_bw=None,
            time_major=False)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertEqual(output_state[0].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))
        self.assertEqual(output_state[1].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))

    def test_bidirectional_dynamic_rnn_time_major(self):
        r"""Tests :meth:`~texar.torch.utils.rnn.bidirectional_dynamic_rnn`.
        """
        inputs = torch.rand(self._max_time, self._batch_size, self._input_size)

        # RNN
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._rnn_fw,
            cell_bw=self._rnn_bw,
            inputs=inputs,
            sequence_length=None,
            initial_state_fw=None,
            initial_state_bw=None,
            time_major=True)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._max_time,
                                                       self._batch_size,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._max_time,
                                                       self._batch_size,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertEqual(output_state[0].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))
        self.assertEqual(output_state[1].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))

        # LSTM
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._lstm_fw,
            cell_bw=self._lstm_bw,
            inputs=inputs,
            sequence_length=None,
            initial_state_fw=None,
            initial_state_bw=None,
            time_major=True)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._max_time,
                                                       self._batch_size,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._max_time,
                                                       self._batch_size,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertIsInstance(output_state[0], tuple)
        self.assertEqual(output_state[0][0].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertEqual(output_state[0][1].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertIsInstance(output_state[1], tuple)
        self.assertEqual(output_state[1][0].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertEqual(output_state[1][1].shape,
                         torch.Size([self._batch_size, self._hidden_size]))

        # GRU
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._gru_fw,
            cell_bw=self._gru_bw,
            inputs=inputs,
            sequence_length=None,
            initial_state_fw=None,
            initial_state_bw=None,
            time_major=True)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._max_time,
                                                       self._batch_size,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._max_time,
                                                       self._batch_size,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertEqual(output_state[0].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))
        self.assertEqual(output_state[1].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))

    def test_bidirectional_dynamic_rnn_sequence_length(self):
        r"""Tests :meth:`~texar.torch.utils.rnn.bidirectional_dynamic_rnn`.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)
        sequence_length = [0, 43, 23, 63, 12, 54, 33, 8]

        # RNN
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._rnn_fw,
            cell_bw=self._rnn_bw,
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state_fw=None,
            initial_state_bw=None,
            time_major=False)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertEqual(output_state[0].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))
        self.assertEqual(output_state[1].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))

        # LSTM
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._lstm_fw,
            cell_bw=self._lstm_bw,
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state_fw=None,
            initial_state_bw=None,
            time_major=False)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertIsInstance(output_state[0], tuple)
        self.assertEqual(output_state[0][0].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertEqual(output_state[0][1].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertIsInstance(output_state[1], tuple)
        self.assertEqual(output_state[1][0].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertEqual(output_state[1][1].shape,
                         torch.Size([self._batch_size, self._hidden_size]))

        # GRU
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._gru_fw,
            cell_bw=self._gru_bw,
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state_fw=None,
            initial_state_bw=None,
            time_major=False)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertEqual(output_state[0].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))
        self.assertEqual(output_state[1].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))

    def test_bidirectional_dynamic_rnn_initial_state(self):
        r"""Tests :meth:`~texar.torch.utils.rnn.bidirectional_dynamic_rnn`.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)

        rnn_initial_state_fw = torch.rand(self._batch_size, self._hidden_size)
        rnn_initial_state_bw = torch.rand(self._batch_size, self._hidden_size)
        lstm_initial_state_fw = (torch.rand(self._batch_size,
                                            self._hidden_size),
                                 torch.rand(self._batch_size,
                                            self._hidden_size))
        lstm_initial_state_bw = (torch.rand(self._batch_size,
                                            self._hidden_size),
                                 torch.rand(self._batch_size,
                                            self._hidden_size))

        # RNN
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._rnn_fw,
            cell_bw=self._rnn_bw,
            inputs=inputs,
            sequence_length=None,
            initial_state_fw=rnn_initial_state_fw,
            initial_state_bw=rnn_initial_state_bw,
            time_major=False)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertEqual(output_state[0].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))
        self.assertEqual(output_state[1].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))

        # LSTM
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._lstm_fw,
            cell_bw=self._lstm_bw,
            inputs=inputs,
            sequence_length=None,
            initial_state_fw=lstm_initial_state_fw,
            initial_state_bw=lstm_initial_state_bw,
            time_major=False)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertIsInstance(output_state[0], tuple)
        self.assertEqual(output_state[0][0].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertEqual(output_state[0][1].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertIsInstance(output_state[1], tuple)
        self.assertEqual(output_state[1][0].shape,
                         torch.Size([self._batch_size, self._hidden_size]))
        self.assertEqual(output_state[1][1].shape,
                         torch.Size([self._batch_size, self._hidden_size]))

        # GRU
        outputs, output_state = bidirectional_dynamic_rnn(
            cell_fw=self._gru_fw,
            cell_bw=self._gru_bw,
            inputs=inputs,
            sequence_length=None,
            initial_state_fw=rnn_initial_state_fw,
            initial_state_bw=rnn_initial_state_bw,
            time_major=False)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       self._hidden_size]))

        self.assertIsInstance(output_state, tuple)
        self.assertEqual(output_state[0].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))
        self.assertEqual(output_state[1].shape, torch.Size([self._batch_size,
                                                            self._hidden_size]))


if __name__ == "__main__":
    unittest.main()
