# -*- coding: utf-8 -*-
#
"""
Unit tests for RNN encoders.
"""

import unittest

import torch

from texar.torch.modules.encoders.rnn_encoders import (
    UnidirectionalRNNEncoder, BidirectionalRNNEncoder)


class UnidirectionalRNNEncoderTest(unittest.TestCase):
    r"""Tests unidirectional rnn encoder.
    """

    def setUp(self):
        self._batch_size = 16
        self._max_time = 8
        self._input_size = 10

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)

        # case 1
        encoder = UnidirectionalRNNEncoder(input_size=self._input_size)
        output, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 4)
        self.assertEqual(output.size()[-1], encoder.output_size)

        # case 2
        hparams = {
            "rnn_cell": {
                "dropout": {
                    "input_keep_prob": 0.5
                }
            }
        }
        encoder = UnidirectionalRNNEncoder(input_size=self._input_size,
                                           hparams=hparams)
        output, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 4)
        self.assertEqual(output.size()[-1], encoder.output_size)

        # case 3
        hparams = {"output_layer": {
            "num_layers": 2,
            "layer_size": [100, 6],
            "activation": "ReLU",
            "final_layer_activation": "Identity",
            "dropout_layer_ids": [0, 1, 2],
            "variational_dropout": False}}

        encoder = UnidirectionalRNNEncoder(input_size=self._input_size,
                                           hparams=hparams)

        output, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 8)
        self.assertEqual(output.size()[-1], encoder.output_size)

    def test_encode(self):
        r"""Tests encoding.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)

        # case 1
        encoder = UnidirectionalRNNEncoder(input_size=self._input_size)
        outputs, state = encoder(inputs)

        cell_dim = encoder.hparams.rnn_cell.kwargs.num_units

        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                    self._max_time,
                                                    cell_dim]))
        self.assertIsInstance(state, tuple)
        self.assertEqual(state[0].shape, torch.Size([self._batch_size,
                                                     cell_dim]))
        self.assertEqual(state[1].shape, torch.Size([self._batch_size,
                                                     cell_dim]))

        # case 2: with output layers
        hparams = {
            "output_layer": {
                "num_layers": 2,
                "layer_size": [100, 6],
                "dropout_layer_ids": [0, 1, 2],
                "variational_dropout": False
            }
        }
        encoder = UnidirectionalRNNEncoder(input_size=self._input_size,
                                           hparams=hparams)
        outputs, state, cell_outputs, output_size = encoder(
            inputs, return_cell_output=True, return_output_size=True)

        self.assertEqual(output_size, 6)
        self.assertEqual(cell_outputs.shape[-1], encoder.cell.hidden_size)

        out_dim = encoder.hparams.output_layer.layer_size[-1]
        self.assertEqual(outputs.shape, torch.Size([self._batch_size,
                                                    self._max_time,
                                                    out_dim]))


class BidirectionalRNNEncoderTest(unittest.TestCase):
    r"""Tests bidirectional rnn encoder.
    """

    def setUp(self):
        self._batch_size = 16
        self._max_time = 8
        self._input_size = 10

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)

        # case 1
        encoder = BidirectionalRNNEncoder(input_size=self._input_size)
        outputs, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 8)
        self.assertEqual(
            (outputs[0].size(-1), outputs[1].size(-1)), (encoder.output_size))

        # case 2
        hparams = {
            "rnn_cell_fw": {
                "dropout": {
                    "input_keep_prob": 0.5
                }
            }
        }
        encoder = BidirectionalRNNEncoder(input_size=self._input_size,
                                          hparams=hparams)
        outputs, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 8)
        self.assertEqual(
            (outputs[0].size(-1), outputs[1].size(-1)), (encoder.output_size))

        # case 3
        hparams = {
            "output_layer_fw": {
                "num_layers": 2,
                "layer_size": [100, 6],
                "activation": "ReLU",
                "final_layer_activation": "Identity",
                "dropout_layer_ids": [0, 1, 2],
                "variational_dropout": False
            },
            "output_layer_bw": {
                "num_layers": 3,
                "other_dense_kwargs": {"bias": False}
            },
            "output_layer_share_config": False
        }
        encoder = BidirectionalRNNEncoder(input_size=self._input_size,
                                          hparams=hparams)
        outputs, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 8 + 3 + 4)
        self.assertEqual(
            (outputs[0].size(-1), outputs[1].size(-1)), (encoder.output_size))

    def test_encode(self):
        r"""Tests encoding.
        """
        inputs = torch.rand(self._batch_size, self._max_time, self._input_size)

        # case 1
        encoder = BidirectionalRNNEncoder(input_size=self._input_size)
        outputs, state = encoder(inputs)

        cell_dim = encoder.hparams.rnn_cell_fw.kwargs.num_units

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       cell_dim]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       cell_dim]))

        self.assertIsInstance(state, tuple)
        self.assertIsInstance(state[0], tuple)
        self.assertEqual(state[0][0].shape, torch.Size([self._batch_size,
                                                        cell_dim]))
        self.assertEqual(state[0][0].shape, torch.Size([self._batch_size,
                                                        cell_dim]))
        self.assertIsInstance(state[1], tuple)
        self.assertEqual(state[1][0].shape, torch.Size([self._batch_size,
                                                        cell_dim]))
        self.assertEqual(state[1][0].shape, torch.Size([self._batch_size,
                                                        cell_dim]))

        # case 2:
        hparams = {
            "output_layer_fw": {
                "num_layers": 2,
                "layer_size": [100, 6],
                "dropout_layer_ids": [0, 1, 2],
                "variational_dropout": False
            }
        }
        encoder = BidirectionalRNNEncoder(input_size=self._input_size,
                                          hparams=hparams)
        outputs, state, cell_outputs, output_size = encoder(
            inputs, return_cell_output=True, return_output_size=True)

        self.assertEqual(output_size[0], 6)
        self.assertEqual(output_size[1], 6)
        self.assertEqual(cell_outputs[0].shape[-1], encoder.cell_fw.hidden_size)
        self.assertEqual(cell_outputs[1].shape[-1], encoder.cell_bw.hidden_size)

        out_dim = encoder.hparams.output_layer_fw.layer_size[-1]

        self.assertEqual(outputs[0].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       out_dim]))
        self.assertEqual(outputs[1].shape, torch.Size([self._batch_size,
                                                       self._max_time,
                                                       out_dim]))


if __name__ == "__main__":
    unittest.main()
