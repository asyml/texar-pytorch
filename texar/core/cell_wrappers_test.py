"""
Unit tests of :mod:`texar.core.cell_wrappers` and
:func:`~texar.core.layers.get_rnn_cell`.
"""

import unittest

import torch
from torch import nn

import texar.core.cell_wrappers as wrappers
from texar import HParams
from texar.core.layers import get_rnn_cell, default_rnn_cell_hparams
from texar.utils import utils


# pylint: disable=too-many-locals, protected-access, unused-variable
# pylint: disable=redefined-builtin

class WrappersTest(unittest.TestCase):
    r"""Tests cell wrappers and :func:`~texar.core.layers.get_rnn_cell`.
    """

    def test_get_rnn_cell(self):
        r"""Tests the HParams class.
        """
        hparams = {
            'type': 'LSTMCell',
            'input_size': 10,
            'kwargs': {
                'hidden_size': 20,
            },
            'num_layers': 3,
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5,
                'state_keep_prob': 0.5,
                'variational_recurrent': True
            },
            'residual': True,
            'highway': True,
        }
        hparams = HParams(hparams, default_rnn_cell_hparams())

        rnn_cell = get_rnn_cell(hparams)
        self.assertIsInstance(rnn_cell, wrappers.MultiRNNCell)
        self.assertEqual(len(rnn_cell._cell), hparams.num_layers)
        self.assertEqual(rnn_cell.input_size, hparams.input_size)
        self.assertEqual(rnn_cell.hidden_size, hparams.kwargs.hidden_size)

        for idx, cell in enumerate(rnn_cell._cell):
            input_size = (hparams.input_size if idx == 0
                          else hparams.kwargs.hidden_size)
            self.assertEqual(cell.input_size, input_size)
            self.assertEqual(cell.hidden_size, hparams.kwargs.hidden_size)

            if idx > 0:
                highway = cell
                residual = highway._cell
                dropout = residual._cell
                self.assertIsInstance(highway, wrappers.HighwayWrapper)
                self.assertIsInstance(residual, wrappers.ResidualWrapper)
            else:
                dropout = cell
            lstm = dropout._cell
            builtin_lstm = lstm._cell
            self.assertIsInstance(dropout, wrappers.DropoutWrapper)
            self.assertIsInstance(lstm, wrappers.LSTMCell)
            self.assertIsInstance(builtin_lstm, nn.LSTMCell)

            for key in ['input', 'output', 'state']:
                self.assertEqual(getattr(dropout, f'_{key}_keep_prob'),
                                 hparams.dropout[f'{key}_keep_prob'])
            self.assertTrue(dropout._variational_recurrent)

        batch_size = 8
        seq_len = 6
        state = None
        for step in range(seq_len):
            input = torch.zeros(batch_size, hparams.input_size)
            output, state = rnn_cell(input, state)
            self.assertEqual(
                output.shape, (batch_size, hparams.kwargs.hidden_size))
            self.assertEqual(len(state), hparams.num_layers)
            utils.map_structure(lambda s: self.assertEqual(
                s.shape, (batch_size, hparams.kwargs.hidden_size)), state)


if __name__ == "__main__":
    unittest.main()
