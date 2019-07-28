"""
Unit tests for RNN decoders.
"""

import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from texar.torch.hyperparams import HParams
from texar.torch.modules.decoders.decoder_helpers import get_helper
from texar.torch.modules.decoders.rnn_decoders import (
    AttentionRNNDecoder, AttentionRNNDecoderOutput, BasicRNNDecoder,
    BasicRNNDecoderOutput)
from texar.torch.modules.embedders.embedders import WordEmbedder
from texar.torch.utils.utils import map_structure


class BasicRNNDecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.decoders.rnn_decoders.BasicRNNDecoder`.
    """

    def setUp(self):
        self._vocab_size = 4
        self._max_time = 8
        self._batch_size = 16
        self._emb_dim = 20
        self._inputs = torch.randint(
            self._vocab_size, size=(self._batch_size, self._max_time))
        embedding = torch.rand(
            self._vocab_size, self._emb_dim, dtype=torch.float)
        self._embedder = WordEmbedder(init_value=embedding)
        self._hparams = HParams(None, BasicRNNDecoder.default_hparams())

    def _test_outputs(self, decoder, outputs, final_state, sequence_lengths,
                      test_mode=False):
        hidden_size = decoder.hparams.rnn_cell.kwargs.num_units

        self.assertIsInstance(outputs, BasicRNNDecoderOutput)
        max_time = (self._max_time if not test_mode
                    else max(sequence_lengths).item())
        self.assertEqual(
            outputs.logits.shape,
            (self._batch_size, max_time, self._vocab_size))
        if not test_mode:
            np.testing.assert_array_equal(
                sequence_lengths, [max_time] * self._batch_size)
        self.assertEqual(final_state[0].shape, (self._batch_size, hidden_size))

    def test_decode_train(self):
        r"""Tests decoding in training mode.
        """
        decoder = BasicRNNDecoder(
            token_embedder=self._embedder, input_size=self._emb_dim,
            vocab_size=self._vocab_size, hparams=self._hparams)
        sequence_length = torch.tensor([self._max_time] * self._batch_size)

        # Helper by default HParams
        helper_train = decoder.create_helper()
        outputs, final_state, sequence_lengths = decoder(
            helper=helper_train, inputs=self._inputs,
            sequence_length=sequence_length)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths)

        # Helper by decoding strategy
        helper_train = decoder.create_helper(decoding_strategy='train_greedy')
        outputs, final_state, sequence_lengths = decoder(
            helper=helper_train, inputs=self._inputs,
            sequence_length=sequence_length)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths)

        # Implicit helper
        outputs, final_state, sequence_lengths = decoder(
            inputs=self._inputs, sequence_length=sequence_length)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths)

        # Eval helper through forward args
        outputs, final_state, sequence_lengths = decoder(
            embedding=self._embedder,
            start_tokens=torch.tensor([1] * self._batch_size),
            end_token=2, infer_mode=True)
        self._test_outputs(
            decoder, outputs, final_state, sequence_lengths, test_mode=True)

    @staticmethod
    def _assert_tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
        if torch.is_tensor(a):
            a = a.detach().numpy()
        if torch.is_tensor(b):
            b = b.detach().numpy()
        if any(np.issubdtype(array.dtype, np.floating) for array in [a, b]):
            return np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8)
        return np.testing.assert_array_equal(a, b)

    def test_decode_train_with_torch(self):
        r"""Compares decoding results with PyTorch built-in decoder.
        """
        decoder = BasicRNNDecoder(
            token_embedder=self._embedder, input_size=self._emb_dim,
            vocab_size=self._vocab_size, hparams=self._hparams)

        input_size = self._emb_dim
        hidden_size = decoder.hparams.rnn_cell.kwargs.num_units
        num_layers = decoder.hparams.rnn_cell.num_layers
        torch_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True)

        # match parameters
        for name in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
            setattr(torch_lstm, f'{name}_l0',
                    getattr(decoder._cell._cell, name))
        torch_lstm.flatten_parameters()

        output_layer = decoder._output_layer
        input_lengths = torch.tensor([self._max_time] * self._batch_size)
        inputs = torch.randint(
            self._vocab_size, size=(self._batch_size, self._max_time))

        # decoder outputs
        helper_train = decoder.create_helper()
        outputs, final_state, sequence_lengths = decoder(
            inputs=inputs,
            sequence_length=input_lengths,
            helper=helper_train)

        # torch LSTM outputs
        lstm_inputs = F.embedding(inputs, self._embedder.embedding)
        torch_outputs, torch_states = torch_lstm(lstm_inputs)
        torch_outputs = output_layer(torch_outputs)
        torch_sample_id = torch.argmax(torch_outputs, dim=-1)

        self.assertEqual(final_state[0].shape,
                         (self._batch_size, hidden_size))

        self._assert_tensor_equal(outputs.logits, torch_outputs)
        self._assert_tensor_equal(outputs.sample_id, torch_sample_id)
        self._assert_tensor_equal(final_state[0], torch_states[0].squeeze(0))
        self._assert_tensor_equal(final_state[1], torch_states[1].squeeze(0))
        self._assert_tensor_equal(sequence_lengths, input_lengths)

    def test_decode_infer(self):
        r"""Tests decoding in inference mode."""
        decoder = BasicRNNDecoder(
            token_embedder=self._embedder, input_size=self._emb_dim,
            vocab_size=self._vocab_size, hparams=self._hparams)

        decoder.eval()
        start_tokens = torch.tensor([self._vocab_size - 2] * self._batch_size)

        helpers = []
        for strategy in ['infer_greedy', 'infer_sample']:
            helper = decoder.create_helper(
                decoding_strategy=strategy,
                start_tokens=start_tokens,
                end_token=self._vocab_size - 1)
            helpers.append(helper)
        for klass in ['TopKSampleEmbeddingHelper', 'SoftmaxEmbeddingHelper',
                      'GumbelSoftmaxEmbeddingHelper']:
            helper = get_helper(
                klass, start_tokens=start_tokens,
                end_token=self._vocab_size - 1,
                top_k=self._vocab_size // 2, tau=2.0,
                straight_through=True)
            helpers.append(helper)

        for helper in helpers:
            max_length = 100
            outputs, final_state, sequence_lengths = decoder(
                helper=helper, max_decoding_length=max_length)
            self.assertLessEqual(max(sequence_lengths), max_length)
            self._test_outputs(decoder, outputs, final_state, sequence_lengths,
                               test_mode=True)


class AttentionRNNDecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.decoders.rnn_decoders.AttentionRNNDecoder`.
    """

    def setUp(self):
        self._vocab_size = 10
        self._max_time = 16
        self._batch_size = 8
        self._emb_dim = 20
        self._attention_dim = 256
        self._inputs = torch.randint(
            self._vocab_size, size=(self._batch_size, self._max_time))
        embedding = torch.rand(
            self._vocab_size, self._emb_dim, dtype=torch.float)
        self._embedder = WordEmbedder(init_value=embedding)
        self._encoder_output = torch.rand(
            self._batch_size, self._max_time, 64)

        self._test_hparams = {}  # (cell_type, is_multi) -> hparams
        for cell_type in ["RNNCell", "LSTMCell", "GRUCell"]:
            hparams = {
                "rnn_cell": {
                    'type': cell_type,
                    'kwargs': {
                        'num_units': 256,
                    },
                },
                "attention": {
                    "kwargs": {
                        "num_units": self._attention_dim
                    },
                }
            }
            self._test_hparams[(cell_type, False)] = HParams(
                hparams, AttentionRNNDecoder.default_hparams())

        hparams = {
            "rnn_cell": {
                'type': 'LSTMCell',
                'kwargs': {
                    'num_units': 256,
                },
                'num_layers': 3,
            },
            "attention": {
                "kwargs": {
                    "num_units": self._attention_dim
                },
            }
        }
        self._test_hparams[("LSTMCell", True)] = HParams(
            hparams, AttentionRNNDecoder.default_hparams())

    def _test_outputs(self, decoder, outputs, final_state, sequence_lengths,
                      test_mode=False):
        hidden_size = decoder.hparams.rnn_cell.kwargs.num_units
        cell_type = decoder.hparams.rnn_cell.type
        is_multi = decoder.hparams.rnn_cell.num_layers > 1

        self.assertIsInstance(outputs, AttentionRNNDecoderOutput)
        max_time = (self._max_time if not test_mode
                    else max(sequence_lengths).item())
        self.assertEqual(
            outputs.logits.shape,
            (self._batch_size, max_time, self._vocab_size))
        if not test_mode:
            np.testing.assert_array_equal(
                sequence_lengths, [max_time] * self._batch_size)

        map_structure(
            lambda t: self.assertEqual(
                t.size(), (self._batch_size, hidden_size)),
            final_state.cell_state)
        state = final_state.cell_state
        if is_multi:
            self.assertIsInstance(state, list)
            state = state[0]
        if cell_type == "LSTMCell":
            self.assertIsInstance(state, tuple)
            state = state[0]
        self.assertIsInstance(state, torch.Tensor)

    def test_decode_infer(self):
        r"""Tests decoding in inference mode.
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = torch.tensor(seq_length)

        for (cell_type, is_multi), hparams in self._test_hparams.items():
            decoder = AttentionRNNDecoder(
                encoder_output_size=64,
                token_embedder=self._embedder,
                vocab_size=self._vocab_size,
                input_size=self._emb_dim,
                hparams=hparams)

            decoder.eval()

            helper_infer = decoder.create_helper(
                start_tokens=torch.tensor([1] * self._batch_size), end_token=2)

            outputs, final_state, sequence_lengths = decoder(
                memory=self._encoder_output,
                memory_sequence_length=encoder_values_length,
                helper=helper_infer)

            self._test_outputs(decoder, outputs, final_state, sequence_lengths,
                               test_mode=True)

    def test_decode_train(self):
        r"""Tests decoding in training mode.
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = torch.tensor(seq_length)

        for (cell_type, is_multi), hparams in self._test_hparams.items():
            decoder = AttentionRNNDecoder(
                encoder_output_size=64,
                token_embedder=self._embedder,
                vocab_size=self._vocab_size,
                input_size=self._emb_dim,
                hparams=hparams)

            sequence_length = torch.tensor([self._max_time] * self._batch_size)

            helper_train = decoder.create_helper()
            outputs, final_state, sequence_lengths = decoder(
                memory=self._encoder_output,
                memory_sequence_length=encoder_values_length,
                helper=helper_train,
                inputs=self._inputs,
                sequence_length=sequence_length)

            self._test_outputs(decoder, outputs, final_state, sequence_lengths)


if __name__ == "__main__":
    unittest.main()
