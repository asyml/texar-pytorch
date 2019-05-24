"""
Unit tests for RNN decoders.
"""

# pylint: disable=invalid-name, not-callable, too-many-arguments
# pylint: disable=too-many-locals, protected-access

import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from texar import HParams
from texar.modules import get_helper
from texar.modules.decoders.rnn_decoders import BasicRNNDecoder, \
    BasicRNNDecoderOutput, AttentionRNNDecoder, AttentionRNNDecoderOutput


class BasicRNNDecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.decoders.rnn_decoders.BasicRNNDecoder`.
    """

    def setUp(self):
        self._vocab_size = 4
        self._max_time = 8
        self._batch_size = 16
        self._emb_dim = 20
        self._inputs = torch.rand(
            self._batch_size, self._max_time, self._emb_dim,
            dtype=torch.float)
        self._embedding = torch.rand(
            self._vocab_size, self._emb_dim, dtype=torch.float)
        self._hparams = HParams(None, BasicRNNDecoder.default_hparams())

    def _test_outputs(self, decoder, outputs, final_state, sequence_lengths,
                      test_mode=False, helper=None):
        hidden_size = decoder.hparams.rnn_cell.kwargs.num_units

        self.assertIsInstance(outputs, BasicRNNDecoderOutput)
        max_time = (self._max_time if not test_mode
                    else max(sequence_lengths).item())
        self.assertEqual(
            outputs.logits.shape,
            (self._batch_size, max_time, self._vocab_size))
        sample_id_shape = tuple() if helper is None else helper.sample_ids_shape
        self.assertEqual(
            outputs.sample_id.shape,
            (self._batch_size, max_time) + sample_id_shape)
        if not test_mode:
            np.testing.assert_array_equal(
                sequence_lengths, [max_time] * self._batch_size)
        self.assertEqual(final_state[0].shape, (self._batch_size, hidden_size))

    def test_decode_train(self):
        r"""Tests decoding in training mode.
        """
        decoder = BasicRNNDecoder(input_size=self._emb_dim,
                                  vocab_size=self._vocab_size,
                                  hparams=self._hparams)
        sequence_length = torch.tensor([self._max_time] * self._batch_size)

        # Helper by default HParams
        helper_train = decoder.create_helper()
        outputs, final_state, sequence_lengths = decoder(
            helper=helper_train,
            inputs=self._inputs,
            sequence_length=sequence_length)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths)

        # Helper by decoding strategy
        helper_train = decoder.create_helper(decoding_strategy='train_greedy')
        outputs, final_state, sequence_lengths = decoder(
            helper=helper_train,
            inputs=self._inputs,
            sequence_length=sequence_length)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths)

        # Implicit helper
        outputs, final_state, sequence_lengths = decoder(
            inputs=self._inputs,
            sequence_length=sequence_length)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths)

        # Eval helper through forward args
        outputs, final_state, sequence_lengths = decoder(
            embedding=self._embedding,
            start_tokens=torch.tensor([1] * self._batch_size),
            end_token=2,
            infer_mode=True)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths,
                           test_mode=True)

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
        decoder = BasicRNNDecoder(input_size=self._emb_dim,
                                  vocab_size=self._vocab_size,
                                  hparams=self._hparams)

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
        embedding = torch.randn(self._vocab_size, self._emb_dim)
        inputs = torch.randint(
            self._vocab_size, size=(self._batch_size, self._max_time))

        # decoder outputs
        helper_train = decoder.create_helper(embedding=embedding)
        outputs, final_state, sequence_lengths = decoder(
            inputs=inputs,
            sequence_length=input_lengths,
            helper=helper_train)

        # torch LSTM outputs
        lstm_inputs = F.embedding(inputs, embedding)
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
        decoder = BasicRNNDecoder(input_size=self._emb_dim,
                                  vocab_size=self._vocab_size,
                                  hparams=self._hparams)

        decoder.eval()
        start_tokens = torch.tensor([self._vocab_size - 2] * self._batch_size)

        helpers = []
        for strategy in ['infer_greedy', 'infer_sample']:
            helper = decoder.create_helper(
                decoding_strategy=strategy,
                embedding=self._embedding,
                start_tokens=start_tokens,
                end_token=self._vocab_size - 1)
            helpers.append(helper)
        for klass in ['TopKSampleEmbeddingHelper', 'SoftmaxEmbeddingHelper',
                      'GumbelSoftmaxEmbeddingHelper']:
            helper = get_helper(
                klass, embedding=self._embedding,
                start_tokens=start_tokens, end_token=self._vocab_size - 1,
                top_k=self._vocab_size // 2, tau=2.0,
                straight_through=True)
            helpers.append(helper)

        for helper in helpers:
            max_length = 100
            outputs, final_state, sequence_lengths = decoder(
                helper=helper, max_decoding_length=max_length)
            self.assertLessEqual(max(sequence_lengths), max_length)
            self._test_outputs(decoder, outputs, final_state, sequence_lengths,
                               test_mode=True, helper=helper)


class AttentionRNNDecoderTest(unittest.TestCase):
    """Tests :class:`~texar.modules.decoders.rnn_decoders.AttentionRNNDecoder`.
    """

    def setUp(self):
        self._vocab_size = 10
        self._max_time = 16
        self._batch_size = 8
        self._emb_dim = 20
        self._attention_dim = 256
        self._inputs = torch.rand(self._batch_size,
                                  self._max_time,
                                  self._emb_dim,
                                  dtype=torch.float)
        self._embedding = torch.rand(self._vocab_size,
                                     self._emb_dim,
                                     dtype=torch.float)
        self._encoder_output = torch.rand(self._batch_size,
                                          self._max_time,
                                          64)

    def _test_outputs(self, decoder, outputs, final_state, sequence_lengths,
                      test_mode=False, helper=None):
        hidden_size = decoder.hparams.rnn_cell.kwargs.num_units

        self.assertIsInstance(outputs, AttentionRNNDecoderOutput)
        max_time = (self._max_time if not test_mode
                    else max(sequence_lengths).item())
        self.assertEqual(
            outputs.logits.shape,
            (self._batch_size, max_time, self._vocab_size))
        sample_id_shape = tuple() if helper is None else helper.sample_ids_shape
        self.assertEqual(
            outputs.sample_id.shape,
            (self._batch_size, max_time) + sample_id_shape)  # TODO: Why?
        if not test_mode:
            np.testing.assert_array_equal(
                sequence_lengths, [max_time] * self._batch_size)
        self.assertEqual(final_state[0].shape, (self._batch_size, hidden_size))

    def test_decode_train(self):
        """Tests decoding in training mode.
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = torch.tensor(seq_length)
        hparams = {
            "attention": {
                "kwargs": {
                    "num_units": self._attention_dim
                }
            }
        }

        decoder = AttentionRNNDecoder(
            memory=self._encoder_output,
            memory_sequence_length=encoder_values_length,
            vocab_size=self._vocab_size,
            hparams=hparams)
        sequence_length = torch.tensor([self._max_time] * self._batch_size)

        helper_train = decoder.create_helper()
        outputs, final_state, sequence_lengths = decoder(
            helper=helper_train,
            inputs=self._inputs,
            sequence_length=sequence_length)
        self.assertEqual(len(decoder.trainable_variables), 5)

        self._test_outputs(decoder, outputs, final_state, sequence_lengths)

    def test_decode_infer(self):
        """Tests decoding in inference mode.
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = torch.tensor(seq_length)
        hparams = {
            "attention": {
                "kwargs": {
                    "num_units": 256,
                }
            }
        }

        decoder = AttentionRNNDecoder(
            memory=self._encoder_output,
            memory_sequence_length=encoder_values_length,
            vocab_size=self._vocab_size,
            hparams=hparams)
        decoder.eval()

        helper_infer = decoder.create_helper()
        outputs, final_state, sequence_lengths = decoder(
            helper=helper_infer,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2)
        self.assertEqual(len(decoder.trainable_variables), 5)

        self._test_outputs(decoder, outputs, final_state, sequence_lengths,
                           test_mode=True)


if __name__ == "__main__":
    unittest.main()
