"""
Unit tests for attention mechanism.
"""

import unittest

import numpy as np
import torch

from texar.torch.core.attention_mechanism import *


class AttentionMechanismTest(unittest.TestCase):
    r"""Tests attention mechanism.
    """

    def setUp(self):
        self._batch_size = 8
        self._max_time = 16
        self._encoder_output_size = 64
        self._attention_dim = 256
        self._memory = torch.rand(
            self._batch_size, self._max_time, self._encoder_output_size
        )
        self._memory_sequence_length = torch.tensor(
            np.random.randint(self._max_time, size=[self._batch_size]) + 1
        )
        self._attention_state = torch.rand(self._batch_size, self._max_time)

    def test_LuongAttention(self):
        r"""Tests `LuongAttention`
        """
        # Case 1
        attention_mechanism = LuongAttention(
            num_units=self._attention_dim,
            encoder_output_size=self._encoder_output_size)

        cell_output = torch.rand(self._batch_size, self._attention_dim)

        attention, alignments, next_attention_state = \
            compute_attention(
                attention_mechanism=attention_mechanism,
                cell_output=cell_output,
                attention_state=self._attention_state,
                memory=self._memory,
                attention_layer=None,
                memory_sequence_length=self._memory_sequence_length)

        self.assertEqual(attention.shape, torch.Size(
            [self._batch_size, self._encoder_output_size]))
        self.assertEqual(alignments.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(next_attention_state.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(len(attention_mechanism.trainable_variables), 1)

        # Case 2
        attention_mechanism = LuongAttention(
            num_units=self._attention_dim,
            encoder_output_size=self._encoder_output_size,
            scale=True)

        cell_output = torch.rand(self._batch_size, self._attention_dim)

        attention, alignments, next_attention_state = \
            compute_attention(
                attention_mechanism=attention_mechanism,
                cell_output=cell_output,
                attention_state=self._attention_state,
                memory=self._memory,
                attention_layer=None,
                memory_sequence_length=self._memory_sequence_length)

        self.assertEqual(attention.shape, torch.Size(
            [self._batch_size, self._encoder_output_size]))
        self.assertEqual(alignments.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(next_attention_state.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(len(attention_mechanism.trainable_variables), 2)

    def test_BahdanauAttention(self):
        r"""Tests BahdanauAttention
        """
        # Case 1
        attention_mechanism = BahdanauAttention(
            num_units=self._attention_dim,
            decoder_output_size=128,
            encoder_output_size=self._encoder_output_size)

        cell_output = torch.rand(self._batch_size, 128)

        attention, alignments, next_attention_state = \
            compute_attention(
                attention_mechanism=attention_mechanism,
                cell_output=cell_output,
                attention_state=self._attention_state,
                memory=self._memory,
                attention_layer=None,
                memory_sequence_length=self._memory_sequence_length)

        self.assertEqual(attention.shape, torch.Size(
            [self._batch_size, self._encoder_output_size]))
        self.assertEqual(alignments.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(next_attention_state.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(len(attention_mechanism.trainable_variables), 3)

        # Case 2
        attention_mechanism = BahdanauAttention(
            num_units=self._attention_dim,
            decoder_output_size=128,
            encoder_output_size=self._encoder_output_size,
            normalize=True)

        cell_output = torch.rand(self._batch_size, 128)

        attention, alignments, next_attention_state = \
            compute_attention(
                attention_mechanism=attention_mechanism,
                cell_output=cell_output,
                attention_state=self._attention_state,
                memory=self._memory,
                attention_layer=None,
                memory_sequence_length=self._memory_sequence_length)

        self.assertEqual(attention.shape, torch.Size(
            [self._batch_size, self._encoder_output_size]))
        self.assertEqual(alignments.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(next_attention_state.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(len(attention_mechanism.trainable_variables), 5)

    def test_LuongMonotonicAttention(self):
        r"""Tests LuongMonotonicAttention
        """
        # Case 1
        attention_mechanism = LuongMonotonicAttention(
            num_units=self._attention_dim,
            encoder_output_size=self._encoder_output_size)

        cell_output = torch.rand(self._batch_size, self._attention_dim)

        attention, alignments, next_attention_state = \
            compute_attention(
                attention_mechanism=attention_mechanism,
                cell_output=cell_output,
                attention_state=self._attention_state,
                memory=self._memory,
                attention_layer=None,
                memory_sequence_length=self._memory_sequence_length)

        self.assertEqual(attention.shape, torch.Size(
            [self._batch_size, self._encoder_output_size]))
        self.assertEqual(alignments.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(next_attention_state.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(len(attention_mechanism.trainable_variables), 2)

        # Case 2
        attention_mechanism = LuongMonotonicAttention(
            num_units=self._attention_dim,
            encoder_output_size=self._encoder_output_size,
            scale=True)

        cell_output = torch.rand(self._batch_size, self._attention_dim)

        attention, alignments, next_attention_state = \
            compute_attention(
                attention_mechanism=attention_mechanism,
                cell_output=cell_output,
                attention_state=self._attention_state,
                memory=self._memory,
                attention_layer=None,
                memory_sequence_length=self._memory_sequence_length)

        self.assertEqual(attention.shape, torch.Size(
            [self._batch_size, self._encoder_output_size]))
        self.assertEqual(alignments.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(next_attention_state.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(len(attention_mechanism.trainable_variables), 3)

    def test_BahdanauMonotonicAttention(self):
        r"""Tests BahdanauMonotonicAttention
        """
        # Case 1
        attention_mechanism = BahdanauMonotonicAttention(
            num_units=self._attention_dim,
            decoder_output_size=128,
            encoder_output_size=self._encoder_output_size)

        cell_output = torch.rand(self._batch_size, 128)

        attention, alignments, next_attention_state = \
            compute_attention(
                attention_mechanism=attention_mechanism,
                cell_output=cell_output,
                attention_state=self._attention_state,
                memory=self._memory,
                attention_layer=None,
                memory_sequence_length=self._memory_sequence_length)

        self.assertEqual(attention.shape, torch.Size(
            [self._batch_size, self._encoder_output_size]))
        self.assertEqual(alignments.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(next_attention_state.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(len(attention_mechanism.trainable_variables), 4)

        # Case 2
        attention_mechanism = BahdanauMonotonicAttention(
            num_units=self._attention_dim,
            decoder_output_size=128,
            encoder_output_size=self._encoder_output_size,
            normalize=True)

        cell_output = torch.rand(self._batch_size, 128)

        attention, alignments, next_attention_state = \
            compute_attention(
                attention_mechanism=attention_mechanism,
                cell_output=cell_output,
                attention_state=self._attention_state,
                memory=self._memory,
                attention_layer=None,
                memory_sequence_length=self._memory_sequence_length)

        self.assertEqual(attention.shape, torch.Size(
            [self._batch_size, self._encoder_output_size]))
        self.assertEqual(alignments.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(next_attention_state.shape, torch.Size(
            [self._batch_size, self._max_time]))
        self.assertEqual(len(attention_mechanism.trainable_variables), 6)


if __name__ == "__main__":
    unittest.main()
