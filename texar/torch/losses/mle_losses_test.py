"""
Unit tests for mle losses.
"""

import unittest

import torch
import torch.nn.functional as F

from texar.torch.losses import mle_losses
from texar.torch.utils.shapes import get_rank


class MLELossesTest(unittest.TestCase):
    """Tests mle losses.
    """

    def setUp(self):
        self._batch_size = 64
        self._max_time = 16
        self._num_classes = 100
        self._labels = torch.randint(
            self._num_classes, (self._batch_size, self._max_time),
            dtype=torch.int64)
        one_hot = torch.eye(self._num_classes)
        one_hot_labels = F.embedding(self._labels, one_hot)
        self._one_hot_labels = torch.reshape(
            one_hot_labels, [self._batch_size, self._max_time, -1])
        self._logits = torch.rand(
            self._batch_size, self._max_time, self._num_classes)
        self._sequence_length = torch.randint(
            high=self._max_time, size=(self._batch_size,), dtype=torch.int64)

    def _test_sequence_loss(self, loss_fn, labels, logits, sequence_length):
        loss = loss_fn(labels, logits, sequence_length)
        rank = get_rank(loss)
        self.assertEqual(rank, 0)

        loss = loss_fn(labels, logits, sequence_length,
                       sum_over_timesteps=False)
        rank = get_rank(loss)
        self.assertEqual(rank, 1)
        self.assertEqual(loss.shape, torch.Size([self._max_time]))

        loss = loss_fn(
            labels, logits, sequence_length, sum_over_timesteps=False,
            average_across_timesteps=True, average_across_batch=False)
        rank = get_rank(loss)
        self.assertEqual(rank, 1)
        self.assertEqual(loss.shape, torch.Size([self._batch_size]))

        loss = loss_fn(
            labels, logits, sequence_length, sum_over_timesteps=False,
            average_across_batch=False)
        rank = get_rank(loss)
        self.assertEqual(rank, 2)
        self.assertEqual(loss.shape, torch.Size([self._batch_size,
                                                 self._max_time]))

        sequence_length_time = torch.randint(size=(self._max_time,),
                                             high=self._batch_size)
        loss = loss_fn(
            labels, logits, sequence_length_time, sum_over_timesteps=False,
            average_across_batch=False, time_major=True)
        self.assertEqual(loss.shape, torch.Size([self._batch_size,
                                                 self._max_time]))

    def test_sequence_softmax_cross_entropy(self):
        """Tests `sequence_softmax_cross_entropy`
        """
        self._test_sequence_loss(
            mle_losses.sequence_softmax_cross_entropy,
            self._one_hot_labels, self._logits, self._sequence_length)

    def test_sequence_sparse_softmax_cross_entropy(self):
        """Tests `sequence_sparse_softmax_cross_entropy`
        """
        self._test_sequence_loss(
            mle_losses.sequence_sparse_softmax_cross_entropy,
            self._labels, self._logits, self._sequence_length)

    def test_sequence_sigmoid_cross_entropy(self):
        """Tests `texar.torch.losses.sequence_sigmoid_cross_entropy`.
        """
        self._test_sequence_loss(
            mle_losses.sequence_sigmoid_cross_entropy,
            self._one_hot_labels, self._logits, self._sequence_length)

        self._test_sequence_loss(
            mle_losses.sequence_sigmoid_cross_entropy,
            self._one_hot_labels[:, :, 0],
            self._logits[:, :, 0],
            self._sequence_length)

        loss = mle_losses.sequence_sigmoid_cross_entropy(
            logits=self._logits[:, :, 0],
            labels=torch.ones([self._batch_size, self._max_time]),
            sequence_length=self._sequence_length)
        rank = get_rank(loss)
        self.assertEqual(rank, 0)


if __name__ == "__main__":
    unittest.main()
