"""
Unit tests for pg losses.
"""

import unittest

import torch

from texar.torch.losses.pg_losses import pg_loss_with_logits
from texar.torch.utils.shapes import get_rank


class PGLossesTest(unittest.TestCase):
    """Tests pg losses
    """

    def setUp(self):
        self._batch_size = 64
        self._max_time = 16
        self._d1 = 3  # use smaller values to speedup testing
        self._d2 = 4
        self._d3 = 5
        self._num_classes = 10
        self._actions_batch = torch.ones(
            self._batch_size, self._max_time, self._d1, self._d2, self._d3,
            dtype=torch.int64)
        self._logits_batch = torch.rand(
            self._batch_size, self._max_time, self._d1, self._d2, self._d3,
            self._num_classes)
        self._advantages_batch = torch.rand(
            self._batch_size, self._max_time, self._d1, self._d2, self._d3)
        self._actions_no_batch = torch.ones(
            self._max_time, self._d1, self._d2, self._d3, dtype=torch.int64)
        self._logits_no_batch = torch.rand(
            self._max_time, self._d1, self._d2, self._d3, self._num_classes)
        self._advantages_no_batch = torch.rand(
            self._max_time, self._d1, self._d2, self._d3)
        self._sequence_length = torch.randint(
            high=self._max_time, size=(self._batch_size,))

    def _test_sequence_loss(self, loss_fn, actions, logits, advantages, batched,
                            sequence_length):
        loss = loss_fn(actions, logits, advantages, batched=batched,
                       sequence_length=sequence_length)
        rank = get_rank(loss)
        self.assertEqual(rank, 0)

        loss = loss_fn(actions, logits, advantages, batched=batched,
                       sequence_length=sequence_length,
                       sum_over_timesteps=False)
        rank = get_rank(loss)
        self.assertEqual(rank, 1)
        self.assertEqual(loss.shape, torch.Size([self._max_time]))

        loss = loss_fn(actions, logits, advantages, batched=batched,
                       sequence_length=sequence_length,
                       sum_over_timesteps=False,
                       average_across_timesteps=True,
                       average_across_batch=False)
        rank = get_rank(loss)
        if batched:
            self.assertEqual(rank, 1)
            self.assertEqual(loss.shape, torch.Size([self._batch_size]))
        else:
            self.assertEqual(rank, 0)

        loss = loss_fn(actions, logits, advantages, batched=batched,
                       sequence_length=sequence_length,
                       sum_over_timesteps=False,
                       average_across_batch=False)
        rank = get_rank(loss)
        if batched:
            self.assertEqual(rank, 2)
            self.assertEqual(loss.shape,
                             torch.Size([self._batch_size, self._max_time]))
        else:
            self.assertEqual(rank, 1)
            self.assertEqual(loss.shape,
                             torch.Size([self._max_time]))

        sequence_length_time = torch.randint(
            high=self._batch_size, size=(self._max_time,))
        loss = loss_fn(actions, logits, advantages, batched=batched,
                       sequence_length=sequence_length_time,
                       sum_over_timesteps=False,
                       average_across_batch=False,
                       time_major=True)
        if batched:
            self.assertEqual(loss.shape, torch.Size([self._batch_size,
                                                     self._max_time]))
        else:
            self.assertEqual(loss.shape, torch.Size([self._max_time]))

    def test_pg_loss_with_logits(self):
        """Tests `texar.torch.losses.pg_loss_with_logits`.
        """
        self._test_sequence_loss(
            pg_loss_with_logits,
            self._actions_batch, self._logits_batch,
            self._advantages_batch, True, self._sequence_length)

        self._test_sequence_loss(
            pg_loss_with_logits,
            self._actions_no_batch, self._logits_no_batch,
            self._advantages_no_batch, False, self._sequence_length)


if __name__ == "__main__":
    unittest.main()
