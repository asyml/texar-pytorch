"""
Unit tests for entropy.
"""

import unittest

import torch

from texar.torch.losses import entropy
from texar.torch.utils.shapes import get_rank


class EntropyTest(unittest.TestCase):
    """Tests entropy.
    """

    def setUp(self):
        self._batch_size = 64
        self._max_time = 128
        self._d = 16
        self._distribution_dim = 32
        self._logits = torch.rand(self._batch_size, self._d,
                                  self._distribution_dim)
        self._sequence_logits = torch.rand(
            self._batch_size, self._max_time, self._d,
            self._distribution_dim)
        self._sequence_length = torch.randint(
            high=self._max_time, size=(self._batch_size,))

    def _test_entropy(self, entropy_fn, logits, sequence_length=None):
        if sequence_length is None:
            entropy = entropy_fn(logits)
            rank = get_rank(entropy)
            self.assertEqual(rank, 0)

            entropy = entropy_fn(logits, average_across_batch=False)
            rank = get_rank(entropy)
            self.assertEqual(rank, 1)
            self.assertEqual(entropy.shape, torch.Size([self._batch_size]))
        else:
            entropy = entropy_fn(logits, sequence_length=sequence_length)
            rank = get_rank(entropy)
            self.assertEqual(rank, 0)

            entropy = entropy_fn(logits, sequence_length=sequence_length,
                                 sum_over_timesteps=False)
            rank = get_rank(entropy)
            self.assertEqual(rank, 1)
            self.assertEqual(entropy.shape, torch.Size([self._max_time]))

            entropy = entropy_fn(logits, sequence_length=sequence_length,
                                 sum_over_timesteps=False,
                                 average_across_timesteps=True,
                                 average_across_batch=False)
            rank = get_rank(entropy)
            self.assertEqual(rank, 1)
            self.assertEqual(entropy.shape, torch.Size([self._batch_size]))

            entropy = entropy_fn(logits, sequence_length=sequence_length,
                                 sum_over_timesteps=False,
                                 average_across_batch=False)
            rank = get_rank(entropy)
            self.assertEqual(rank, 2)
            self.assertEqual(entropy.shape, torch.Size([self._batch_size,
                                                        self._max_time]))

            sequence_length_time = torch.randint(size=(self._max_time,),
                                                 high=self._batch_size)
            entropy = entropy_fn(logits,
                                 sequence_length=sequence_length_time,
                                 sum_over_timesteps=False,
                                 average_across_batch=False,
                                 time_major=True)
            self.assertEqual(entropy.shape, torch.Size([self._batch_size,
                                                        self._max_time]))

    def test_entropy_with_logits(self):
        """Tests `entropy_with_logits`
        """
        self._test_entropy(
            entropy.entropy_with_logits, self._logits)

    def test_sequence_entropy_with_logits(self):
        """Tests `sequence_entropy_with_logits`
        """
        self._test_entropy(
            entropy.sequence_entropy_with_logits, self._sequence_logits,
            sequence_length=self._sequence_length)


if __name__ == "__main__":
    unittest.main()
