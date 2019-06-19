"""
Unit tests for RL rewards.
"""

import unittest

import numpy as np
import torch

import texar as tx


class RewardTest(unittest.TestCase):
    """Tests reward related functions.
    """

    def test_discount_reward(self):
        """Tests :func:`texar.losses.rewards.discount_reward`
        """
        # 1D
        reward = np.ones([2], dtype=np.float64)
        sequence_length = [3, 5]

        discounted_reward = tx.losses.discount_reward(
            reward, sequence_length, discount=1.)
        discounted_reward_n = tx.losses.discount_reward(
            reward, sequence_length, discount=.1, normalize=True)

        discounted_reward_ = tx.losses.discount_reward(
            torch.tensor(reward, dtype=torch.float64),
            sequence_length, discount=1.)
        discounted_reward_n_ = tx.losses.discount_reward(
            torch.tensor(reward, dtype=torch.float64),
            sequence_length, discount=.1, normalize=True)

        np.testing.assert_array_almost_equal(
            discounted_reward, discounted_reward_, decimal=6)

        np.testing.assert_array_almost_equal(
            discounted_reward_n, discounted_reward_n_, decimal=6)

        # 2D
        reward = np.ones([2, 10], dtype=np.float64)
        sequence_length = [5, 10]

        discounted_reward = tx.losses.discount_reward(
            reward, sequence_length, discount=1.)
        discounted_reward_n = tx.losses.discount_reward(
            reward, sequence_length, discount=.1, normalize=True)

        discounted_reward_ = tx.losses.discount_reward(
            torch.tensor(reward, dtype=torch.float64), sequence_length,
            discount=1.)
        discounted_reward_n_ = tx.losses.discount_reward(
            torch.tensor(reward, dtype=torch.float64), sequence_length,
            discount=.1, normalize=True)

        np.testing.assert_array_almost_equal(
            discounted_reward, discounted_reward_, decimal=6)

        np.testing.assert_array_almost_equal(
            discounted_reward_n, discounted_reward_n_, decimal=6)

    def test_discount_reward_tensor_1d(self):
        """Tests :func:`texar.losses.rewards._discount_reward_tensor_1d`
        """
        reward = torch.ones([2], dtype=torch.float64)
        sequence_length = [3, 5]

        discounted_reward_1 = tx.losses._discount_reward_tensor_1d(
            reward, sequence_length, discount=1.)

        discounted_reward_2 = tx.losses._discount_reward_tensor_1d(
            reward, sequence_length, discount=.1)

        for i in range(5):
            if i < 3:
                self.assertEqual(discounted_reward_1[0, i], 1)
            else:
                self.assertEqual(discounted_reward_1[0, i], 0)
            self.assertEqual(discounted_reward_1[1, i], 1)

        for i in range(5):
            if i < 3:
                self.assertAlmostEqual(discounted_reward_2[0, i].item(),
                                       0.1 ** (2 - i))
            else:
                self.assertAlmostEqual(discounted_reward_2[0, i].item(), 0)
            self.assertAlmostEqual(discounted_reward_2[1, i].item(),
                                   0.1 ** (4 - i))

    def test_discount_reward_tensor_2d(self):
        """Tests :func:`texar.losses.rewards._discount_reward_tensor_2d`
        """
        reward = torch.ones([2, 10], dtype=torch.float64)
        sequence_length = [5, 10]

        discounted_reward_1 = tx.losses._discount_reward_tensor_2d(
            reward, sequence_length, discount=1.)

        discounted_reward_2 = tx.losses._discount_reward_tensor_2d(
            reward, sequence_length, discount=.1)

        for i in range(10):
            if i < 5:
                self.assertEqual(discounted_reward_1[0, i], 5 - i)
            else:
                self.assertEqual(discounted_reward_1[0, i], 0)
            self.assertEqual(discounted_reward_1[1, i], 10 - i)

        for i in range(10):
            if i < 5:
                self.assertEqual(discounted_reward_2[0, i],
                                 int(11111. / 10 ** i) / 10 ** (4 - i))
            else:
                self.assertEqual(discounted_reward_2[0, i], 0)
            self.assertEqual(discounted_reward_2[1, i],
                             int(1111111111. / 10 ** i) / 10 ** (9 - i))


if __name__ == "__main__":
    unittest.main()
