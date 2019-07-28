"""
Unit tests for RL rewards.
"""

import unittest

import numpy as np
import torch

from texar.torch.losses import rewards


class RewardTest(unittest.TestCase):
    """Tests reward related functions.
    """

    def test_discount_reward(self):
        """Tests :func:`texar.torch.losses.rewards.discount_reward`
        """
        # 1D
        reward = torch.ones(2)
        sequence_length = torch.tensor([3, 5])

        discounted_reward = rewards.discount_reward(
            reward, sequence_length, discount=1.)
        discounted_reward_n = rewards.discount_reward(
            reward, sequence_length, discount=.1, normalize=True)

        discounted_reward_ = rewards.discount_reward(
            reward, sequence_length, discount=1.)
        discounted_reward_n_ = rewards.discount_reward(
            reward, sequence_length, discount=.1, normalize=True)

        np.testing.assert_array_almost_equal(
            discounted_reward, discounted_reward_, decimal=6)

        np.testing.assert_array_almost_equal(
            discounted_reward_n, discounted_reward_n_, decimal=6)

        # 2D
        reward = torch.ones(2, 10)
        sequence_length = torch.tensor([5, 10])

        discounted_reward = rewards.discount_reward(
            reward, sequence_length, discount=1.)
        discounted_reward_n = rewards.discount_reward(
            reward, sequence_length, discount=.1, normalize=True)

        discounted_reward_ = rewards.discount_reward(
            reward, sequence_length, discount=1.)
        discounted_reward_n_ = rewards.discount_reward(
            reward, sequence_length, discount=.1, normalize=True)

        np.testing.assert_array_almost_equal(
            discounted_reward, discounted_reward_, decimal=6)

        np.testing.assert_array_almost_equal(
            discounted_reward_n, discounted_reward_n_, decimal=6)

    def test_discount_reward_tensor_1d(self):
        """Tests :func:`texar.torch.losses.rewards._discount_reward_tensor_1d`
        """
        reward = torch.ones(2)
        sequence_length = torch.tensor([3, 5])

        discounted_reward_1 = rewards._discount_reward_tensor_1d(
            reward, sequence_length, discount=1.)

        discounted_reward_2 = rewards._discount_reward_tensor_1d(
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
        """Tests :func:`texar.torch.losses.rewards._discount_reward_tensor_2d`
        """
        reward = torch.ones(2, 10)
        sequence_length = torch.tensor([5, 10])

        discounted_reward_1 = rewards._discount_reward_tensor_2d(
            reward, sequence_length, discount=1.)

        discounted_reward_2 = rewards._discount_reward_tensor_2d(
            reward, sequence_length, discount=.1)

        for i in range(10):
            if i < 5:
                self.assertEqual(discounted_reward_1[0, i].item(), 5 - i)
            else:
                self.assertEqual(discounted_reward_1[0, i].item(), 0)
            self.assertEqual(discounted_reward_1[1, i].item(), 10 - i)

        for i in range(10):
            if i < 5:
                self.assertAlmostEqual(discounted_reward_2[0, i].item(),
                                       int(11111. / 10 ** i) / 10 ** (4 - i),
                                       places=6)
            else:
                self.assertAlmostEqual(discounted_reward_2[0, i].item(), 0)
            self.assertAlmostEqual(discounted_reward_2[1, i].item(),
                                   int(1111111111. / 10 ** i) / 10 ** (9 - i),
                                   places=6)


if __name__ == "__main__":
    unittest.main()
