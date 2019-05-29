# -*- coding: utf-8 -*-
#
"""
Unit tests for RL rewards.
"""

# pylint: disable=invalid-name

import unittest

import torch
import numpy as np

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




if __name__ == "__main__":
    unittest.main()