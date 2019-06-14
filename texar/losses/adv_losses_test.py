"""
Unit tests for adv_losses.
"""

import unittest

import torch

import texar as tx


class AdvLossesTest(unittest.TestCase):
    """Tests adversarial losses.
    """

    def test_binary_adversarial_losses(self):
        """Tests :meth:`~texar.losses.adv_losses.binary_adversarial_losse`.
        """
        batch_size = 16
        data_dim = 64
        real_data = torch.zeros(size=(batch_size, data_dim),
                                dtype=torch.float32)
        fake_data = torch.ones(size=(batch_size, data_dim),
                               dtype=torch.float32)
        const_logits = torch.zeros(size=(batch_size,), dtype=torch.float32)
        # Use a dumb discriminator that always outputs logits=0.
        gen_loss, disc_loss = tx.losses.binary_adversarial_losses(
            real_data, fake_data, lambda x: const_logits)
        gen_loss_2, disc_loss_2 = tx.losses.binary_adversarial_losses(
            real_data, fake_data, lambda x: const_logits, mode="min_fake")

        self.assertAlmostEqual(gen_loss.item(), -gen_loss_2.item())
        self.assertAlmostEqual(disc_loss.item(), disc_loss_2.item())


if __name__ == "__main__":
    unittest.main()
