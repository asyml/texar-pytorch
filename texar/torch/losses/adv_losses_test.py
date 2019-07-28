"""
Unit tests for adv_losses.
"""

import unittest

import torch

from texar.torch.losses.adv_losses import binary_adversarial_losses


class AdvLossesTest(unittest.TestCase):
    """Tests adversarial losses.
    """

    def test_binary_adversarial_losses(self):
        """Tests :meth:`~texar.torch.losses.adv_losses.binary_adversarial_losse`.
        """
        batch_size = 16
        data_dim = 64
        real_data = torch.zeros(size=(batch_size, data_dim),
                                dtype=torch.float32)
        fake_data = torch.ones(size=(batch_size, data_dim),
                               dtype=torch.float32)
        const_logits = torch.zeros(size=(batch_size,), dtype=torch.float32)
        # Use a dumb discriminator that always outputs logits=0.
        gen_loss, disc_loss = binary_adversarial_losses(
            real_data, fake_data, lambda x: const_logits)
        gen_loss_2, disc_loss_2 = binary_adversarial_losses(
            real_data, fake_data, lambda x: const_logits, mode="min_fake")

        self.assertAlmostEqual(gen_loss.item(), -gen_loss_2.item())
        self.assertAlmostEqual(disc_loss.item(), disc_loss_2.item())


if __name__ == "__main__":
    unittest.main()
