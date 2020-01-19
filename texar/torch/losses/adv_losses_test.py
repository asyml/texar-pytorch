# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
