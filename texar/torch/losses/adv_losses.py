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
Adversarial losses.
"""

from typing import Callable, Tuple

import torch
import torch.nn.functional as F

from texar.torch.utils.types import MaybeTuple

__all__ = [
    'binary_adversarial_losses',
]


def binary_adversarial_losses(
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        discriminator_fn: Callable[[torch.Tensor], MaybeTuple[torch.Tensor]],
        mode: str = "max_real") -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes adversarial losses of real/fake binary discrimination game.

    Example:

    .. code-block:: python

        # Using BERTClassifier as the discriminator, which can accept
        # "soft" token ids for gradient backpropagation
        discriminator = tx.modules.BERTClassifier('bert-base-uncased')

        G_loss, D_loss = tx.losses.binary_adversarial_losses(
            real_data=real_token_ids,  # [batch_size, max_time]
            fake_data=fake_soft_token_ids,  # [batch_size, max_time, vocab_size]
            discriminator_fn=discriminator)

    Args:
        real_data (Tensor or array): Real data of shape
            `[num_real_examples, ...]`.
        fake_data (Tensor or array): Fake data of shape
            `[num_fake_examples, ...]`. `num_real_examples` does not
            necessarily equal `num_fake_examples`.
        discriminator_fn: A callable takes data (e.g., :attr:`real_data` and
            :attr:`fake_data`) and returns the logits of being real. The
            signature of `discriminator_fn` must be:
            :python:`logits, ... = discriminator_fn(data)`.
            The return value of `discriminator_fn` can be the logits, or
            a tuple where the logits are the first element.
        mode (str): Mode of the generator loss. Either "max_real" or "min_fake".

            - **"max_real"** (default): minimizing the generator loss is to
              maximize the probability of fake data being classified as real.

            - **"min_fake"**: minimizing the generator loss is to minimize the
              probability of fake data being classified as fake.

    Returns:
        A tuple `(generator_loss, discriminator_loss)` each of which is
        a scalar Tensor, loss to be minimized.
    """
    real_logits = discriminator_fn(real_data)
    if isinstance(real_logits, (list, tuple)):
        real_logits = real_logits[0]
    real_loss = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits))

    fake_logits = discriminator_fn(fake_data)
    if isinstance(fake_logits, (list, tuple)):
        fake_logits = fake_logits[0]
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits))

    d_loss = real_loss + fake_loss

    if mode == "min_fake":
        g_loss = -fake_loss
    elif mode == "max_real":
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        g_loss = bce_loss(fake_logits, torch.ones_like(fake_logits))
    else:
        raise ValueError("Unknown mode: %s. Only 'min_fake' and 'max_real' "
                         "are allowed.")

    return g_loss, d_loss
