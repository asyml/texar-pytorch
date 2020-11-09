# Copyright 2020 The Texar Authors. All Rights Reserved.
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
Unit tests for informational theory losses.
"""

import unittest

import torch
from torch.nn.functional import log_softmax

from texar.torch.losses import info_loss
from texar.torch.utils.shapes import get_rank


class InfoLossTest(unittest.TestCase):
    """Tests KL divergence loss.
    """
    def setUp(self):
        self._batch_size = 64
        self._d = 16
        self._distribution_dim = 4

    def test_kl_loss(self):
        input_logits = torch.rand(self._batch_size, self._d,
                                  self._distribution_dim)
        target_logits = torch.rand(self._batch_size, self._d,
                                  self._distribution_dim)

        kld = info_loss.kl_divg_loss_with_logits(
            target_logits,
            input_logits,
            softmax_temperature=0.4,
            confidence_threshold=0.3,
            reduction="mean"
        )

        rank = get_rank(kld)
        self.assertEqual(rank, 0)

        kld = info_loss.kl_divg_loss_with_logits(
            target_logits,
            input_logits,
            softmax_temperature=0.4,
            confidence_threshold=0.3,
            reduction="sum"
        )

        rank = get_rank(kld)
        self.assertEqual(rank, 0)

        kld = info_loss.kl_divg_loss_with_logits(
            target_logits,
            input_logits,
            softmax_temperature=0.4,
            confidence_threshold=0.3,
            reduction="none"
        )

        rank = get_rank(kld)
        self.assertEqual(rank, get_rank(input_logits))

        kld = info_loss.kl_divg_loss_with_logits(
            target_logits,
            input_logits,
            softmax_temperature=0.4,
            confidence_threshold=0.3,
            reduction="batchmean"
        )

        rank = get_rank(kld)
        self.assertEqual(rank, 0)

        kld = info_loss.kl_divg_loss_with_logits(
            target_logits,
            target_logits,
            softmax_temperature=1.0,
            confidence_threshold=-1,
            reduction="mean"
        )

        self.assertLess(kld, 1e-5)


if __name__ == "__main__":
    unittest.main()