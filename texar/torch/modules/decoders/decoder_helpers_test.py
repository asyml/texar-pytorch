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
Unit tests for Transformer decoder.
"""
import unittest

import torch

from texar.torch.modules.decoders.decoder_helpers import (
    GreedyEmbeddingHelper, TopKSampleEmbeddingHelper, TopPSampleEmbeddingHelper)


class SamplerTest(unittest.TestCase):
    r"""Tests decoder helper utilities.
    """

    def setUp(self):
        self.logits = torch.Tensor([[0.2, 0.2, 0.55, 0.05]])
        # softmax values for above tensor is
        # tensor([[0.2337, 0.2337, 0.3316, 0.2011]])
        self.start_token = torch.LongTensor([1])
        self.end_token = 2

    def test_greedy_sampler(self):
        """Tests Greedy Sampler."""
        sampler = GreedyEmbeddingHelper(start_tokens=self.start_token,
                                        end_token=self.end_token)
        index = sampler.sample(time=0, outputs=self.logits)
        assert torch.equal(index, torch.argmax(self.logits, dim=1))

    def test_top_k_sampler(self):
        """Tests Top-K Sampler."""
        sampler = TopKSampleEmbeddingHelper(start_tokens=self.start_token,
                                            end_token=self.end_token, top_k=1)
        index = sampler.sample(time=0, outputs=self.logits)
        assert torch.equal(index, torch.argmax(self.logits, dim=1))

    def test_top_p_sampler(self):
        """Tests Top-P Sampler also known as Nucleus Sampler."""
        sampler = TopPSampleEmbeddingHelper(start_tokens=self.start_token,
                                            end_token=self.end_token, p=0.6)
        index = sampler.sample(time=0, outputs=self.logits)
        assert index.item() in [0, 1, 2]


if __name__ == "__main__":
    unittest.main()
