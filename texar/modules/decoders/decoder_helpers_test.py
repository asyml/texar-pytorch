"""
Unit tests for Transformer decoder.
"""
import unittest

import torch

from texar.modules.decoders.decoder_helpers import GreedyEmbeddingHelper, \
    TopKSampleEmbeddingHelper, TopPSampleEmbeddingHelper


class TransformerDecoderTest(unittest.TestCase):
    r"""Tests decoder helper utilities.
    """

    def setUp(self):
        self.logits = torch.Tensor([[0.2, 0.2, 0.55, 0.05]])
        # softmax values for above tensor is
        # tensor([[0.2337, 0.2337, 0.3316, 0.2011]])
        self.emb = torch.rand(10, 5)
        self.start_token = torch.LongTensor([1])
        self.end_token = 2

    def test_greedy_sampler(self):
        """Tests Greedy Sampler."""
        sampler = GreedyEmbeddingHelper(embedding=self.emb,
                                        start_tokens=self.start_token,
                                        end_token=self.end_token)
        index = sampler.sample(time=0, outputs=self.logits)
        assert torch.equal(index, torch.argmax(self.logits, dim=1))

    def test_top_k_sampler(self):
        """Tests Top-K Sampler."""
        sampler = TopKSampleEmbeddingHelper(embedding=self.emb,
                                            start_tokens=self.start_token,
                                            end_token=self.end_token, top_k=1)
        index = sampler.sample(time=0, outputs=self.logits)
        assert torch.equal(index, torch.argmax(self.logits, dim=1))

    def test_top_p_sampler(self):
        """Tests Top-P Sampler also known as Nucleus Sampler."""
        sampler = TopPSampleEmbeddingHelper(embedding=self.emb,
                                            start_tokens=self.start_token,
                                            end_token=self.end_token, p=0.6)
        index = sampler.sample(time=0, outputs=self.logits)
        assert index.item() in [0, 1, 2]


if __name__ == "__main__":
    unittest.main()
