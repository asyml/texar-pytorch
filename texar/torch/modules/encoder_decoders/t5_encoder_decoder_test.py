"""
Unit tests for T5 encoder-decoder
"""

import unittest

import torch

from texar.torch.modules.encoder_decoders import T5EncoderDecoder
from texar.torch.utils.test import pretrained_test


class T5EncodoerDecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.T5EncoderDecoder` class.

    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    def test_model_loading(self):
        r"""Tests model loading functionality."""
        for pretrained_model_name in T5EncoderDecoder.available_checkpoints():
            encoder = T5EncoderDecoder(pretrained_model_name=pretrained_model_name)
            import pdb;pdb.set_trace()
            #_, _ = encoder(self.inputs)


if __name__ == "__main__":
    unittest.main()