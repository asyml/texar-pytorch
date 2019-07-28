"""
Unit tests for XLNet decoder.
"""
import unittest

import torch

from texar.torch.modules.decoders.xlnet_decoder import *
from texar.torch.utils.test import pretrained_test


class XLNetDecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.XLNetDecoder`
    """

    def setUp(self) -> None:
        self.batch_size = 2
        self.max_length = 3
        self.start_tokens = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long)

    @pretrained_test
    def test_hparams(self):
        r"""Tests the priority of the decoder arch parameters.
        """
        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
        }
        decoder = XLNetDecoder(pretrained_model_name="xlnet-base-cased",
                               hparams=hparams)
        self.assertEqual(decoder.hparams.num_layers, 12)

        _, _ = decoder(start_tokens=self.start_tokens,
                       end_token=1, max_decoding_length=self.max_length)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
            "num_layers": 6,
        }
        decoder = XLNetDecoder(hparams=hparams)
        self.assertEqual(decoder.hparams.num_layers, 24)

        _, _ = decoder(start_tokens=self.start_tokens,
                       end_token=1, max_decoding_length=self.max_length)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "num_layers": 6,
        }
        decoder = XLNetDecoder(hparams=hparams)
        self.assertEqual(decoder.hparams.num_layers, 6)

        _, _ = decoder(start_tokens=self.start_tokens,
                       end_token=1, max_decoding_length=self.max_length)

        # case 4: using default hparams
        decoder = XLNetDecoder()
        self.assertEqual(decoder.hparams.num_layers, 12)

        _, _ = decoder(start_tokens=self.start_tokens,
                       end_token=1, max_decoding_length=self.max_length)

    @pretrained_test
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        # case 1
        decoder = XLNetDecoder()
        self.assertEqual(len(decoder.trainable_variables), 182 + 1)

        _, _ = decoder(start_tokens=self.start_tokens,
                       end_token=1, max_decoding_length=self.max_length)

        # case 2
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
        }
        decoder = XLNetDecoder(hparams=hparams)
        self.assertEqual(len(decoder.trainable_variables), 362 + 1)

        _, _ = decoder(start_tokens=self.start_tokens,
                       end_token=1, max_decoding_length=self.max_length)

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "num_layers": 6
        }
        decoder = XLNetDecoder(hparams=hparams)
        self.assertEqual(len(decoder.trainable_variables), 92 + 1)

        _, _ = decoder(start_tokens=self.start_tokens,
                       end_token=1, max_decoding_length=self.max_length)

    @pretrained_test
    def test_decode_infer_sample(self):
        r"""Tests train_greedy."""
        hparams = {
            "pretrained_model_name": None,
        }
        decoder = XLNetDecoder(hparams=hparams)
        decoder.train()

        inputs = torch.randint(32000, (self.batch_size, self.max_length))
        outputs, _ = decoder(
            inputs, max_decoding_length=self.max_length, end_token=2)

        self.assertIsInstance(outputs, XLNetDecoderOutput)


if __name__ == "__main__":
    unittest.main()
