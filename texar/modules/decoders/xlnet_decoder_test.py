"""
Unit tests for XLNet decoder.
"""
import unittest

import torch

from texar.modules.decoders.xlnet_decoder import *


@unittest.skip("Manual test only")
class XLNetDecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.XLNetDecoder`
    """

    def test_hparams(self):
        r"""Tests the priority of the decoer arch parameter.
        """
        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
        }
        decoder = XLNetDecoder(pretrained_model_name="xlnet-base-cased",
                               hparams=hparams)

        _, _ = decoder(start_tokens=torch.zeros(16, 8, dtype=torch.int64),
                       end_token=1,
                       max_decoding_length=8)

        self.assertEqual(decoder.hparams.num_layers, 12)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
            "num_layers": 6
        }
        decoder = XLNetDecoder(hparams=hparams)

        _, _ = decoder(start_tokens=torch.zeros(16, 8, dtype=torch.int64),
                       end_token=1,
                       max_decoding_length=8)

        self.assertEqual(decoder.hparams.num_layers, 24)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "num_layers": 6
        }
        decoder = XLNetDecoder(hparams=hparams)

        _, _ = decoder(start_tokens=torch.zeros(16, 8, dtype=torch.int64),
                       end_token=1,
                       max_decoding_length=8)
        self.assertEqual(decoder.hparams.num_layers, 6)

        # case 4: using default hparams
        decoder = XLNetDecoder()

        _, _ = decoder(start_tokens=torch.zeros(16, 8, dtype=torch.int64),
                       end_token=1,
                       max_decoding_length=8)
        self.assertEqual(decoder.hparams.num_layers, 12)

    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """
        # case 1
        decoder = XLNetDecoder()

        _, _ = decoder(start_tokens=torch.zeros(16, 8, dtype=torch.int64),
                       end_token=1,
                       max_decoding_length=8)
        self.assertEqual(len(decoder.trainable_variables), 182 + 1)

        # case 2
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
        }
        decoder = XLNetDecoder(hparams=hparams)

        _, _ = decoder(start_tokens=torch.zeros(16, 8, dtype=torch.int64),
                       end_token=1,
                       max_decoding_length=8)
        self.assertEqual(len(decoder.trainable_variables), 362 + 1)

        # case 3
        hparams = {
            "pretrained_model_name": None,
            "num_layers": 6
        }
        decoder = XLNetDecoder(hparams=hparams)

        _, _ = decoder(start_tokens=torch.zeros(16, 8, dtype=torch.int64),
                       end_token=1,
                       max_decoding_length=8)
        self.assertEqual(len(decoder.trainable_variables), 92 + 1)


if __name__ == "__main__":
    unittest.main()
