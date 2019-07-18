"""
Unit tests for XLNet decoder.
"""
import unittest

import torch

from texar.modules.decoders.xlnet_decoder import *


class XLNetDecoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.modules.XLNetDecoder`
    """

    def test_hparams(self):
        r"""Tests the priority of the decoer arch parameter.
        """
        inputs = torch.zeros(1, 1, dtype=torch.int64)

        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
        }
        decoder = XLNetDecoder(pretrained_model_name="xlnet-base-cased",
                               hparams=hparams)

        _, _ = decoder(start_tokens=inputs,
                       end_token=1)
        self.assertEqual(decoder.hparams.num_layers, 12)


if __name__ == "__main__":
    unittest.main()
