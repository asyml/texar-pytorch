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
        inputs = torch.zeros(32, 16, dtype=torch.int64)

        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": None,
        }
        decoder = XLNetDecoder(pretrained_model_name="xlnet-large-cased",
                               hparams=hparams)

        import pdb
        pdb.set_trace()

        _, _ = decoder(start_tokens=inputs,
                       end_token=10)
        self.assertEqual(decoder.hparams.num_layers, 24)


if __name__ == "__main__":
    unittest.main()
