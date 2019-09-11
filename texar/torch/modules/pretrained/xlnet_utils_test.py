"""
Unit tests for XLNet model utils.
"""

import unittest

import torch

from texar.torch.modules.pretrained.xlnet_utils import *


class XLNetModelUtilsTest(unittest.TestCase):
    r"""Tests XLNet model utils.
    """

    def test_PositionWiseFF(self):

        # Case 1
        model = PositionWiseFF()
        inputs = torch.rand(32, model._hparams.hidden_dim)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, torch.Size([32,
                                                    model._hparams.hidden_dim]))

        # Case 2
        hparams = {
            "hidden_dim": 16,
            "ffn_inner_dim": 32,
            "dropout": 0.1,
            "activation": 'relu',
        }
        model = PositionWiseFF(hparams=hparams)
        inputs = torch.rand(32, 16)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, torch.Size([32, 16]))

        # Case 3
        hparams = {
            "hidden_dim": 16,
            "ffn_inner_dim": 32,
            "dropout": 0.1,
            "activation": 'gelu',
        }
        model = PositionWiseFF(hparams=hparams)
        inputs = torch.rand(32, 16)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, torch.Size([32, 16]))

    def test_RelativeMultiheadAttention(self):

        model = RelativeMultiheadAttention()

        states_h = torch.rand(16, 32, model._hparams.hidden_dim)
        pos_embed = torch.rand(24, 32, model._hparams.hidden_dim)

        output_h, output_g = model(states_h=states_h, pos_embed=pos_embed)

        self.assertEqual(output_h.shape,
                         torch.Size([16, 32, model._hparams.hidden_dim]))
        self.assertEqual(output_g, None)

    def test_RelativePositionalEncoding(self):

        batch_size = 16
        seq_len = 8
        total_len = 32

        # Case 1
        model = RelativePositionalEncoding()
        pos_embed = model(batch_size=batch_size,
                          seq_len=seq_len,
                          total_len=total_len)
        self.assertEqual(pos_embed.shape,
                         torch.Size([40, 16, model._hparams.dim]))

        # Case 2
        model = RelativePositionalEncoding()
        pos_embed = model(batch_size=batch_size,
                          seq_len=seq_len,
                          total_len=total_len,
                          attn_type='uni')
        self.assertEqual(pos_embed.shape,
                         torch.Size([33, 16, model._hparams.dim]))


if __name__ == "__main__":
    unittest.main()
