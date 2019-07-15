"""
Unit tests for xlnet utils.
"""

import os
import unittest

import torch

from texar.modules.pretrained.xlnet_utils import *


class XLNetUtilsTest(unittest.TestCase):
    r"""Tests xlnet utils.
    """

    def test_load_pretrained_model_AND_transform_xlnet_to_texar_config(self):

        pretrained_model_dir = load_pretrained_model(
            pretrained_model_name="xlnet-base-cased")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('spiece.model', files)
        self.assertIn('xlnet_model.ckpt.meta', files)
        self.assertIn('xlnet_model.ckpt.data-00000-of-00001', files)
        self.assertIn('xlnet_model.ckpt.index', files)
        self.assertIn('xlnet_config.json', files)

        model_config = transform_xlnet_to_texar_config(pretrained_model_dir)

        exp_config = {'head_dim': 64,
                      'ffn_inner_dim': 3072,
                      'hidden_dim': 768,
                      'activation': 'gelu',
                      'num_heads': 12,
                      'num_layers': 12,
                      'vocab_size': 32000,
                      'untie_r': True}

        self.assertDictEqual(model_config, exp_config)

    def test_sum_tensors(self):

        inputs = [torch.tensor(1), torch.tensor(2)]
        self.assertEqual(sum_tensors(inputs), torch.tensor(3))

        inputs = [torch.tensor(1), None, torch.tensor(2)]
        self.assertEqual(sum_tensors(inputs), torch.tensor(3))

        inputs = [torch.tensor(1), None, None]
        self.assertEqual(sum_tensors(inputs), torch.tensor(1))

        inputs = [None, None, None]
        self.assertEqual(sum_tensors(inputs), None)


if __name__ == "__main__":
    unittest.main()
