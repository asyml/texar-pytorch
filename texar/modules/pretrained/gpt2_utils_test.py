"""
Unit tests for GPT2 utils.
"""

import os
import unittest

from texar.modules.pretrained.gpt2_utils import *


class GPTUtilsTest(unittest.TestCase):
    r"""Tests GPT2 utils.
    """

    def test_load_pretrained_model_AND_transform_gpt2_to_texar_config(self):

        pretrained_model_dir = load_pretrained_model(
            pretrained_model_name="117M")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('checkpoint', files)
        self.assertIn('encoder.json', files)
        self.assertIn('hparams.json', files)
        self.assertIn('model.ckpt.data-00000-of-00001', files)
        self.assertIn('model.ckpt.index', files)
        self.assertIn('model.ckpt.meta', files)
        self.assertIn('vocab.bpe', files)

        model_config = transform_gpt2_to_texar_config(pretrained_model_dir)

        exp_config = {
            'vocab_size': 50257,
            'context_size': 1024,
            'embedding_size': 768,
            'embed': {
                'dim': 768
            },
            'position_size': 1024,
            'position_embed': {
                'dim': 768
            },
            'decoder': {
                'dim': 768,
                'num_blocks': 12,
                'use_gpt_config': True,
                'embedding_dropout': 0,
                'residual_dropout': 0,
                'multihead_attention': {
                    'use_bias': True,
                    'num_units': 768,
                    'num_heads': 12,
                    'output_dim': 768
                },
                'initializer': {
                    'type': 'variance_scaling_initializer',
                    'kwargs': {
                        'factor': 1.0,
                        'mode': 'FAN_AVG',
                        'uniform': True
                    }
                },
                'poswise_feedforward': {
                    'layers':
                        [
                            {
                                'type': 'Linear',
                                'kwargs': {
                                    'in_features': 768,
                                    'out_features': 3072,
                                    'bias': True
                                }
                            },
                            {
                                'type': 'GPTGELU',
                                'kwargs': {}
                            },
                            {
                                'type': 'Linear',
                                'kwargs': {
                                    'in_features': 3072,
                                    'out_features': 768,
                                    'bias': True
                                }
                            }
                        ],
                    'name': 'ffn'
                }
            }
        }

        self.assertDictEqual(model_config, exp_config)


if __name__ == "__main__":
    unittest.main()
