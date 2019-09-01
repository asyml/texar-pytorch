"""
Unit tests for DistilBERT utils.
"""

import os
import unittest

from texar.torch.modules.pretrained.pretrained_distilbert import *
from texar.torch.utils.test import pretrained_test


class DistilBERTUtilsTest(unittest.TestCase):
    r"""Tests DistilBERT utils.
    """

    @pretrained_test
    def test_load_pretrained_distilbert_AND_transform_to_texar_config(self):

        pretrained_model_dir = PretrainedDistilBERTMixin.download_checkpoint(
            pretrained_model_name="distilbert-base-uncased")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('distilbert-base-uncased-pytorch_model.bin', files)
        self.assertIn('distilbert-base-uncased-config.json', files)

        model_config = PretrainedDistilBERTMixin._transform_config(
            pretrained_model_name="distilbert-base-uncased",
            cache_dir=pretrained_model_dir)

        exp_config = {
            'hidden_size': 768,
            'embed': {
                'name': 'word_embeddings',
                'dim': 768
            },
            'vocab_size': 30522,
            'use_sinusoidal_pos_embed': True,
            'position_embed': {
                'name': 'position_embeddings',
                'dim': 768
            },
            'position_size': 512,
            'encoder': {
                'name': 'encoder',
                'embedding_dropout': 0.1,
                'num_blocks': 6,
                'multihead_attention': {
                    'use_bias': True,
                    'num_units': 768,
                    'num_heads': 12,
                    'output_dim': 768,
                    'dropout_rate': 0.1,
                    'name': 'self'
                },
                'residual_dropout': 0.1,
                'dim': 768,
                'use_bert_config': True,
                'poswise_feedforward': {
                    'layers': [
                        {
                            'type': 'Linear',
                            'kwargs': {
                                'in_features': 768,
                                'out_features': 3072,
                                'bias': True
                            }
                        },
                        {'type': 'BertGELU'},
                        {
                            'type': 'Linear',
                            'kwargs': {
                                'in_features': 3072,
                                'out_features': 768,
                                'bias': True
                            }
                        }
                    ]
                }
            }
        }

        self.assertDictEqual(model_config, exp_config)


if __name__ == "__main__":
    unittest.main()
