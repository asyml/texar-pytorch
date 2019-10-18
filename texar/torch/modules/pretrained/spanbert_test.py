"""
Unit tests for SpanBERT utils.
"""

import os
import unittest

from texar.torch.modules.pretrained.spanbert import *
from texar.torch.utils.test import pretrained_test


class SpanBERTUtilsTest(unittest.TestCase):
    r"""Tests SpanBERT utils.
    """

    @pretrained_test
    def test_load_spanbert_AND_transform_spanbert_to_texar_config(
            self):
        pretrained_model_dir = PretrainedSpanBERTMixin.download_checkpoint(
            pretrained_model_name="spanbert-base-cased")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('config.json', files)
        self.assertIn('pytorch_model.bin', files)

        model_config = PretrainedSpanBERTMixin._transform_config(
            pretrained_model_name="spanbert-base",
            cache_dir=pretrained_model_dir)

        exp_config = {
            'hidden_size': 768,
            'embed': {
                'name': 'word_embeddings',
                'dim': 768
            },
            'vocab_size': 28996,
            'position_embed': {
                'name': 'position_embeddings',
                'dim': 768
            },
            'position_size': 512,
            'encoder': {
                'name': 'encoder',
                'embedding_dropout': 0.1,
                'num_blocks': 12,
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
