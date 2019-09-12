"""
Unit tests for BioBERT utils.
"""

import os
import unittest

from texar.torch.modules.pretrained.biobert import *
from texar.torch.utils.test import pretrained_test


class BioBERTUtilsTest(unittest.TestCase):
    r"""Tests BioBERT utils.
    """

    @pretrained_test
    def test_load_biobert_AND_transform_biobert_to_texar_config(self):

        pretrained_model_dir = PretrainedBioBERTMixin.download_checkpoint(
            pretrained_model_name="biobert-v1.0-pmc")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('biobert_model.ckpt.meta', files)
        self.assertIn('biobert_model.ckpt.data-00000-of-00001', files)
        self.assertIn('biobert_model.ckpt.index', files)
        self.assertIn('bert_config.json', files)

        model_config = PretrainedBioBERTMixin._transform_config(
            pretrained_model_name="biobert-v1.0-pmc",
            cache_dir=pretrained_model_dir)

        exp_config = {
            'hidden_size': 768,
            'embed': {
                'name': 'word_embeddings',
                'dim': 768
            },
            'vocab_size': 28996,
            'segment_embed': {
                'name': 'token_type_embeddings',
                'dim': 768
            },
            'type_vocab_size': 2,
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
