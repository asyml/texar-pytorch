# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for T5 utils.
"""

import os
import unittest

from texar.torch.modules.pretrained.t5 import *
from texar.torch.utils.test import pretrained_test


class T5UtilsTest(unittest.TestCase):
    r"""Tests T5 utils.
    """

    @pretrained_test
    def test_load_pretrained_t5_AND_transform_t5_to_texar_config(self):

        pretrained_model_dir = PretrainedT5Mixin.download_checkpoint(
            pretrained_model_name="T5-Base")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('checkpoint', files)
        self.assertIn('operative_config.gin', files)

        # Check vocab file
        self.assertIn('sentencepiece.model', files)

        model_config = PretrainedT5Mixin._transform_config(
            pretrained_model_name="T5-Base",
            cache_dir=pretrained_model_dir)

        exp_config = {
            'hidden_size': 768,
            'embed': {
                'name': 'word_embeddings',
                'dim': 768
            },
            'vocab_size': 32128,

            'encoder': {
                'name': 'encoder',
                'embedding_dropout': 0.1,
                'num_blocks': 12,
                'multihead_attention': {
                    'use_bias': False,
                    'num_units': 768,
                    'num_heads': 12,
                    'output_dim': 768,
                    'dropout_rate': 0.1,
                    'name': 'self',
                    'is_decoder': False,
                    'relative_attention_num_buckets': 32
                },
                'residual_dropout': 0.1,
                'dim': 768,
                'eps': 1e-6,
                'poswise_feedforward': {
                    'layers': [
                        {
                            'type': 'Linear',
                            'kwargs': {
                                'in_features': 768,
                                'out_features': 3072,
                                'bias': False
                            }
                        },
                        {'type': 'ReLU'},
                        {
                            'type': 'Linear',
                            'kwargs': {
                                'in_features': 3072,
                                'out_features': 768,
                                'bias': False
                            }
                        }
                    ]
                }
            },
            'decoder': {
                'name': 'decoder',
                'embedding_dropout': 0.1,
                'num_blocks': 12,
                'multihead_attention': {
                    'use_bias': False,
                    'num_units': 768,
                    'num_heads': 12,
                    'output_dim': 768,
                    'dropout_rate': 0.1,
                    'name': 'self',
                    'is_decoder': True,
                    'relative_attention_num_buckets': 32
                },
                'residual_dropout': 0.1,
                'dim': 768,
                'eps': 1e-6,
                'poswise_feedforward': {
                    'layers': [
                        {
                            'type': 'Linear',
                            'kwargs': {
                                'in_features': 768,
                                'out_features': 3072,
                                'bias': False
                            }
                        },
                        {'type': 'ReLU'},
                        {
                            'type': 'Linear',
                            'kwargs': {
                                'in_features': 3072,
                                'out_features': 768,
                                'bias': False
                            }
                        }
                    ]
                }
            }
        }

        self.assertDictEqual(model_config, exp_config)


if __name__ == "__main__":
    unittest.main()
