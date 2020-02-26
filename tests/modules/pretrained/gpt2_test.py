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
Unit tests for GPT2 utils.
"""

import os
import unittest

from texar.torch.modules.pretrained.gpt2 import *
from texar.torch.utils.test import pretrained_test


class GPT2UtilsTest(unittest.TestCase):
    r"""Tests GPT2 utils.
    """

    @pretrained_test
    def test_load_pretrained_gpt2_AND_transform_gpt2_to_texar_config(self):
        pretrained_model_dir = PretrainedGPT2Mixin.download_checkpoint(
            pretrained_model_name="gpt2-small")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('checkpoint', files)
        self.assertIn('encoder.json', files)
        self.assertIn('hparams.json', files)
        self.assertIn('model.ckpt.data-00000-of-00001', files)
        self.assertIn('model.ckpt.index', files)
        self.assertIn('model.ckpt.meta', files)
        self.assertIn('vocab.bpe', files)

        model_config = PretrainedGPT2Mixin._transform_config(
            pretrained_model_name="gpt2-small",
            cache_dir=pretrained_model_dir)

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

            'encoder': {
                'dim': 768,
                'num_blocks': 12,
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
                'eps': 1e-6,
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
                },
                'use_bert_config': False
            }
        }

        self.assertDictEqual(model_config, exp_config)


if __name__ == "__main__":
    unittest.main()
