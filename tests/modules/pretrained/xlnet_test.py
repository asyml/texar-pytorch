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
Unit tests for XLNet utils.
"""

import os
import unittest

from texar.torch.modules.pretrained.xlnet import *
from texar.torch.utils.test import pretrained_test


class XLNetUtilsTest(unittest.TestCase):
    r"""Tests XLNet utils.
    """

    @pretrained_test
    def test_load_pretrained_xlnet_AND_transform_xlnet_to_texar_config(self):

        pretrained_model_dir = PretrainedXLNetMixin.download_checkpoint(
            pretrained_model_name="xlnet-base-cased")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('spiece.model', files)
        self.assertIn('xlnet_model.ckpt.meta', files)
        self.assertIn('xlnet_model.ckpt.data-00000-of-00001', files)
        self.assertIn('xlnet_model.ckpt.index', files)
        self.assertIn('xlnet_config.json', files)

        model_config = PretrainedXLNetMixin._transform_config(
            pretrained_model_name="xlnet-base-cased",
            cache_dir=pretrained_model_dir)

        exp_config = {'head_dim': 64,
                      'ffn_inner_dim': 3072,
                      'hidden_dim': 768,
                      'activation': 'gelu',
                      'num_heads': 12,
                      'num_layers': 12,
                      'vocab_size': 32000,
                      'untie_r': True}

        self.assertDictEqual(model_config, exp_config)


if __name__ == "__main__":
    unittest.main()
