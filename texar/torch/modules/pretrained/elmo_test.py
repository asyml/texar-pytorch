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
Unit tests for ELMo utils.
"""

import os
import unittest

from texar.torch.modules.pretrained.elmo import *
from texar.torch.utils.test import pretrained_test


class ELMoUtilsTest(unittest.TestCase):
    r"""Tests ELMo Utils.
    """

    @pretrained_test
    def test_load_pretrained_elmo_AND_transform_elmo_to_texar_config(self):
        pretrained_model_dir = PretrainedELMoMixin.download_checkpoint(
            pretrained_model_name="elmo-small")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5', files)
        self.assertIn('elmo_2x1024_128_2048cnn_1xhighway_options.json', files)

        model_config = PretrainedELMoMixin._transform_config(
            pretrained_model_name="elmo-small",
            cache_dir=pretrained_model_dir)

        exp_config = {
            'encoder': {
                "lstm": {
                    "use_skip_connections": True,
                    "projection_dim": 128,
                    "cell_clip": 3,
                    "proj_clip": 3,
                    "dim": 1024,
                    "n_layers": 2
                },
                "char_cnn": {
                    "activation": "relu",
                    "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256],
                                [6, 512], [7, 1024]],
                    "n_highway": 1,
                    "embedding": {
                        "dim": 16
                    },
                    "n_characters": 262,
                    "max_characters_per_token": 50
                }
            },
        }

        self.assertDictEqual(model_config, exp_config)


if __name__ == "__main__":
    unittest.main()
