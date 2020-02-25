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
Unit test for gin utility functions
"""

import unittest

import os
import tempfile

from texar.torch.modules.pretrained.t5_utils import read_t5_gin_config_file


class GinTest(unittest.TestCase):
    r"""Tests for T5 gin file.

    """

    def setUp(self):
        test_gin = '''d_ff = 2048\n''' \
                   '''d_kv = 64\n''' \
                   '''d_model = 512\n'''\
                   '''dropout_rate = 0.1\n''' \
                   '''inputs_length = 512\n''' \
                   '''mean_noise_span_length = 3.0\n''' \
                   '''MIXTURE_NAME = 'all_mix'\n''' \
                   '''noise_density = 0.15\n''' \
                   '''num_heads = 8\n''' \
                   '''num_layers = 6\n''' \
                   '''targets_length = 512\n''' \
                   '''init_checkpoint = "gs://t5-data/pretrained_models/''' \
                   '''small/model.ckpt-1000000"\n''' \
                   '''tokens_per_batch = 1048576\n''' \
                   '''\n''' \
                   '''AdafactorOptimizer.beta1 = 0.0\n''' \
                   '''AdafactorOptimizer.clipping_threshold = 1.0\n''' \
                   '''AdafactorOptimizer.decay_rate = None\n''' \
                   '''AdafactorOptimizer.epsilon1 = 1e-30\n''' \
                   '''AdafactorOptimizer.epsilon2 = 0.001\n''' \
                   '''AdafactorOptimizer.min_dim_size_to_factor = 128\n''' \
                   '''AdafactorOptimizer.multiply_by_parameter_scale = ''' \
                   '''True\n''' \
                   '''Bitransformer.shared_embedding = True\n'''
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.gin_file = os.path.join(self.tmp_dir.name, 'config.gin')
        with open(self.gin_file, 'w') as gin_writer:
            gin_writer.write(test_gin)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_read_t5_gin_config_file(self):
        r"""Tests :meth:`~texar.torch.utils.gin.read_t5_gin_config_file`.
        """
        config = read_t5_gin_config_file(self.gin_file)

        expect_config = {'d_ff': 2048,
                         'd_kv': 64,
                         'd_model': 512,
                         'dropout_rate': 0.1,
                         'inputs_length': 512,
                         'num_heads': 8,
                         'num_layers': 6
                         }

        self.assertEqual(config, expect_config)


if __name__ == "__main__":
    unittest.main()
