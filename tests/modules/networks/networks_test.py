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
Unit tests for feed forward neural networks.
"""

import unittest

import torch

from texar.torch.modules.networks.networks import FeedForwardNetwork


class FeedForwardNetworkTest(unittest.TestCase):
    """Tests the class
    :class:`~texar.torch.modules.networks.networks.FeedForwardNetwork`.
    """

    def test_feedforward(self):
        """Tests feed-forward.
        """
        hparams = {
            "layers": [
                {
                    "type": "torch.nn.Linear",
                    "kwargs": {
                        "in_features": 32,
                        "out_features": 64
                    }
                },
                {
                    "type": "torch.nn.Linear",
                    "kwargs": {
                        "in_features": 64,
                        "out_features": 128
                    }
                }
            ]
        }

        nn = FeedForwardNetwork(hparams=hparams)

        self.assertEqual(len(nn.layers), len(hparams["layers"]))
        outputs = nn(torch.ones(64, 16, 32))
        self.assertEqual(len(nn.trainable_variables),
                         len(hparams["layers"]) * 2)
        self.assertEqual(outputs.size(-1), nn.output_size)


if __name__ == "__main__":
    unittest.main()
