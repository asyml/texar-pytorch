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
Unit tests for embedder utils.
"""

import unittest

from texar.torch.modules.embedders import embedder_utils


class GetEmbeddingTest(unittest.TestCase):
    """Tests embedding creator.
    """

    def test_get_embedding(self):
        """Tests :func:`~texar.torch.modules.embedder.embedder_utils.get_embedding`.
        """
        vocab_size = 100
        emb = embedder_utils.get_embedding(num_embeds=vocab_size)
        self.assertEqual(emb.size(0), vocab_size)
        self.assertEqual(emb.size(1),
                         embedder_utils.default_embedding_hparams()["dim"])

        hparams = {
            "initializer": {
                "type": "torch.nn.init.uniform_",
                "kwargs": {'a': -0.1, 'b': 0.1}
            }
        }
        emb = embedder_utils.get_embedding(
            hparams=hparams, num_embeds=vocab_size, )
        self.assertEqual(emb.size(0), vocab_size)
        self.assertEqual(emb.size(1),
                         embedder_utils.default_embedding_hparams()["dim"])


if __name__ == "__main__":
    unittest.main()
