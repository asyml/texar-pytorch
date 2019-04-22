#
"""
Unit tests for embedder utils.
"""

# pylint: disable=no-member

import unittest

from texar.modules.embedders import embedder_utils

class GetEmbeddingTest(unittest.TestCase):
    """Tests embedding creator.
    """
    def test_get_embedding(self):
        """Tests :func:`~texar.modules.embedder.embedder_utils.get_embedding`.
        """
        vocab_size = 100
        emb = embedder_utils.get_embedding(num_embeds=vocab_size)
        self.assertEqual(emb.shape[0], vocab_size)
        self.assertEqual(emb.shape[1],
                         embedder_utils.default_embedding_hparams()["dim"])

        hparams = {
            "initializer": {
                "type": "torch.nn.init.uniform_",
                "kwargs": {'a': -0.1, 'b': 0.1}
            }
        }
        emb = embedder_utils.get_embedding(
            hparams=hparams, num_embeds=vocab_size,)
        self.assertEqual(emb.shape[0], vocab_size)
        self.assertEqual(emb.shape[1],
                         embedder_utils.default_embedding_hparams()["dim"])


if __name__ == "__main__":
    unittest.main()
