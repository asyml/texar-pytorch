#
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
