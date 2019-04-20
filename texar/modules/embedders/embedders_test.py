#
"""
Unit tests for embedders.
"""

# pylint: disable=no-member


import torch
import numpy as np
import unittest


from texar.modules.embedders.embedders import WordEmbedder
from texar.modules.embedders.position_embedders import PositionEmbedder

class EmbedderTest(unittest.TestCase):
    """Tests parameterized embedder.
    """

    def _test_word_embedder(self, hparams):
        """Tests :class:`texar.modules.WordEmbedder`.
        """
        embedder = WordEmbedder(
            vocab_size=100, hparams=hparams)

        inputs = torch.ones([64, 16], dtype=torch.int32)
        outputs = embedder(inputs)

        inputs_soft = torch.ones([64, 16, embedder.vocab_size], dtype=torch.float32)
        outputs_soft = embedder(soft_ids=inputs_soft)

        emb_dim = embedder.dim
        if not isinstance(emb_dim, (list, tuple)):
            emb_dim = [emb_dim]

        hparams_dim = hparams["dim"]
        if not isinstance(hparams["dim"], (list, tuple)):
            hparams_dim = [hparams["dim"]]

        self.assertEqual(list(outputs.shape), [64, 16] + emb_dim)
        self.assertEqual(list(outputs_soft.shape), [64, 16] + emb_dim)
        self.assertEqual(emb_dim, hparams_dim)
        self.assertEqual(embedder.vocab_size, 100)
        self.assertEqual(tuple(outputs.shape), (64, 16) + tuple(emb_dim))
        self.assertEqual(tuple(outputs_soft.shape), (64, 16) + tuple(emb_dim))

    def _test_position_embedder(self, hparams):
        """Tests :class:`texar.modules.PositionEmbedder`.
        """
        pos_size = 100
        embedder = PositionEmbedder(
            position_size=pos_size, hparams=hparams)
        inputs = torch.ones([64, 16], dtype=torch.int32)
        outputs = embedder(inputs)

        emb_dim = embedder.dim
        if not isinstance(emb_dim, (list, tuple)):
            emb_dim = [emb_dim]

        hparams_dim = hparams["dim"]
        if not isinstance(hparams["dim"], (list, tuple)):
            hparams_dim = [hparams["dim"]]

        self.assertEqual(list(outputs.shape), [64, 16] + emb_dim)
        self.assertEqual(emb_dim, hparams_dim)
        self.assertEqual(embedder.position_size, 100)
        seq_length = torch.empty([64]).uniform_(pos_size).type(torch.int32)
        outputs = embedder(sequence_length=seq_length)

    def test_embedder(self):
        """Tests various embedders.
        """
        # no dropout
        hparams = {"dim": 1024, "dropout_rate": 0}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        # dropout with default strategy
        hparams = {"dim": 1024, "dropout_rate": 0.3}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0.3}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0.3}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        # dropout with different strategies
        hparams = {"dim": 1024, "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        hparams = {"dim": 1024, "dropout_rate": 0.3,
                   "dropout_strategy": "item_type"}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0.3,
                   "dropout_strategy": "item_type"}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0.3,
                   "dropout_strategy": "item_type"}
        self._test_word_embedder(hparams)
        self._test_position_embedder(hparams)

    def test_embedder_multi_calls(self):
        """Tests embedders called by multiple times.
        """
        hparams = {"dim": 1024, "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        embedder = WordEmbedder(
            vocab_size=100, hparams=hparams)
        inputs = torch.ones([64, 16], dtype=torch.int32)
        outputs = embedder(inputs)

        emb_dim = embedder.dim
        if not isinstance(emb_dim, (list, tuple)):
            emb_dim = [emb_dim]
        self.assertEqual(list(outputs.shape), [64, 16] + emb_dim)

        # Call with inputs in a different shape
        inputs = torch.ones([64, 10, 20], dtype=torch.int32)
        outputs = embedder(inputs)

        emb_dim = embedder.dim
        if not isinstance(emb_dim, (list, tuple)):
            emb_dim = [emb_dim]
        self.assertEqual(list(outputs.shape), [64, 10, 20] + emb_dim)

    def test_word_embedder_soft_ids(self):
        """Tests the correctness of using soft ids.
        """
        init_value = np.expand_dims(np.arange(5), 1)
        embedder = WordEmbedder(init_value=init_value)

        ids = np.array([3])
        soft_ids = np.array([[0, 0, 0, 1, 0]])

        outputs = embedder(ids=torch.from_numpy(ids))
        soft_outputs = embedder(soft_ids=torch.from_numpy(soft_ids))
        self.assertEqual(outputs, soft_outputs)

if __name__ == "__main__":
    unittest.main()
