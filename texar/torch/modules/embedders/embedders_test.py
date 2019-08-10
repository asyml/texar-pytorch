#
"""
Unit tests for embedders.
"""

import unittest

import numpy as np
import torch

from texar.torch.modules.embedders.embedders import WordEmbedder
from texar.torch.modules.embedders.position_embedders import (
    PositionEmbedder, SinusoidsPositionEmbedder)


class EmbedderTest(unittest.TestCase):
    """Tests parameterized embedder.
    """

    def _test_word_embedder(self, hparams):
        """Tests :class:`texar.torch.modules.WordEmbedder`.
        """
        embedder = WordEmbedder(
            vocab_size=100, hparams=hparams)

        inputs = torch.randint(embedder.vocab_size, (64, 16), dtype=torch.long)
        outputs = embedder(inputs)

        inputs_soft = torch.randn(
            (64, 16, embedder.vocab_size), dtype=torch.float32)
        outputs_soft = embedder(soft_ids=inputs_soft)

        if isinstance(embedder.dim, (list, tuple)):
            emb_dim = tuple(embedder.dim)
        else:
            emb_dim = (embedder.dim,)

        if isinstance(hparams["dim"], (list, tuple)):
            hparams_dim = tuple(hparams["dim"])
        else:
            hparams_dim = (hparams["dim"],)

        self.assertEqual(outputs.size(), (64, 16) + emb_dim)
        self.assertEqual(outputs.size(-1), embedder.output_size)
        self.assertEqual(outputs_soft.size(), (64, 16) + emb_dim)
        self.assertEqual(outputs_soft.size(-1), embedder.output_size)
        self.assertEqual(emb_dim, hparams_dim)
        self.assertEqual(embedder.vocab_size, 100)
        self.assertEqual(outputs.size(), (64, 16) + emb_dim)
        self.assertEqual(outputs_soft.size(), (64, 16) + emb_dim)

    def _test_position_embedder(self, hparams):
        """Tests :class:`texar.torch.modules.PositionEmbedder`.
        """
        pos_size = 100
        embedder = PositionEmbedder(
            position_size=pos_size, hparams=hparams)
        inputs = torch.randint(embedder.num_embeds, (64, 16), dtype=torch.long)
        outputs = embedder(inputs)

        if isinstance(embedder.dim, (list, tuple)):
            emb_dim = tuple(embedder.dim)
        else:
            emb_dim = (embedder.dim,)

        if isinstance(hparams["dim"], (list, tuple)):
            hparams_dim = tuple(hparams["dim"])
        else:
            hparams_dim = (hparams["dim"],)

        self.assertEqual(outputs.size(), (64, 16) + emb_dim)
        self.assertEqual(outputs.size(-1), embedder.output_size)
        self.assertEqual(emb_dim, hparams_dim)
        self.assertEqual(embedder.position_size, 100)
        seq_length = torch.empty(64).uniform_(0, pos_size).long()
        outputs = embedder(sequence_length=seq_length)

    def test_sinusoids_position_embedder(self):
        """Tests :class:`texar.torch.modules.SinusoidsPositionEmbedder`.
        """
        position_size = 64
        input_size = (23, 18)
        hparams = {'dim': 513}  # use odd dimension to ensure padding correct
        embedder = SinusoidsPositionEmbedder(position_size, hparams=hparams)
        inputs = torch.randint(position_size - 1, input_size)
        outputs = embedder(inputs)
        self.assertEqual(outputs.size(), input_size + (hparams['dim'],))
        self.assertEqual(outputs.size(-1), embedder.output_size)

        embedder_no_cache = SinusoidsPositionEmbedder(
            None, hparams={**hparams, 'cache_embeddings': False})
        wide_inputs = torch.randint(
            -position_size, position_size * 2, input_size)
        wide_outputs = embedder_no_cache(wide_inputs)
        self.assertEqual(wide_outputs.size(), input_size + (hparams['dim'],))
        no_cache_outputs = embedder_no_cache(inputs)
        np.testing.assert_array_equal(outputs, no_cache_outputs)

    def test_embedder(self):
        """Tests various embedders.
        """
        test_dims = [
            14,
            [14],
            [14, 10],
        ]
        test_hparams = [
            # no dropout
            {"dropout_rate": 0},
            # dropout with default strategy
            {"dropout_rate": 0.3},
            # dropout with different strategies
            {"dropout_rate": 0.3, "dropout_strategy": "item"},
            {"dropout_rate": 0.3, "dropout_strategy": "item_type"},
        ]

        for base_hparams in test_hparams:
            for dim in test_dims:
                hparams = base_hparams.copy()
                hparams['dim'] = dim
                self._test_word_embedder(hparams)
                self._test_position_embedder(hparams)

    def test_embedder_multi_calls(self):
        """Tests embedders called by multiple times.
        """
        hparams = {"dim": 26, "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        embedder = WordEmbedder(
            vocab_size=100, hparams=hparams)
        inputs = torch.randint(embedder.vocab_size, (64, 16), dtype=torch.long)
        outputs = embedder(inputs)

        if isinstance(embedder.dim, (list, tuple)):
            emb_dim = tuple(embedder.dim)
        else:
            emb_dim = (embedder.dim,)
        self.assertEqual(outputs.size(), (64, 16) + emb_dim)

        # Call with inputs in a different shape
        inputs = torch.randint(
            embedder.vocab_size, (64, 10, 20), dtype=torch.long)
        outputs = embedder(inputs)

        self.assertEqual(outputs.size(), (64, 10, 20) + emb_dim)

    def test_word_embedder_soft_ids(self):
        """Tests the correctness of using soft ids.
        """
        init_value = np.expand_dims(np.arange(5), 1)
        embedder = WordEmbedder(init_value=init_value)

        ids = torch.tensor([3])
        soft_ids = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float)

        outputs = embedder(ids=ids)
        soft_outputs = embedder(soft_ids=soft_ids)
        self.assertEqual(outputs, soft_outputs)


if __name__ == "__main__":
    unittest.main()
