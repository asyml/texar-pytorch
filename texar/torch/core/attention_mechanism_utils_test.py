"""
Unit tests for attention mechanism utils.
"""

import unittest

import torch

from texar.torch.core.attention_mechanism import (
    maybe_mask_score, prepare_memory, safe_cumprod)
from texar.torch.core.attention_mechanism_utils import hardmax, sparsemax


class AttentionMechanismUtilsTest(unittest.TestCase):
    r"""Tests attention mechanism utils.
    """

    def test_prepare_memory(self):
        r"""Tests `texar.torch.core.attention_mechanism_utils.prepare_memory`.
        """
        memory = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [9, 8, 7]],
                               [[3, 2, 1], [6, 5, 4], [9, 8, 7], [4, 5, 6]]])
        memory_sequence_length = torch.tensor([3, 2])
        masked_memory = prepare_memory(memory, memory_sequence_length)

        expected_memory = [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
                           [[3, 2, 1], [6, 5, 4], [0, 0, 0], [0, 0, 0]]]

        self.assertEqual(masked_memory.tolist(), expected_memory)

    def test_maybe_mask_score(self):
        r"""Tests `texar.torch.core.attention_mechanism_utils.maybe_mask_score`.
        """
        score = torch.tensor([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]])
        score_mask_value = torch.tensor(-1)
        memory_sequence_length = torch.tensor([1, 2, 3])
        masked_score = maybe_mask_score(
            score, score_mask_value, memory_sequence_length)

        expected_score = [[1, -1, -1], [4, 5, -1], [7, 8, 9]]

        self.assertEqual(masked_score.tolist(), expected_score)

    def test_hardmax(self):
        r"""Tests `texar.torch.core.attention_mechanism_utils.hardmax`.
        """
        logits = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [9, 8, 7]],
                               [[3, 2, 1], [6, 5, 4], [9, 8, 7], [4, 5, 6]]])
        outputs = hardmax(logits)

        expected_outputs = [[[0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]],
                            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]]]

        self.assertEqual(outputs.tolist(), expected_outputs)

    def test_safe_cumprod(self):
        r"""Tests `texar.torch.core.attention_mechanism_utils.safe_cumprod`.
        """
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        outputs = safe_cumprod(x, dim=0)

        expected_outputs = [0.1, 0.02, 0.006, 0.0024, 0.0012]

        outputs = outputs.tolist()
        for i in range(5):
            self.assertAlmostEqual(outputs[i], expected_outputs[i])

    def test_sparsemax(self):
        r"""Tests `texar.torch.core.attention_mechanism_utils.sparsemax`.
        """
        logits = torch.tensor([[1.2, 3.2, 0.1, 4.3, 5.0],
                               [3.3, 4.6, 9.5, 5.4, 8.7]])
        outputs = sparsemax(logits)

        expected_outputs = [[0.0, 0.0, 0.0, 0.1500001, 0.8499999],
                            [0.0, 0.0, 0.8999996, 0.0, 0.09999943]]

        outputs = outputs.tolist()
        for i in range(2):
            for j in range(5):
                self.assertAlmostEqual(outputs[i][j],
                                       expected_outputs[i][j],
                                       places=5)


if __name__ == "__main__":
    unittest.main()
