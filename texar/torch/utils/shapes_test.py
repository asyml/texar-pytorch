"""
Unit tests for shape-related utility functions.
"""

import unittest

import numpy as np
import torch

from texar.torch.utils import shapes


class ShapesTest(unittest.TestCase):
    r"""Tests shape-related utility functions.
    """

    def test_mask_sequences(self):
        r"""Tests :func:`texar.torch.utils.shapes.mask_sequences`.
        """
        seq = torch.ones(3, 4, 3, dtype=torch.int32)
        seq_length = torch.tensor([3, 2, 1], dtype=torch.int32)

        masked_seq = shapes.mask_sequences(seq, seq_length)
        np.testing.assert_array_equal(masked_seq.shape, seq.shape)
        seq_sum = torch.sum(masked_seq, dim=(1, 2))
        np.testing.assert_array_equal(seq_sum, seq_length * 3)

    def test_pad_and_concat_long(self):
        r"""Test :func:`texar.torch.utils.shapes.pad_and_concat` with
        torch.LongTensor.
        """
        a = torch.ones(3, 10, 2, dtype=torch.long)
        b = torch.ones(4, 20, 3, dtype=torch.long)
        c = torch.ones(5, 1, 4, dtype=torch.long)

        t = shapes.pad_and_concat([a, b, c], 0)
        np.testing.assert_array_equal(t.shape, [3 + 4 + 5, 20, 4])
        t = shapes.pad_and_concat([a, b, c], 1)
        np.testing.assert_array_equal(t.shape, [5, 10 + 20 + 1, 4])
        t = shapes.pad_and_concat([a, b, c], 2)
        np.testing.assert_array_equal(t.shape, [5, 20, 2 + 3 + 4])

    def test_pad_and_concat_float(self):
        r"""Test :func:`texar.torch.utils.shapes.pad_and_concat` with
        torch.FloatTensor.
        """
        a = torch.ones(3, 10, 2)
        b = torch.ones(4, 20, 3)
        c = torch.ones(5, 1, 4)

        t = shapes.pad_and_concat([a, b, c], 0)
        np.testing.assert_array_equal(t.shape, [3 + 4 + 5, 20, 4])
        t = shapes.pad_and_concat([a, b, c], 1)
        np.testing.assert_array_equal(t.shape, [5, 10 + 20 + 1, 4])
        t = shapes.pad_and_concat([a, b, c], 2)
        np.testing.assert_array_equal(t.shape, [5, 20, 2 + 3 + 4])


if __name__ == '__main__':
    unittest.main()
