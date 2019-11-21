"""
Unit tests for sampler related operations.
"""
import unittest
from typing import no_type_check

import numpy as np

from texar.torch.data.data.data_base import (
    DatasetBase, DataSource, IterDataSource, SequenceDataSource)
from texar.torch.data.data.sampler import BufferShuffleSampler


class SamplerTest(unittest.TestCase):
    r"""Tests samplers.
    """

    class MockDataBase(DatasetBase):
        def __init__(self, size: int, lazy_strategy: str,
                     cache_strategy: str, unknown_size: bool = False):
            data = list(range(size))
            source: DataSource[int]
            if unknown_size:
                source = IterDataSource(data)
            else:
                source = SequenceDataSource(data)
            hparams = {
                'lazy_strategy': lazy_strategy,
                'cache_strategy': cache_strategy,
            }
            super().__init__(source, hparams=hparams)

    def setUp(self) -> None:
        self.size = 10
        self.buffer_size = 5

    @no_type_check
    def _test_data(self, data: DatasetBase,
                   returns_data: bool = False,
                   always_returns_data: bool = False):
        sampler = BufferShuffleSampler(data, self.buffer_size)
        for epoch in range(2):
            indices = list(iter(sampler))
            if always_returns_data or (returns_data and epoch == 0):
                examples = [ex[1] for ex in indices]
                indices = [ex[0] for ex in indices]
                np.testing.assert_array_equal(indices, examples)
            self.assertEqual(len(set(indices)), self.size)
            self.assertEqual(min(indices), 0)
            self.assertEqual(max(indices), self.size - 1)
            data._fully_cached = True

    def test_known_size(self):
        data = self.MockDataBase(self.size, 'none', 'processed')
        self._test_data(data)
        data = self.MockDataBase(self.size, 'all', 'none', unknown_size=True)
        self._test_data(data, always_returns_data=True)

    def test_non_lazy_loading(self):
        strategies = [
            ('none', 'processed'),
            ('process', 'loaded'),
            ('process', 'processed'),
        ]
        for lazy, cache in strategies:
            data = self.MockDataBase(self.size, lazy, cache)
            self._test_data(data)

    def test_lazy_loading(self):
        data = self.MockDataBase(self.size, 'all', 'loaded', unknown_size=True)
        self._test_data(data, returns_data=True)
        data = self.MockDataBase(self.size, 'all', 'processed',
                                 unknown_size=True)
        self._test_data(data, returns_data=True)


if __name__ == "__main__":
    unittest.main()
