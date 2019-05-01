import unittest

import torch
from torch import nn

from texar.core import layers


class ReducePoolingLayerTest(unittest.TestCase):
    """Tests reduce pooling layer.
    """
    def setUp(self):
        unittest.TestCase.setUp(self)

        self._batch_size = 64
        self._emb_dim = 100
        self._seq_length = 16

    def test_max_reduce_pooling_layer(self):
        """Tests :class:`texar.core.MaxReducePool1d`."""

        pool_layer = layers.MaxReducePool1d()

        inputs = torch.randn(self._batch_size, self._emb_dim, self._seq_length)
        output = pool_layer(inputs)
        output_reduce, _ = torch.max(inputs, dim=2, keepdim=True)
        self.assertEqual(output.shape, torch.Size([self._batch_size, self._emb_dim, 1]))
        self.assertEqual(torch.all(torch.eq(output, output_reduce)), 1)

    def test_average_reduce_pooling_layer(self):
        """Tests :class:`texar.core.AvgReducePool1d`."""

        pool_layer = layers.AvgReducePool1d()

        inputs = torch.randn(self._batch_size, self._emb_dim, self._seq_length)
        output = pool_layer(inputs)
        output_reduce = torch.mean(inputs, dim=2, keepdim=True)
        self.assertEqual(output.shape, torch.Size([self._batch_size, self._emb_dim, 1]))
        self.assertEqual(torch.all(torch.eq(output, output_reduce)), 1)


class MergeLayerTest(unittest.TestCase):
    """Tests MergeLayer.
    """

    def test_layer_logics(self):
        """Test the logic of MergeLayer.
        """
        layers_ = []
        layers_.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3))
        layers_.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4))
        layers_.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5))
        layers_.append(nn.Linear(in_features=10, out_features=64))
        layers_.append(nn.Linear(in_features=10, out_features=64))
        m_layer = layers.MergeLayer(layers_)

        input = torch.randn(32, 32, 10)
        output = m_layer(input)
        self.assertEqual(output.shape, torch.Size([32, 32, 149]))

