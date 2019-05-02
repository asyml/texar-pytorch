"""
Unit tests for various layers.
"""
import unittest

import torch
from torch import nn
import torch.nn.functional as F

import texar as tx
from texar.core import layers


class GetActivationFnTest(unittest.TestCase):
    """Tests :func:`texar.core.layers.get_activation_fn`.
    """
    def test_get_activation_fn(self):
        """Tests.
        """
        fn = layers.get_activation_fn()
        self.assertEqual(fn, None)

        fn = layers.get_activation_fn('relu')
        self.assertEqual(fn, F.relu)

        inputs = torch.randn(64, 100)

        fn = layers.get_activation_fn('leaky_relu')
        fn_output = fn(inputs)
        ref_output = F.leaky_relu(inputs)
        self.assertEqual(torch.all(torch.eq(fn_output, ref_output)), 1)

        fn = layers.get_activation_fn('leaky_relu', kwargs={'negative_slope': 0.1})
        fn_output = fn(inputs)
        ref_output = F.leaky_relu(inputs, negative_slope=0.1)
        self.assertEqual(torch.all(torch.eq(fn_output, ref_output)), 1)


class GetLayerTest(unittest.TestCase):
    """Tests layer creator.
    """
    def test_get_layer(self):
        """Tests :func:`texar.core.layers.get_layer`.
        """
        hparams = {"type": "Conv1d",
                   "kwargs": {"in_channels": 16,
                              "out_channels": 32,
                              "kernel_size": 2}
                   }

        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, nn.Conv1d))

        hparams = {
            "type": "MergeLayer",
            "kwargs": {
                "layers": [
                    {"type": "Conv1d",
                     "kwargs": {"in_channels": 16,
                                "out_channels": 32,
                                "kernel_size": 2}},
                    {"type": "Conv1d",
                     "kwargs": {"in_channels": 16,
                                "out_channels": 32,
                                "kernel_size": 2}}
                ]
            }
        }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, tx.core.MergeLayer))

        hparams = {"type": "Conv1d",
                   "kwargs": {"in_channels": 16,
                              "out_channels": 32,
                              "kernel_size": 2}
                   }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, nn.Conv1d))

        hparams = {
            "type": nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, nn.Conv1d))


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
        layers_ = list()
        layers_.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3))
        layers_.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4))
        layers_.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5))
        layers_.append(nn.Linear(in_features=10, out_features=64))
        layers_.append(nn.Linear(in_features=10, out_features=64))
        m_layer = layers.MergeLayer(layers_)

        input = torch.randn(32, 32, 10)
        output = m_layer(input)
        self.assertEqual(output.shape, torch.Size([32, 32, 149]))

