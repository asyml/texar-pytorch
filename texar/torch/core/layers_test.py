"""
Unit tests for various layers.
"""
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from texar.torch.core import layers


class GetActivationFnTest(unittest.TestCase):
    r"""Tests :func:`texar.torch.core.layers.get_activation_fn`.
    """

    def test_get_activation_fn(self):
        r"""Tests.
        """
        fn = layers.get_activation_fn()
        self.assertEqual(fn, None)

        fn = layers.get_activation_fn('relu')
        self.assertEqual(fn, torch.relu)

        inputs = torch.randn(64, 100)

        fn = layers.get_activation_fn('leaky_relu')
        fn_output = fn(inputs)
        ref_output = F.leaky_relu(inputs)
        self.assertEqual(torch.all(torch.eq(fn_output, ref_output)), 1)

        fn = layers.get_activation_fn('leaky_relu',
                                      kwargs={'negative_slope': 0.1})
        fn_output = fn(inputs)
        ref_output = F.leaky_relu(inputs, negative_slope=0.1)
        self.assertEqual(torch.all(torch.eq(fn_output, ref_output)), 1)


class GetLayerTest(unittest.TestCase):
    r"""Tests layer creator.
    """

    def test_get_layer(self):
        r"""Tests :func:`texar.torch.core.layers.get_layer`.
        """
        hparams = {"type": "Conv1d",
                   "kwargs": {"in_channels": 16,
                              "out_channels": 32,
                              "kernel_size": 2}
                   }

        layer = layers.get_layer(hparams)
        self.assertIsInstance(layer, nn.Conv1d)

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
        self.assertIsInstance(layer, layers.MergeLayer)

        hparams = {"type": "Conv1d",
                   "kwargs": {"in_channels": 16,
                              "out_channels": 32,
                              "kernel_size": 2}
                   }
        layer = layers.get_layer(hparams)
        self.assertIsInstance(layer, nn.Conv1d)

        hparams = {
            "type": nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        }
        layer = layers.get_layer(hparams)
        self.assertIsInstance(layer, nn.Conv1d)


class ReducePoolingLayerTest(unittest.TestCase):
    r"""Tests reduce pooling layer.
    """

    def setUp(self):
        unittest.TestCase.setUp(self)

        self._batch_size = 64
        self._emb_dim = 100
        self._seq_length = 16

    def test_max_reduce_pooling_layer(self):
        r"""Tests :class:`texar.torch.core.MaxReducePool1d`."""

        pool_layer = layers.MaxReducePool1d()
        inputs = torch.randn(self._batch_size, self._emb_dim, self._seq_length)
        output = pool_layer(inputs)
        output_reduce, _ = torch.max(inputs, dim=2)
        self.assertEqual(output.shape, torch.Size([self._batch_size,
                                                   self._emb_dim]))
        self.assertEqual(torch.all(torch.eq(output, output_reduce)), 1)

    def test_average_reduce_pooling_layer(self):
        r"""Tests :class:`texar.torch.core.AvgReducePool1d`."""

        pool_layer = layers.AvgReducePool1d()
        inputs = torch.randn(self._batch_size, self._emb_dim, self._seq_length)
        output = pool_layer(inputs)
        output_reduce = torch.mean(inputs, dim=2)
        self.assertEqual(output.shape, torch.Size([self._batch_size,
                                                   self._emb_dim]))
        self.assertEqual(torch.all(torch.eq(output, output_reduce)), 1)


class MergeLayerTest(unittest.TestCase):
    r"""Tests MergeLayer.
    """

    def test_layer_logic(self):
        r"""Test the logic of MergeLayer.
        """
        layers_ = list()
        layers_.append(nn.Conv1d(in_channels=32, out_channels=32,
                                 kernel_size=3))
        layers_.append(nn.Conv1d(in_channels=32, out_channels=32,
                                 kernel_size=3))
        layers_.append(nn.Conv1d(in_channels=32, out_channels=32,
                                 kernel_size=3))

        modes = ["concat", "sum", "mean", "prod", "max", "min", "logsumexp",
                 "elemwise_sum", "elemwise_mul"]

        for mode in modes:
            m_layer = layers.MergeLayer(layers_, mode=mode)
            input = torch.randn(32, 32, 10)
            output = m_layer(input)

            if mode == "concat":
                self.assertEqual(output.shape, torch.Size([32, 32, 24]))
            elif mode == "elemwise_sum" or mode == "elemwise_mul":
                self.assertEqual(output.shape, torch.Size([32, 32, 8]))
            else:
                self.assertEqual(output.shape, torch.Size([32, 32]))

        for mode in ["and", "or"]:
            m_layer = layers.MergeLayer(layers=None, mode=mode)
            input = torch.ones(32, 32, 10, dtype=torch.uint8)
            output = m_layer(input)

            self.assertEqual(output.shape, torch.Size([32, 32]))

    def test_empty_merge_layer(self):
        r"""Test the output of MergeLayer with empty layers.
        """
        m_layer = layers.MergeLayer(layers=None)
        input = torch.randn(32, 32, 10)
        output = m_layer(input)
        self.assertEqual(torch.all(torch.eq(output, input)), 1)


if __name__ == "__main__":
    unittest.main()
