"""
Unit tests for conv encoders.
"""
import unittest

import torch
from torch import nn

from texar.torch.core.layers import MergeLayer
from texar.torch.modules.encoders.conv_encoders import Conv1DEncoder


class Conv1DEncoderTest(unittest.TestCase):
    r"""Tests :class:`~texar.torch.modules.Conv1DEncoder` class.
    """

    def test_encode(self):
        r"""Tests encode.
        """
        inputs_1 = torch.ones([128, 32, 300])
        encoder_1 = Conv1DEncoder(in_channels=inputs_1.size(1),
                                  in_features=inputs_1.size(2))
        self.assertEqual(len(encoder_1.layers), 4)
        self.assertIsInstance(
            encoder_1.layer_by_name("MergeLayer"), MergeLayer)
        for layer in encoder_1.layers[0].layers:
            self.assertIsInstance(layer, nn.Sequential)

        outputs_1 = encoder_1(inputs_1)
        self.assertEqual(outputs_1.size(), torch.Size([128, 256]))
        self.assertEqual(outputs_1.size(-1), encoder_1.output_size)

        inputs_2 = torch.ones([128, 64, 300])
        hparams = {
            # Conv layers
            "num_conv_layers": 2,
            "out_channels": 128,
            "kernel_size": [[3, 4, 5], 4],
            "other_conv_kwargs": {"padding": 0},
            # Pooling layers
            "pooling": "AvgPool1d",
            "pool_size": 2,
            "pool_stride": 1,
            # Dense layers
            "num_dense_layers": 3,
            "out_features": [128, 128, 10],
            "dense_activation": "ReLU",
            "other_dense_kwargs": None,
            # Dropout
            "dropout_conv": [0, 1, 2],
            "dropout_dense": 2
        }
        network_2 = Conv1DEncoder(in_channels=inputs_2.size(1),
                                  in_features=inputs_2.size(2),
                                  hparams=hparams)
        # dropout-merge-dropout-conv-avgpool-dropout-flatten-
        # (Sequential(Linear,ReLU))-(Sequential(Linear,ReLU))-dropout-linear
        self.assertEqual(len(network_2.layers), 1 + 1 + 1 + 3 + 4 + 1)
        self.assertIsInstance(
            network_2.layer_by_name("MergeLayer"), MergeLayer)
        for layer in network_2.layers[1].layers:
            self.assertIsInstance(layer, nn.Sequential)

        outputs_2 = network_2(inputs_2)
        self.assertEqual(outputs_2.size(), torch.Size([128, 10]))
        self.assertEqual(outputs_2.size(-1), network_2.output_size)


if __name__ == "__main__":
    unittest.main()
