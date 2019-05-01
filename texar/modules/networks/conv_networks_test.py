#
"""
Unit tests for conv networks.
"""
import unittest
import torch
from torch import nn

import texar as tx
from texar.modules.networks.conv_networks import Conv1DNetwork


class Conv1DNetworkTest(unittest.TestCase):
    """Tests :class:`~texar.modules.Conv1DNetwork` class.
    """

    def test_feedforward(self):
        """Tests feed forward.
        """
        network_1 = Conv1DNetwork()
        self.assertEqual(len(network_1.layers), 4)
        self.assertTrue(isinstance(network_1.layer_by_name("MergeLayer"),
                                   tx.core.MergeLayer))
        for layer in network_1.layers[0].layers:
            self.assertTrue(isinstance(layer, nn.Sequential))

        inputs_1 = torch.ones([128, 32, 300])
        outputs_1 = network_1(inputs_1)
        self.assertEqual(outputs_1.shape, torch.Size([128, 256]))

        """hparams = {
            # Conv layers
            "num_conv_layers": 2,
            "filters": 128,
            "kernel_size": [[3, 4, 5], 4],
            "other_conv_kwargs": {"padding": "same"},
            # Pooling layers
            "pooling": "AveragePooling",
            "pool_size": 2,
            "pool_strides": 1,
            # Dense layers
            "num_dense_layers": 3,
            "dense_size": [128, 128, 10],
            "dense_activation": "relu",
            "other_dense_kwargs": {"use_bias": False},
            # Dropout
            "dropout_conv": [0, 1, 2],
            "dropout_dense": 2
        }
        network_2 = Conv1DNetwork(hparams)
        # nlayers = nconv-pool + nconv + npool + ndense + ndropout + flatten
        self.assertEqual(len(network_2.layers), 1+1+1+3+4+1)
        self.assertTrue(isinstance(network_2.layer_by_name("conv_pool_1"),
                                   tx.core.MergeLayer))
        for layer in network_2.layers[1].layers:
            self.assertTrue(isinstance(layer, tx.core.SequentialLayer))

        inputs_2 = tf.ones([64, 16, 300], tf.float32)
        outputs_2 = network_2(inputs_2)
        self.assertEqual(outputs_2.shape, [64, 10])"""


if __name__ == "__main__":
    unittest.main()
