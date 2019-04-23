"""
Unit tests for feed forward neural networks.
"""

import unittest

import torch

from texar.modules.networks.networks import FeedForwardNetwork

# pylint: disable=no-member, invalid-name

class FeedForwardNetworkTest(unittest.TestCase):
    """Tests the class
    :class:`~texar.modules.networks.networks.FeedForwardNetwork`.
    """

    def test_feedforward(self):
        """Tests feed-forward.
        """
        hparams = {
            "layers": [
                {
                    "type": "torch.nn.Linear",
                    "kwargs": {
                        "in_features": 32,
                        "out_features": 64
                    }
                },
                {
                    "type": "torch.nn.Linear",
                    "kwargs": {
                        "in_features": 64,
                        "out_features": 128
                    }
                }
            ]
        }

        nn = FeedForwardNetwork(hparams=hparams)

        self.assertEqual(len(nn.layers), len(hparams["layers"]))
        _ = nn(torch.ones(64, 16, 32))
        self.assertEqual(len(nn.trainable_variables),
                         len(hparams["layers"]) * 2)


if __name__ == "__main__":
    unittest.main()
